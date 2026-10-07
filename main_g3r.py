import os
import psutil
import argparse
from time import time
from tqdm import tqdm
import numpy as np
import torch

import wn_treecode

from g3r.harmonic_library import build_harmonic_library


def morton_order(points: np.ndarray, depth: int = 15) -> np.ndarray:
    """Return sorted-index -> original-index, preserving ties in input order.

    Each octant uses x + 2*y + 4*z, matching the existing CPU tree builder.
    Float64 quantization preserves split-plane decisions for float32 inputs.
    Points on the upper root boundary belong to the last cell.
    """
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if not 1 <= depth <= 21:
        raise ValueError("depth must be between 1 and 21 for uint64 codes")
    if not np.isfinite(points).all() or (np.abs(points) > 1).any():
        raise ValueError("points must be finite and inside [-1, 1]^3")
    cells = 1 << depth
    # Floor before adding the integer offset: adding 1 to a tiny negative
    # coordinate first can round it onto the x/y/z=0 splitting plane.
    grid = np.minimum(
        np.floor(points.astype(np.float64) * (cells / 2)) + cells // 2,
        cells - 1,
    ).astype(np.uint64)
    codes = np.zeros(len(points), dtype=np.uint64)
    for bit in range(depth):
        for axis in range(3):
            codes |= ((grid[:, axis] >> bit) & 1) << (3 * bit + axis)
    return np.argsort(codes, kind="stable")


time_start = time()

parser = argparse.ArgumentParser()
parser.add_argument('input', type=str, help='input point cloud file name, must have extension xyz/ply/obj/npy')
parser.add_argument('--epsilon', type=float, default=1e-3, help='epsilon for regularization, default 1e-3')
parser.add_argument(
    '--test_funcs',
    type=str,
    default='poly(:2)',
    help='flexibly define the list of test functions'
         'Syntax: "fund(10, 5.0); poly(:3); poly(4, -2:2:2)"'
)
parser.add_argument(
    '--iter',
    type=str,
    nargs='+',
    default=['10', 'sd', '20', 'cg'],
    help='Define the solver schedule. '
         'Example: --iter 10 sd 20 cg. '
         'Methods: sd (Steepest Descent), cg (Conjugate Gradient).'
)
parser.add_argument('--out_dir', type=str, default='results')
parser.add_argument('--cpu', action='store_true', help='use cpu code only')
parser.add_argument('--morton-order', action=argparse.BooleanOptionalAction, default=True,
                    help='sort points on CPU before solving (default: enabled)')
parser.add_argument('--tqdm', action='store_true', help='use tqdm bar')
parser.add_argument('--log', action='store_true', help='log the results to a file, default is not to log')
args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

normals_groundtruth = None

if os.path.splitext(args.input)[-1] == '.xyz':
    points_normals = np.loadtxt(args.input)
    points_unnormalized = points_normals[:, :3]
    normals_groundtruth = (points_normals[:, 3:6] / np.linalg.norm(points_normals[:, 3:6], axis=1, keepdims=True)
                           if points_normals.shape[1] > 3 else None)
    normals_groundtruth = torch.from_numpy(normals_groundtruth).float() if normals_groundtruth is not None else None
elif os.path.splitext(args.input)[-1] in ['.ply', '.obj']:
    import trimesh
    pcd = trimesh.load(args.input, process=False)
    points_unnormalized = np.array(pcd.vertices) # type: ignore
elif os.path.splitext(args.input)[-1] == '.npy':
    pcd = np.load(args.input)
    points_unnormalized = pcd[:, :3]
else:
    raise NotImplementedError('The input file must be have extension xyz/ply/obj/npy')

# Parse solver schedule
if len(args.iter) % 2 != 0:
    raise ValueError('--iter argument must have even number of entries, e.g., --iter 10 sd 20 cg')

schedule = []
total_iters = 0
valid_methods = {'sd', 'cg'}

for i in range(0, len(args.iter), 2):
    try:
        steps = int(args.iter[i])
        if steps <= 0:
            raise ValueError(f"Number of iterations must be positive, got {steps}")
    except ValueError:
        raise ValueError(f"Invalid number of iterations: {args.iter[i]}")

    method = args.iter[i+1].lower()
    if method not in valid_methods:
        raise ValueError(f"Unknown method: {method}. Must be one of {valid_methods}")
        
    schedule.append((method, steps))
    total_iters += steps

print(f"[LOG] Solver schedule: {schedule}")
print(f"[LOG] Total number of iterations: {total_iters}")

time_preprocess_start = time()

bbox_scale = 1.1
bbox_center = (points_unnormalized.min(0) + points_unnormalized.max(0)) / 2.
bbox_len = (points_unnormalized.max(0) - points_unnormalized.min(0)).max()
points_normalized = (points_unnormalized - bbox_center) * (2 / (bbox_len * bbox_scale))

morton_permutation = None
morton_sort_seconds = 0.0
if args.morton_order:
    morton_start = time()
    # Use the same float32 positions as the tree and CUDA kernels.
    points_normalized = points_normalized.astype(np.float32)
    morton_permutation = morton_order(points_normalized)
    points_normalized = points_normalized[morton_permutation]
    if normals_groundtruth is not None:
        normals_groundtruth = normals_groundtruth[torch.from_numpy(morton_permutation)]
    morton_sort_seconds = time() - morton_start

points_normalized = torch.from_numpy(points_normalized).contiguous().float()
normals = torch.zeros_like(points_normalized)

if not args.cpu:
    points_normalized = points_normalized.cuda()
    normals_groundtruth = normals_groundtruth.cuda() if normals_groundtruth is not None else None
    normals = normals.cuda()

wn_func = wn_treecode.WindingNumberTreecode(points_normalized)

print(f'[LOG] input: {args.input}, epsilon: {args.epsilon}, functions: {args.test_funcs}')

out_file_path = os.path.join(args.out_dir, os.path.basename(args.input)[:-4] + f'.xyz')
log_file_path = os.path.join(args.out_dir, os.path.basename(args.input)[:-4] + f'.log')
if args.log:
    header_string = [
        f"Hyperparameters:",
        f"Input File: {args.input}",
        f"Output Dir: {args.out_dir}",
        f"Iterations: {args.iter}",
        f"Epsilon: {args.epsilon}",
        f"Test Functions: {args.test_funcs}",
        f"CPU Only: {args.cpu}",
        f"Morton Order: {args.morton_order}",
        f"Iteration Log:\n"
    ]
    with open(log_file_path, 'w') as log_file:
        log_file.write("\n".join(header_string))

time_iter_start = time()

if args.epsilon > 0:
    epsilon = args.epsilon
else:
    epsilon = -args.epsilon * 2.0 / 2.0 ** wn_func.tree_depth

if wn_func.is_cuda:
    torch.cuda.synchronize(device=None)

test_functions = build_harmonic_library(
    args.test_funcs,
    epsilon=epsilon
)

test_specs = []
for tf in test_functions:
    tf_values = tf.eval_func(points_normalized)
    tf_grads = tf.eval_grad(points_normalized)
    tf_smoothed = tf.eval_smoothed(points_normalized)
    test_specs.append({
        "tf": tf,
        "values": tf_values,
        "grads": tf_grads,
        "smoothed": tf_smoothed,
    })


def AutAu_operator(v: torch.Tensor) -> torch.Tensor:
    prepared_nodes = wn_func.prepare_Au(v)
    result = torch.zeros_like(v)
    for spec in test_specs:
        values = wn_func.forward_Au(
            v, spec["values"], spec["grads"], spec["tf"], epsilon, prepared_nodes,
        )
        result.add_(wn_func.forward_AuT(
            values, spec["values"], spec["grads"], epsilon,
        ))
    return result

with torch.no_grad():
    AuT_u_hat = torch.zeros_like(normals)
    for spec in test_specs:
        AuT_u_hat.add_(wn_func.forward_AuT(
            spec["smoothed"], spec["values"], spec["grads"], epsilon,
        ))
    b = AuT_u_hat / 2.0

    r = b - AutAu_operator(normals)
    
    max_iters = total_iters
    pbar = tqdm(total=max_iters, desc="Solver Schedule", ncols=100, disable=not args.tqdm)

    p = torch.empty_like(r)
    
    global_iter = 0
    cg_initialized = False

    for method, steps in schedule:
        for local_iter in range(steps):
            
            if method == 'sd':
                # --- Steepest Descent (SD) ---
                pbar.set_description(f"SD Solve ({local_iter+1}/{steps})")
                
                AutAu_r = AutAu_operator(r)
                alpha = (r * r).sum() / (r * AutAu_r).sum().clamp(min=1e-10)
                
                normals = normals + alpha * r
                r = r - alpha * AutAu_r

                # recompute residual every 5 iterations to avoid accumulation of numerical errors
                if local_iter % 5 == 0 and local_iter > 0:
                    r = b - AutAu_operator(normals)
                
                cg_initialized = False

            elif method == 'cg':
                # --- Conjugate Gradient (CG) ---
                if not cg_initialized:
                    pbar.set_description(f"CG Init ({local_iter+1}/{steps})")
                    p = r.clone()
                    cg_initialized = True
                
                pbar.set_description(f"CG Solve ({local_iter+1}/{steps})")

                # recompute residual every 5 iterations to avoid accumulation of numerical errors
                if local_iter % 5 == 0 and local_iter > 0:
                    r = b - AutAu_operator(normals)
                    p = r.clone()

                AutAu_p = AutAu_operator(p)
                rr_dot = (r * r).sum()
                
                alpha = rr_dot / (p * AutAu_p).sum().clamp(min=1e-10)

                normals = normals + alpha * p
                r_new = r - alpha * AutAu_p

                beta = (r_new * r_new).sum() / rr_dot.clamp(min=1e-10)
                p = r_new + beta * p

                r = r_new

            r_norm = r.norm().item()
            pbar.set_postfix(res=f"{r_norm:.4e}")

            if normals_groundtruth is not None:
                normals_normalized = normals / torch.linalg.norm(normals, dim=-1, keepdim=True)
                dot_prod = torch.einsum('ij,ij->i', normals_normalized, normals_groundtruth)
                mean_angle_error = (1 - dot_prod).mean().item() / 2
                pco = (torch.sum(dot_prod > 0) / dot_prod.shape[0]) * 100
                if args.log:
                    with open(log_file_path, 'a') as log_file:
                        log_file.write(f"Iter {global_iter+1}/{max_iters}, Method: {method.upper()}, Residual: {r_norm:.4e}, Mean Angle Error: {mean_angle_error:.4f}, PCO: {pco:.4f}\n")
            else:
                mean_angle_error = None
            
            global_iter += 1
            pbar.update(1)

    pbar.close()

    out_normals = normals
if wn_func.is_cuda:
    torch.cuda.synchronize(device=None)

time_iter_end = time()
print(f'[LOG] time_preproc: {time_iter_start - time_preprocess_start}')
print(f'[LOG] time_morton_sort: {morton_sort_seconds}')
print(f'[LOG] time_main: {time_iter_end - time_iter_start}')

with torch.no_grad():
    out_normals_numpy = out_normals.detach().cpu().numpy()
    if morton_permutation is not None:
        restored_normals = np.empty_like(out_normals_numpy)
        restored_normals[morton_permutation] = out_normals_numpy
        out_normals_numpy = restored_normals
    out_points_normals = np.concatenate([points_unnormalized, out_normals_numpy], -1) # type: ignore
    np.savetxt(out_file_path, out_points_normals)

process = psutil.Process(os.getpid())
mem_info = process.memory_info()    # bytes
mem = mem_info.rss
if wn_func.is_cuda:
    # gpu_mem = torch.cuda.mem_get_info(0)[1]-torch.cuda.mem_get_info(0)[0]
    gpu_mem = torch.cuda.max_memory_allocated()
    mem += gpu_mem
print('[LOG] mem:', mem / 1024/1024)     # megabytes
