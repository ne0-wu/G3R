import argparse
import os
import time

import numpy as np
from scipy.special import erf

import sympy
from sympy.functions.special.spherical_harmonics import Ynm


def fundamental_solution(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    y = np.asarray(y)
    r = np.linalg.norm(x[:, np.newaxis, :] - y[np.newaxis, :, :], axis=-1)
    r = np.maximum(r, 1e-10)
    return 1 / (4 * np.pi * r)


def fundamental_solution_grad_y(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    y = np.asarray(y)
    r = np.linalg.norm(x[:, np.newaxis, :] - y[np.newaxis, :, :], axis=-1)
    r = np.maximum(r, 1e-10)
    diff = x[:, np.newaxis, :] - y[np.newaxis, :, :]
    return 1 / (4 * np.pi) * diff / (r[..., np.newaxis] ** 3)


def fundsol_eps(x: np.ndarray, y: np.ndarray, epsilon: float) -> np.ndarray:
    x = np.asarray(x)
    y = np.asarray(y)
    r = np.linalg.norm(x[:, np.newaxis, :] - y[np.newaxis, :, :], axis=-1)
    r = np.maximum(r, 1e-10)
    return erf(r / (np.sqrt(2) * epsilon)) / (4 * np.pi * r)


def fundsol_eps_grad_y(x: np.ndarray, y: np.ndarray, epsilon: float) -> np.ndarray:
    x = np.asarray(x)
    y = np.asarray(y)
    diff = x[:, np.newaxis, :] - y[np.newaxis, :, :]
    r = np.linalg.norm(diff, axis=-1)
    r = np.maximum(r, 1e-10)

    # Compute the regularizer Regularizer3d[r/ε] = -E^(-(r/ε)²/2) * √(2/π) * (r/ε) + Erf[(r/ε)/√2]
    r_scaled = r / epsilon
    exp_term = -np.exp(-(r_scaled**2) / 2) * np.sqrt(2 / np.pi) * r_scaled
    erf_term = erf(r_scaled / np.sqrt(2))
    regularizer = exp_term + erf_term

    # Compute the Poisson kernel part: 1/(4π) * (x-y) / ||x-y||³
    poisson_kernel = 1 / (4 * np.pi) * diff / (r[..., np.newaxis] ** 3)

    # Combine: Regularizer * PoissonKernel
    return regularizer[..., np.newaxis] * poisson_kernel


def real_solid_harmonic(l: int, m: int, vars):
    x, y, z = vars
    if not isinstance(l, int) or l < 0:
        raise ValueError("Degree 'l' must be a non-negative integer.")
    if not isinstance(m, int) or abs(m) > l:
        raise ValueError("Order 'm' must satisfy -l <= m <= l.")

    theta, phi = sympy.symbols("theta phi", real=True)
    r = sympy.sqrt(x**2 + y**2 + z**2)
    r_xy = sympy.sqrt(x**2 + y**2)

    ylm_spherical = Ynm(l, abs(m), theta, phi).doit()  # type: ignore
    ylm_expanded = ylm_spherical.expand(complex=True).expand(trig=True)

    solid_harmonic_expanded = r**l * ylm_expanded

    substitutions = {
        sympy.sin(theta): r_xy / r,  # type: ignore
        sympy.Abs(sympy.sin(theta)): r_xy / r,  # type: ignore
        sympy.cos(theta): z / r,
        sympy.sin(phi): y / r_xy,
        sympy.cos(phi): x / r_xy,
    }

    poly_complex = solid_harmonic_expanded.subs(substitutions)
    poly_complex = sympy.simplify(poly_complex).n()

    if m >= 0:
        return sympy.re(poly_complex)
    else:
        return sympy.im(poly_complex)


class TestFunction:
    def __init__(self, name: str, epsilon: float) -> None:
        self.name = name
        self.epsilon = float(epsilon)

    def eval_func(self, points: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def eval_grad(self, points: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def eval_smoothed(self, points: np.ndarray, epsilon: float) -> np.ndarray:
        raise NotImplementedError


class RegularSolidHarmonic(TestFunction):
    def __init__(self, l: int, m: int, epsilon: float) -> None:
        name = f"solid_harmonic_l{l}_m{m}"
        super().__init__(name=name, epsilon=epsilon)
        self.l = int(l)
        self.m = int(m)
        x_sym, y_sym, z_sym = sympy.symbols("x y z", real=True)
        expr = real_solid_harmonic(self.l, self.m, (x_sym, y_sym, z_sym))
        grad_exprs = [sympy.diff(expr, var) for var in (x_sym, y_sym, z_sym)]
        self.value_fn = sympy.lambdify((x_sym, y_sym, z_sym), expr, modules="numpy")
        self.grad_fn = sympy.lambdify(
            (x_sym, y_sym, z_sym), grad_exprs, modules="numpy"
        )

    def eval_func(self, points: np.ndarray) -> np.ndarray:
        x_vals, y_vals, z_vals = points[:, 0], points[:, 1], points[:, 2]
        values = self.value_fn(x_vals, y_vals, z_vals)
        return np.broadcast_to(np.asarray(values, dtype=np.float64), (points.shape[0],))

    def eval_grad(self, points: np.ndarray) -> np.ndarray:
        x_vals, y_vals, z_vals = points[:, 0], points[:, 1], points[:, 2]
        size = points.shape[0]
        grad_components = self.grad_fn(x_vals, y_vals, z_vals)

        if isinstance(grad_components, (tuple, list)):
            return np.stack(
                [
                    np.broadcast_to(np.asarray(comp, dtype=np.float64), (size,))
                    for comp in grad_components
                ],
                axis=1,
            )

        grad_array = np.asarray(grad_components, dtype=np.float64)
        if grad_array.ndim == 1:
            grad_array = np.tile(grad_array.reshape(-1, 1), (1, size)).T

        if grad_array.shape[0] == 3 and grad_array.shape[1] == size:
            return grad_array.T
        elif grad_array.shape == (size, 3):
            return grad_array
        else:
            raise ValueError(f"Unexpected gradient shape: {grad_array.shape}")

    def eval_smoothed(self, points: np.ndarray, epsilon: float) -> np.ndarray:
        return self.eval_func(points)


def _parse_range_spec(spec: str) -> list[int]:
    parts = [p.strip() for p in spec.split(":")]

    if len(parts) == 1:
        return [int(parts[0])]

    if len(parts) == 2:
        start_str, stop_str = parts
        start = 0 if not start_str else int(start_str)
        stop = int(stop_str)
        return list(range(start, stop + 1))

    if len(parts) == 3:
        start_str, step_str, stop_str = parts
        start = 0 if not start_str else int(start_str)
        step = int(step_str)
        stop = int(stop_str)
        if step == 0:
            raise ValueError("Step cannot be zero.")
        return list(range(start, stop + 1, step))

    raise ValueError(f"Invalid range spec format: {spec}")


def build_harmonic_library(test_func_specs: str, epsilon: float):
    test_functions = []
    added_harmonics = set()

    specs = test_func_specs.split(";")
    for spec in specs:
        spec = spec.strip()
        if not spec:
            continue

        if spec.startswith("poly(") and spec.endswith(")"):
            params_str = spec[5:-1]
            params = [p.strip() for p in params_str.split(",") if p.strip()]

            # poly(l_spec)
            if len(params) == 1:
                l_values = _parse_range_spec(params[0])
                added_count = 0
                for l in l_values:
                    if l < 0:
                        raise ValueError(f"l must be >= 0, got {l}")
                    for m in range(-l, l + 1):
                        if (l, m) not in added_harmonics:
                            test_functions.append(RegularSolidHarmonic(l, m, epsilon))
                            added_harmonics.add((l, m))
                            added_count += 1

            # poly(l, m_spec)
            elif len(params) == 2:
                l_val = int(params[0])
                if l_val < 0:
                    raise ValueError(f"l must be >= 0, got {l_val}")
                m_values = _parse_range_spec(params[1])
                added_count = 0
                for m in m_values:
                    if (l_val, m) not in added_harmonics:
                        test_functions.append(RegularSolidHarmonic(l_val, m, epsilon))
                        added_harmonics.add((l_val, m))
                        added_count += 1

            else:
                raise ValueError(f"poly() requires 1 or 2 arguments, got {len(params)}")

        else:
            raise ValueError(f"Unsupported test function spec '{spec}'.")

    if not test_functions:
        raise ValueError("No test functions could be constructed.")

    return test_functions


def green_representation_coefficients(
    x: np.ndarray,
    y: np.ndarray,
    harmonic_func,
    harmonic_grad_func,
    fundamental_solution,
    fundamental_solution_grad_y,
) -> np.ndarray:
    # Single layer contribution: Phi(x,y) * grad_u(y) * ay
    single_layer_coeff = (
        fundamental_solution(x, y)[:, :, np.newaxis]  # (n, m, 1)
        * harmonic_grad_func(y)[np.newaxis, :, :]  # (1, m, 3)
    )

    # Double layer contribution: -u(y) * grad_y_Phi(x,y) * ay
    double_layer_coeff = -harmonic_func(y)[
        np.newaxis, :, np.newaxis
    ] * fundamental_solution_grad_y(  # (1, m, 1)
        x, y
    )  # (n, m, 3)

    return single_layer_coeff + double_layer_coeff


def reconstruct_normals(points: np.ndarray, harmonic_library, epsilon: float):
    start_time = time.time()
    print(f"Building linear system for {len(harmonic_library)} harmonic functions...")

    num_cols = 3 * points.shape[0]
    A = np.zeros((num_cols, num_cols))
    b = np.zeros((num_cols,))

    for test_func in harmonic_library:
        coeffs = green_representation_coefficients(
            points,
            points,
            lambda x: test_func.eval_func(x),
            lambda x: test_func.eval_grad(x),
            lambda x, y: fundsol_eps(x, y, epsilon),
            lambda x, y: fundsol_eps_grad_y(x, y, epsilon),
        )
        Au = coeffs.reshape(coeffs.shape[0], -1)
        A += Au.T @ Au
        u_hat = test_func.eval_smoothed(points, epsilon)
        b += Au.T @ u_hat / 2.0

    print(
        f"Built linear system A shape: {A.shape}, b shape: {b.shape} in {time.time() - start_time:.2f} s"
    )

    start_time = time.time()
    print("Solving the linear system...")

    normals = np.linalg.solve(A, b).reshape(points.shape)
    print(f"Solved the linear system in {time.time() - start_time:.2f} s")

    return normals


def angle_difference(estimated: np.ndarray, groundtruth: np.ndarray):
    estimated = estimated / np.linalg.norm(estimated, axis=1, keepdims=True)
    groundtruth = groundtruth / np.linalg.norm(groundtruth, axis=1, keepdims=True)

    return np.arccos(np.clip(np.einsum("ij,ij->i", estimated, groundtruth), -1.0, 1.0))


def main():
    parser = argparse.ArgumentParser(
        description="Green's 3rd Identity Normal Reconstruction with Flexible Test Functions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use solid harmonics up to degree 2
    %(prog)s data.xyz --test_funcs "poly(:2)" --epsilon 5e-4
  
  # Use specific harmonic orders
    %(prog)s data.xyz --test_funcs "poly(3, -1:1)" --epsilon 5e-4

Test Function Specification Syntax:
    poly(l_spec)       - Generate solid harmonics for degrees in l_spec
    poly(l, m_spec)    - Generate solid harmonics for degree l and orders in m_spec
  
Range Specifications:
  N       - Single value [N]
  :N      - Range [0, 1, ..., N]
  N1:N2   - Range [N1, N1+1, ..., N2]
  N1:S:N2 - Range [N1, N1+S, ..., ≤N2] with step S
        """,
    )

    parser.add_argument(
        "input", type=str, help="Input point cloud file path (.xyz format)"
    )

    parser.add_argument(
        "--test_funcs",
        type=str,
        default="poly(:1)",
        help='Test function specification (solid harmonics only). Examples: "poly(:3)", "poly(3, -1:1)"',
    )

    parser.add_argument(
        "--epsilon",
        type=float,
        default=5e-4,
        help="Regularization parameter (default: 5e-4)",
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="results",
        help="Output directory (default: results)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("G3R: Green's 3rd Identity Normal Reconstruction")
    print("=" * 70)
    print(f"Input file: {args.input}")
    print(f"Test functions: {args.test_funcs}")
    print(f"Epsilon: {args.epsilon}")
    print("=" * 70)

    # Load points from the input file
    try:
        points_original = np.loadtxt(args.input, usecols=(0, 1, 2), dtype=np.float64)
    except Exception as e:
        print(f"Error: Unable to read file '{args.input}': {e}")
        return 1

    print(f"\nLoaded {len(points_original)} points")

    # Normalize points to [-1, 1]^3
    min_points = np.min(points_original, axis=0)
    max_points = np.max(points_original, axis=0)
    bbox_size = (max_points - min_points).max()
    points_normalized = (points_original - min_points) / bbox_size * 2 - 1

    print("Generating test function library...")
    harmonic_library = build_harmonic_library(args.test_funcs, args.epsilon)

    print("Reconstructing normals...")
    normals = reconstruct_normals(points_normalized, harmonic_library, args.epsilon)

    # Calculate angle differences if ground truth is available
    try:
        normals_gt = np.loadtxt(args.input, usecols=(3, 4, 5), dtype=np.float64)
        angle_diffs = angle_difference(normals, normals_gt)
        mean_angle_diff = np.mean(angle_diffs)
        median_angle_diff = np.median(angle_diffs)

        print("\n" + "=" * 70)
        print("Evaluation Metrics (compared to ground truth):")
        print("=" * 70)
        print(f"Mean angle difference:   {mean_angle_diff * 180 / np.pi:.4f} degrees")
        print(f"Median angle difference: {median_angle_diff * 180 / np.pi:.4f} degrees")
        print(
            f"Max angle difference:    {np.max(angle_diffs) * 180 / np.pi:.4f} degrees"
        )

        # Correct orientation ratio
        correct_ratio = np.mean(angle_diffs < np.pi / 2) * 100
        print(f"Correct orientation:     {correct_ratio:.2f}%")
        print("=" * 70)
    except Exception:
        print("\nNo ground truth normals available for comparison.")

    # Save results
    os.makedirs(args.out_dir, exist_ok=True)
    output_path = os.path.join(
        args.out_dir, os.path.splitext(os.path.basename(args.input))[0] + ".xyz"
    )

    # Save with original coordinates
    results = np.hstack((points_original, normals))
    np.savetxt(output_path, results, fmt="%.6f")
    print(f"\nResults saved to: {output_path}")

    return 0


if __name__ == "__main__":
    exit(main())
