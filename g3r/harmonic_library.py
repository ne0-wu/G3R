import math
import torch

from .test_functions import (
    FundamentalSolution,
    RegularSolidHarmonic,
)


def fibonacci_lattice(samples: int = 10):
    idx = torch.arange(0, samples, dtype=torch.float32)

    y = 1.0 - (idx + 0.5) / samples * 2.0
    radius = torch.sqrt(torch.clamp(1.0 - y * y, 0.0, 1.0))
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    theta = idx * phi * (2.0 * math.pi)  # Use phi for golden angle
    x = radius * torch.cos(theta)
    z = radius * torch.sin(theta)

    return torch.stack((x, y, z), dim=-1)


def build_harmonic_lib_fundsol(
    num_funcs: int,
    radius: float,
    epsilon: float,
) -> list[FundamentalSolution]:
    """Generate fundamental solution test functions on a sphere."""

    source_points = fibonacci_lattice(samples=num_funcs)
    functions = []
    for source in source_points:
        functions.append(FundamentalSolution(source * radius, epsilon))
    return functions


def build_harmonic_lib_poly(
    max_l: int,
    epsilon: float,
) -> list[RegularSolidHarmonic]:
    """Generate regular solid harmonic test functions up to degree ``max_l``."""

    functions = []
    for l in range(max_l + 1):
        for m in range(-l, l + 1):
            functions.append(RegularSolidHarmonic(l, m, epsilon))

    num_funcs = len(functions)
    assert num_funcs == (max_l + 1) ** 2, f"Expected {(max_l + 1)**2} functions, got {num_funcs}"

    return functions


def build_harmonic_library(
    test_func_specs: str,
    epsilon: float,
) -> list:
    """
    - "fund(num, radius)"
    - "poly(l_spec)"
    - "poly(l, m_spec)"

    Spec syntax:
    - "N" -> [N]
    - ":N" -> [0, 1, ..., N]
    - "N1:N2" -> [N1, N1+1, ..., N2]
    - "N1:S:N2" -> [N1, N1+S, N1+2S, ..., (<=N2)]
    """

    def parse_range_spec(spec: str) -> list[int]:
        parts = [p.strip() for p in spec.split(":")]

        try:
            if len(parts) == 1:  # "N"
                val = int(parts[0])
                return [val]

            if len(parts) == 2:  # ":N" or "N1:N2"
                start_str, stop_str = parts
                start = 0 if not start_str else int(start_str)
                stop = int(stop_str)
                return list(range(start, stop + 1, 1))

            if len(parts) == 3:  # "N1:S:N2"
                start_str, step_str, stop_str = parts

                start = 0 if not start_str else int(start_str)
                step = int(step_str)
                stop = int(stop_str)

                if step == 0:
                    raise ValueError("Step cannot be zero.")
                return list(range(start, stop + 1, step))

        except Exception as e:
            raise ValueError(f"Invalid range spec '{spec}': {e}")

        raise ValueError(f"Invalid range spec format: {spec}")

    test_functions = []

    added_harmonics = set()

    specs = test_func_specs.split(";")

    for spec in specs:
        spec = spec.strip()
        if not spec:
            continue

        try:
            # --- fund(num, radius) ---
            if spec.startswith("fund(") and spec.endswith(")"):
                params_str = spec[5:-1]
                params = [p.strip() for p in params_str.split(",")]
                if len(params) != 2:
                    raise ValueError(f"fund() requires 2 arguments, got {len(params)}")
                num_funcs = int(params[0])
                radius = float(params[1])
                test_functions.extend(
                    build_harmonic_lib_fundsol(num_funcs, radius, epsilon)
                )

            # --- poly(l_spec) or poly(l, m_spec) ---
            elif spec.startswith("poly(") and spec.endswith(")"):
                params_str = spec[5:-1]
                params = [p.strip() for p in params_str.split(",")]

                # Case 1: poly(l_spec) -> e.g. poly(3), poly(:3), poly(3:4)
                if len(params) == 1:
                    l_spec = params[0]
                    l_values = parse_range_spec(l_spec)

                    for l in l_values:
                        if l < 0:
                            raise ValueError(f"l must be >= 0, got {l}")
                        # Add all m
                        for m in range(-l, l + 1):
                            if (l, m) not in added_harmonics:
                                test_functions.append(
                                    RegularSolidHarmonic(l, m, epsilon)
                                )
                                added_harmonics.add((l, m))

                # Case 2: poly(l, m_spec) -> e.g. poly(3, 1), poly(4, -1:2:4)
                elif len(params) == 2:
                    l_str, m_spec = params

                    # l must be a single integer
                    try:
                        l = int(l_str)
                    except ValueError:
                        raise ValueError(
                            f"In poly(l, m_spec), l must be a single integer, not '{l_str}'"
                        )
                    if l < 0:
                        raise ValueError(f"l must be >= 0, got {l}")

                    m_values = parse_range_spec(m_spec)

                    for m in m_values:
                        if abs(m) > l:
                            print(f"[WARN] Skipping poly(l={l}, m={m}) because |m| > l.")
                            continue
                        if (l, m) not in added_harmonics:
                            test_functions.append(RegularSolidHarmonic(l, m, epsilon))
                            added_harmonics.add((l, m))

                else:
                    raise ValueError(f"poly() requires 1 or 2 arguments, got {len(params)}")

            else:
                print(f"[WARN] Unrecognized test function spec: {spec}")

        except Exception as e:
            print(f"[ERROR] Failed to parse '{spec}': {e}")
            raise

    if not test_functions:
        raise ValueError("No test functions could be constructed.")

    print(f"[LOG] A total of {len(test_functions)} test functions were constructed.")

    return test_functions
