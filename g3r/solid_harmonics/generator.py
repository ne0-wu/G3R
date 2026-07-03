import argparse
from pathlib import Path

import sympy
from sympy.functions.special.spherical_harmonics import Ynm

DEFAULT_OUTPUT = Path(__file__).with_name("data.py")


def _validate_indices(l: int, m: int) -> None:
    if not isinstance(l, int) or l < 0:
        raise ValueError("Degree 'l' must be a non-negative integer.")
    if not isinstance(m, int) or abs(m) > l:
        raise ValueError("Order 'm' must satisfy -l <= m <= l.")


def regular_solid_harmonic(l: int, m: int, vars):
    """Return the real regular solid harmonic polynomial for (l, m)."""

    _validate_indices(l, m)

    x, y, z = vars
    theta, phi = sympy.symbols("theta phi", real=True)
    r = sympy.sqrt(x**2 + y**2 + z**2)
    r_xy = sympy.sqrt(x**2 + y**2)

    ylm_raw = Ynm(l, abs(m), theta, phi)
    ylm_spherical = ylm_raw.doit() # pyright: ignore[reportAttributeAccessIssue]
    ylm_expanded = ylm_spherical.expand(complex=True).expand(trig=True)

    solid_harmonic_expanded = r**l * ylm_expanded

    substitutions = {
        sympy.sin(theta): r_xy / r,  # pyright: ignore[reportOperatorIssue]
        sympy.Abs(sympy.sin(theta)): r_xy / r, # pyright: ignore[reportOperatorIssue]
        sympy.cos(theta): z / r,
        sympy.sin(phi): y / r_xy,
        sympy.cos(phi): x / r_xy,
    }

    poly_complex = solid_harmonic_expanded.subs(substitutions)
    if m != 0:
        poly_complex = poly_complex * sympy.sqrt(2)
    poly_complex = sympy.expand(sympy.simplify(poly_complex))

    if m >= 0:
        return sympy.re(poly_complex)
    return sympy.im(poly_complex)


def _polynomial_terms(expr, variables) -> list[tuple[float, int, int, int]]:
    poly = sympy.Poly(expr, *variables)
    terms: list[tuple[float, int, int, int]] = []
    for (i, j, k), coeff in poly.terms():
        coeff_val = float(sympy.N(coeff, 25))
        if abs(coeff_val) < 1e-18:
            continue
        terms.append((coeff_val, int(i), int(j), int(k)))
    return terms


def generate_python_module(max_l: int, output_path: Path, precision: int = 18) -> None:
    """Emit a Python module with polynomial coefficients up to ``max_l``."""

    x, y, z = sympy.symbols("x y z", real=True)
    variables = (x, y, z)

    lines: list[str] = []
    lines.append('"""Auto-generated solid harmonic coefficients."""\n')
    lines.append(
        "# Do not edit by hand – run `python -m g3r.solid_harmonics.generator --max-l {max_l}` instead.\n".format(
            max_l=max_l
        )
    )
    lines.append("from __future__ import annotations\n\n")
    lines.append(f"MAX_L = {max_l}\n\n")
    lines.append("COEFFICIENTS = {\n")

    for l_val in range(max_l + 1):
        for m_val in range(-l_val, l_val + 1):
            expr = regular_solid_harmonic(l_val, m_val, variables)
            terms = _polynomial_terms(expr, variables)
            lines.append(f"    ({l_val}, {m_val}): [\n")
            for coeff, i, j, k in terms:
                coeff_str = f"{coeff:.{precision}e}"
                lines.append(f"        ({coeff_str}, {i}, {j}, {k}),\n")
            lines.append("    ],\n")
    lines.append("}\n")

    output_path.write_text("".join(lines))
    print(f"[INFO] Wrote coefficients for MAX_L={max_l} to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_l", type=int, default=7)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--precision", type=int, default=18)
    args = parser.parse_args()

    generate_python_module(args.max_l, args.output, args.precision)
