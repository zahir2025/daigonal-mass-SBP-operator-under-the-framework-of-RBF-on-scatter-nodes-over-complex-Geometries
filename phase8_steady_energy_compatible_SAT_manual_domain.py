"""
Phase 8 - Steady Advection Compatible-SAT Energy Diagnostics
============================================================

Manual workflow:
    1. Change DOMAIN_TYPE below to one of:
           "annulus"
           "box_minus_circle"
           "box_minus_airfoil"
    2. Optionally edit NODE_REFINEMENTS_BY_DOMAIN for that domain.
    3. Run this file.

This script reuses the Phase 3c Path A diagonal-mass SBP-SAT construction and
computes energy diagnostics for the steady advection MMS:

    lambda . grad(u) = f
    lambda = [1, 1]
    u_exact(x,y) = exp(x+y)
    f(x,y) = 2 exp(x+y)

Energy quantities:

    E_exact_ref = 0.5 * integral_Omega u_exact^2 dOmega
        High-order reference integral over the physical domain.

    E_exact_ML = 0.5 * sum_i w_i u_exact(x_i)^2
        Exact solution measured with the lumped mass matrix.

    E_num_ML = 0.5 * sum_i w_i u_h(x_i)^2
        Numerical steady compatible-SAT solution measured with M_L.

The table separates the exact lumped-mass quadrature energy error from the
numerical-solution energy error.

Outputs:
    outputs_phase8_steady_energy/
        phase8_energy_<domain>_raw.csv
        phase8_energy_<domain>_summary.csv
        phase8_energy_<domain>_values_vs_N.png
        phase8_energy_<domain>_relative_error_vs_N.png
        phase8_energy_<domain>_relative_error_vs_h.png
        phase8_energy_<domain>_energy_vs_solution_error.png
"""

from pathlib import Path
import csv
import math
from datetime import datetime

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shapely.ops import triangulate, unary_union

import phase3c_pathA_minimal_SAT_steady_MMS as steady


# ============================================================
# Manual configuration
# ============================================================

# Change this manually, then rerun the script.
DOMAIN_TYPE = "annulus"

POLY_DEGREES = list(range(1, 8))

# Keep the coarsest N comfortably larger than the p=7 stencil size.
NODE_REFINEMENTS_BY_DOMAIN = {
    "annulus": [400, 800, 1600, 3200],
    "box_minus_circle": [260, 380, 560, 820],
    "box_minus_airfoil": [450, 650, 950, 1350],
}

PHS_ORDER = 5
STENCIL_FACTOR = 3.5
STENCIL_MINIMUM = 25

# Full compatible SAT. Change to 0.5 to study the minimal stable SAT.
SAT_THETA = 1.0

# Gauss-Legendre order used in the high-order reference domain integral.
REFERENCE_QUAD_ORDER = 16

FIG_DPI = 180

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs_phase8_steady_energy"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


RAW_FIELDS = [
    "status",
    "message",
    "domain",
    "p",
    "N",
    "h",
    "theta",
    "Np",
    "Nc",
    "phs_order",
    "E_exact_ref",
    "E_exact_ML",
    "E_num_ML",
    "energy_quad_abs_error",
    "energy_quad_rel_error",
    "energy_num_abs_error",
    "energy_num_rel_error",
    "energy_solution_abs_error",
    "energy_solution_rel_error",
    "solution_rel_error",
    "linear_residual",
    "observed_energy_rate",
    "min_eig_H",
    "SBP_x",
    "SBP_y",
    "poly_x",
    "poly_y",
    "Qx_relative_change",
    "Qy_relative_change",
    "min_diag",
    "max_diag",
    "mass_condition",
    "area_error",
    "union_error",
    "reference_area",
    "reference_area_error",
    "max_fd_cond",
]

SUMMARY_FIELDS = [
    "domain",
    "p",
    "theta",
    "num_successful_refinements",
    "N_coarsest",
    "N_finest",
    "h_coarsest",
    "h_finest",
    "E_exact_ref",
    "E_num_finest",
    "energy_error_finest",
    "solution_error_finest",
    "last_step_energy_rate",
    "fitted_energy_order",
    "max_SBP_x",
    "max_poly_x",
    "max_linear_residual",
    "max_fd_cond",
]


# ============================================================
# Generic helpers
# ============================================================

def finite_positive(value):
    return np.isfinite(value) and value > 0.0


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return np.nan


def convergence_rate(err_coarse, err_fine, h_coarse, h_fine):
    if not (
        finite_positive(err_coarse)
        and finite_positive(err_fine)
        and finite_positive(h_coarse)
        and finite_positive(h_fine)
        and h_coarse != h_fine
    ):
        return np.nan

    return math.log(err_coarse / err_fine) / math.log(h_coarse / h_fine)


def fitted_energy_order(rows):
    good = [
        r for r in rows
        if r["status"] == "ok"
        and finite_positive(safe_float(r["h"]))
        and finite_positive(safe_float(r["energy_num_rel_error"]))
    ]

    if len(good) < 2:
        return np.nan

    h = np.array([safe_float(r["h"]) for r in good], dtype=float)
    err = np.array([safe_float(r["energy_num_rel_error"]) for r in good], dtype=float)

    slope, _ = np.polyfit(np.log(h), np.log(err), 1)
    return float(slope)


def write_csv(path, rows, fieldnames):
    try:
        with open(path, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        return path

    except PermissionError:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback = path.with_name(f"{path.stem}_{stamp}{path.suffix}")

        with open(fallback, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"WARNING: could not overwrite locked CSV -> {path}")
        print(f"         saved this run instead as      -> {fallback}")

        return fallback


def save_figure(fig, path):
    try:
        fig.savefig(path, dpi=FIG_DPI)

        return path

    except PermissionError:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback = path.with_name(f"{path.stem}_{stamp}{path.suffix}")
        fig.savefig(fallback, dpi=FIG_DPI)

        print(f"WARNING: could not overwrite locked figure -> {path}")
        print(f"         saved this run instead as        -> {fallback}")

        return fallback


def ml_energy(u, weights):
    return 0.5 * float(np.sum(weights * u * u))


# ============================================================
# Reference energy integral
# ============================================================

def triangle_integral_exp2(vertices, quad_order):
    """
    Integrate exp(2x+2y) over one triangle by tensor Gauss-Legendre with a
    Duffy transform from the unit square to the reference triangle.
    """
    nodes, weights = np.polynomial.legendre.leggauss(quad_order)
    nodes = 0.5 * (nodes + 1.0)
    weights = 0.5 * weights

    v0 = vertices[0]
    v1 = vertices[1]
    v2 = vertices[2]

    e1 = v1 - v0
    e2 = v2 - v0
    det = abs(e1[0] * e2[1] - e1[1] * e2[0])

    total = 0.0

    for i, u in enumerate(nodes):
        one_minus_u = 1.0 - u

        for j, v in enumerate(nodes):
            xi = u
            eta = one_minus_u * v

            xy = v0 + xi * e1 + eta * e2
            total += weights[i] * weights[j] * one_minus_u * math.exp(
                2.0 * xy[0] + 2.0 * xy[1]
            )

    return det * total


def polygon_to_interior_triangles(poly):
    triangles = []

    for tri in triangulate(poly):
        clipped = tri.intersection(poly)

        for part in steady.geometry_parts(clipped):
            if part.area <= 1.0e-14:
                continue

            # triangulate(part) is used so non-triangular clipped polygons are
            # reduced to triangles before quadrature.
            for subtri in triangulate(part):
                piece = subtri.intersection(part)

                for final in steady.geometry_parts(piece):
                    if final.area <= 1.0e-14:
                        continue

                    coords = np.asarray(final.exterior.coords[:-1], dtype=float)

                    if coords.shape[0] == 3:
                        triangles.append(coords)
                    else:
                        # A rare clipped piece may still have more than three
                        # vertices; triangulate once more and keep the pieces.
                        for final_tri in triangulate(final):
                            final_piece = final_tri.intersection(final)
                            for tri_part in steady.geometry_parts(final_piece):
                                if tri_part.area <= 1.0e-14:
                                    continue
                                tri_coords = np.asarray(
                                    tri_part.exterior.coords[:-1],
                                    dtype=float,
                                )
                                if tri_coords.shape[0] == 3:
                                    triangles.append(tri_coords)

    return triangles


def reference_exact_energy(domain, quad_order):
    triangles = []

    for poly in steady.geometry_parts(domain):
        triangles.extend(polygon_to_interior_triangles(poly))

    area_quad = 0.0
    integral_u2 = 0.0

    for vertices in triangles:
        v0, v1, v2 = vertices
        det = abs(np.cross(v1 - v0, v2 - v0))
        area_quad += 0.5 * det
        integral_u2 += triangle_integral_exp2(vertices, quad_order)

    E_exact_ref = 0.5 * integral_u2
    area_error = abs(area_quad - domain.area) / max(abs(domain.area), 1.0e-14)

    return {
        "E_exact_ref": float(E_exact_ref),
        "area_quad": float(area_quad),
        "area_error": float(area_error),
        "num_triangles": int(len(triangles)),
    }


# ============================================================
# One energy solve
# ============================================================

def failure_row(domain_type, domain_area, ref_energy, ref_area_error, p, N_target, message):
    h_cloud = math.sqrt(domain_area / N_target) if domain_area > 0.0 else np.nan
    Np = steady.polynomial_term_count_2d(p)
    Nc = steady.choose_stencil_size(p, STENCIL_FACTOR, STENCIL_MINIMUM)

    row = {field: np.nan for field in RAW_FIELDS}
    row.update({
        "status": "failed",
        "message": str(message),
        "domain": domain_type,
        "p": p,
        "N": N_target,
        "h": h_cloud,
        "theta": SAT_THETA,
        "Np": Np,
        "Nc": Nc,
        "phs_order": PHS_ORDER,
        "E_exact_ref": ref_energy,
        "reference_area": domain_area,
        "reference_area_error": ref_area_error,
    })

    return row


def run_one_case(domain_type, domain, p, N_target, ref):
    points = steady.generate_nodes(domain_type, domain, N_target)

    cells = steady.clipped_voronoi(points, domain)
    weights = steady.lumped_mass_matrix(cells)

    if np.any(weights <= 0.0):
        raise RuntimeError("nonpositive Voronoi mass weight found")

    N = len(points)
    h_cloud = math.sqrt(domain.area / N)

    Np = steady.polynomial_term_count_2d(p)
    Nc = steady.choose_stencil_size(p, STENCIL_FACTOR, STENCIL_MINIMUM)

    if Nc >= N:
        raise RuntimeError(f"Nc={Nc} >= N={N}; increase node refinement levels")

    area_error = abs(float(np.sum(weights)) - domain.area) / max(abs(domain.area), 1.0e-14)
    union_error = abs(unary_union(cells).area - domain.area) / max(abs(domain.area), 1.0e-14)

    print("-" * 78)
    print(f"domain={domain_type}, p={p}, N={N}, h={h_cloud:.6e}, Np={Np}, Nc={Nc}")

    Dx_raw, Dy_raw, conds = steady.build_rbf_fd_derivative_matrices(
        points,
        Nc,
        PHS_ORDER,
        p,
    )

    sbp = steady.build_pathA_sbp_operators(points, weights, Dx_raw, Dy_raw, p)

    Qx = sbp["Qx"]
    Qy = sbp["Qy"]
    Ex = sbp["Ex"]
    Ey = sbp["Ey"]
    P = sbp["P"]
    Px = sbp["Px"]
    Py = sbp["Py"]

    SBP_x = steady.relative_residual(Qx + Qx.T, Ex)
    SBP_y = steady.relative_residual(Qy + Qy.T, Ey)
    poly_x = steady.relative_residual(Qx @ P, steady.weighted_matrix(weights, Px))
    poly_y = steady.relative_residual(Qy @ P, steady.weighted_matrix(weights, Py))

    sat = steady.build_sat_matrices(Qx, Qy, Ex, Ey, steady.LAMBDA_VEC)
    sol = steady.solve_steady_mms(SAT_THETA, weights, sat, points)

    u_exact = steady.exact_u(points)
    u_h = sol["u_h"]

    E_exact_ref = ref["E_exact_ref"]
    E_exact_ML = ml_energy(u_exact, weights)
    E_num_ML = ml_energy(u_h, weights)

    energy_quad_abs_error = abs(E_exact_ML - E_exact_ref)
    energy_num_abs_error = abs(E_num_ML - E_exact_ref)
    energy_solution_abs_error = abs(E_num_ML - E_exact_ML)

    denom_ref = max(abs(E_exact_ref), 1.0e-14)
    denom_ml = max(abs(E_exact_ML), 1.0e-14)

    A_theta = sat["Q_lambda"] + SAT_THETA * sat["B_minus"]
    min_eig_H = steady.min_eig_symmetric(A_theta)

    print(
        f"  E_num={E_num_ML:.12e}, "
        f"rel_energy_error={energy_num_abs_error / denom_ref:.6e}, "
        f"solution_error={sol['rel_err']:.6e}"
    )

    return {
        "status": "ok",
        "message": "",
        "domain": domain_type,
        "p": p,
        "N": N,
        "h": float(h_cloud),
        "theta": SAT_THETA,
        "Np": Np,
        "Nc": Nc,
        "phs_order": PHS_ORDER,
        "E_exact_ref": float(E_exact_ref),
        "E_exact_ML": float(E_exact_ML),
        "E_num_ML": float(E_num_ML),
        "energy_quad_abs_error": float(energy_quad_abs_error),
        "energy_quad_rel_error": float(energy_quad_abs_error / denom_ref),
        "energy_num_abs_error": float(energy_num_abs_error),
        "energy_num_rel_error": float(energy_num_abs_error / denom_ref),
        "energy_solution_abs_error": float(energy_solution_abs_error),
        "energy_solution_rel_error": float(energy_solution_abs_error / denom_ml),
        "solution_rel_error": float(sol["rel_err"]),
        "linear_residual": float(sol["rel_res"]),
        "observed_energy_rate": np.nan,
        "min_eig_H": float(min_eig_H),
        "SBP_x": float(SBP_x),
        "SBP_y": float(SBP_y),
        "poly_x": float(poly_x),
        "poly_y": float(poly_y),
        "Qx_relative_change": float(sbp["info_x"]["relative_change"]),
        "Qy_relative_change": float(sbp["info_y"]["relative_change"]),
        "min_diag": float(np.min(weights)),
        "max_diag": float(np.max(weights)),
        "mass_condition": float(np.max(weights) / np.min(weights)),
        "area_error": float(area_error),
        "union_error": float(union_error),
        "reference_area": float(ref["area_quad"]),
        "reference_area_error": float(ref["area_error"]),
        "max_fd_cond": float(np.nanmax(conds)),
    }


def add_observed_rates(rows):
    for p in POLY_DEGREES:
        group = [r for r in rows if r["p"] == p]
        group.sort(key=lambda r: safe_float(r["N"]))

        previous = None

        for row in group:
            if row["status"] != "ok":
                continue

            if previous is None:
                row["observed_energy_rate"] = np.nan
            else:
                row["observed_energy_rate"] = convergence_rate(
                    safe_float(previous["energy_num_rel_error"]),
                    safe_float(row["energy_num_rel_error"]),
                    safe_float(previous["h"]),
                    safe_float(row["h"]),
                )

            previous = row


def build_summary(rows, domain_type):
    summary = []

    for p in POLY_DEGREES:
        good = [r for r in rows if r["p"] == p and r["status"] == "ok"]
        good.sort(key=lambda r: safe_float(r["N"]))

        if not good:
            summary.append({
                "domain": domain_type,
                "p": p,
                "theta": SAT_THETA,
                "num_successful_refinements": 0,
                "N_coarsest": np.nan,
                "N_finest": np.nan,
                "h_coarsest": np.nan,
                "h_finest": np.nan,
                "E_exact_ref": np.nan,
                "E_num_finest": np.nan,
                "energy_error_finest": np.nan,
                "solution_error_finest": np.nan,
                "last_step_energy_rate": np.nan,
                "fitted_energy_order": np.nan,
                "max_SBP_x": np.nan,
                "max_poly_x": np.nan,
                "max_linear_residual": np.nan,
                "max_fd_cond": np.nan,
            })
            continue

        last_rate = good[-1]["observed_energy_rate"] if len(good) >= 2 else np.nan

        summary.append({
            "domain": domain_type,
            "p": p,
            "theta": SAT_THETA,
            "num_successful_refinements": len(good),
            "N_coarsest": good[0]["N"],
            "N_finest": good[-1]["N"],
            "h_coarsest": good[0]["h"],
            "h_finest": good[-1]["h"],
            "E_exact_ref": good[-1]["E_exact_ref"],
            "E_num_finest": good[-1]["E_num_ML"],
            "energy_error_finest": good[-1]["energy_num_rel_error"],
            "solution_error_finest": good[-1]["solution_rel_error"],
            "last_step_energy_rate": last_rate,
            "fitted_energy_order": fitted_energy_order(good),
            "max_SBP_x": float(np.nanmax([safe_float(r["SBP_x"]) for r in good])),
            "max_poly_x": float(np.nanmax([safe_float(r["poly_x"]) for r in good])),
            "max_linear_residual": float(np.nanmax([safe_float(r["linear_residual"]) for r in good])),
            "max_fd_cond": float(np.nanmax([safe_float(r["max_fd_cond"]) for r in good])),
        })

    return summary


# ============================================================
# Plotting
# ============================================================

def plot_energy_values(rows, domain_type, E_exact_ref):
    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    fig.patch.set_facecolor("white")

    all_N = sorted({safe_float(r["N"]) for r in rows if r["status"] == "ok"})

    if all_N:
        ax.plot(
            all_N,
            [E_exact_ref for _ in all_N],
            "k--",
            linewidth=1.4,
            label="reference exact energy",
        )

    for p in POLY_DEGREES:
        group = [r for r in rows if r["p"] == p and r["status"] == "ok"]
        group.sort(key=lambda r: safe_float(r["N"]))

        if not group:
            continue

        N = np.array([safe_float(r["N"]) for r in group], dtype=float)
        E_num = np.array([safe_float(r["E_num_ML"]) for r in group], dtype=float)

        ax.plot(N, E_num, "o-", linewidth=1.5, markersize=4.0, label=f"p={p}")

    ax.set_xlabel("number of nodes N")
    ax.set_ylabel("energy")
    ax.set_title(f"Exact and discrete steady energies: {domain_type}")
    ax.grid(True, alpha=0.30)
    ax.legend(ncol=2, fontsize=8)

    fig.tight_layout()
    path = OUTPUT_DIR / f"phase8_energy_{domain_type}_values_vs_N.png"
    path = save_figure(fig, path)
    plt.close(fig)
    return path


def plot_energy_error_vs_N(rows, domain_type):
    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    fig.patch.set_facecolor("white")

    for p in POLY_DEGREES:
        group = [r for r in rows if r["p"] == p and r["status"] == "ok"]
        group.sort(key=lambda r: safe_float(r["N"]))

        if not group:
            continue

        N = np.array([safe_float(r["N"]) for r in group], dtype=float)
        err = np.array([safe_float(r["energy_num_rel_error"]) for r in group], dtype=float)
        quad = np.array([safe_float(r["energy_quad_rel_error"]) for r in group], dtype=float)

        ax.loglog(N, err, "o-", linewidth=1.6, markersize=4.5, label=f"num p={p}")
        ax.loglog(N, quad, ":", linewidth=1.0, alpha=0.55)

    ax.set_xlabel("number of nodes N")
    ax.set_ylabel("relative energy error")
    ax.set_title(f"Steady compatible-SAT energy error vs N: {domain_type}")
    ax.grid(True, which="both", alpha=0.30)
    ax.legend(ncol=2, fontsize=8)

    fig.tight_layout()
    path = OUTPUT_DIR / f"phase8_energy_{domain_type}_relative_error_vs_N.png"
    path = save_figure(fig, path)
    plt.close(fig)
    return path


def plot_energy_error_vs_h(rows, domain_type):
    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    fig.patch.set_facecolor("white")

    for p in POLY_DEGREES:
        group = [r for r in rows if r["p"] == p and r["status"] == "ok"]
        group.sort(key=lambda r: safe_float(r["h"]), reverse=True)

        if not group:
            continue

        h = np.array([safe_float(r["h"]) for r in group], dtype=float)
        err = np.array([safe_float(r["energy_num_rel_error"]) for r in group], dtype=float)

        ax.loglog(h, err, "o-", linewidth=1.6, markersize=4.5, label=f"p={p}")

    ax.invert_xaxis()
    ax.set_xlabel("cloud spacing h = sqrt(|Omega|/N)")
    ax.set_ylabel("relative energy error")
    ax.set_title(f"Steady compatible-SAT energy error vs h: {domain_type}")
    ax.grid(True, which="both", alpha=0.30)
    ax.legend(ncol=2, fontsize=8)

    fig.tight_layout()
    path = OUTPUT_DIR / f"phase8_energy_{domain_type}_relative_error_vs_h.png"
    path = save_figure(fig, path)
    plt.close(fig)
    return path


def plot_energy_vs_solution_error(rows, domain_type):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    fig.patch.set_facecolor("white")

    for p in POLY_DEGREES:
        group = [r for r in rows if r["p"] == p and r["status"] == "ok"]

        if not group:
            continue

        sol_err = np.array([safe_float(r["solution_rel_error"]) for r in group], dtype=float)
        energy_err = np.array([safe_float(r["energy_num_rel_error"]) for r in group], dtype=float)

        ax.loglog(sol_err, energy_err, "o", markersize=5.0, label=f"p={p}")

    ax.set_xlabel("relative solution error")
    ax.set_ylabel("relative energy error")
    ax.set_title(f"Energy error vs solution error: {domain_type}")
    ax.grid(True, which="both", alpha=0.30)
    ax.legend(ncol=2, fontsize=8)

    fig.tight_layout()
    path = OUTPUT_DIR / f"phase8_energy_{domain_type}_energy_vs_solution_error.png"
    path = save_figure(fig, path)
    plt.close(fig)
    return path


# ============================================================
# Main
# ============================================================

def main():
    domain_type = DOMAIN_TYPE

    if domain_type not in NODE_REFINEMENTS_BY_DOMAIN:
        raise ValueError(
            f"Unknown DOMAIN_TYPE={domain_type!r}. "
            f"Use one of {list(NODE_REFINEMENTS_BY_DOMAIN)}."
        )

    steady.LAMBDA_VEC = np.array([1.0, 1.0])

    domain = steady.build_domain(domain_type)
    node_refinements = NODE_REFINEMENTS_BY_DOMAIN[domain_type]

    print("=" * 78)
    print("PHASE 8 - STEADY COMPATIBLE-SAT ENERGY DIAGNOSTICS")
    print("=" * 78)
    print(f"manual DOMAIN_TYPE       = {domain_type}")
    print(f"node refinements         = {node_refinements}")
    print(f"polynomial degrees       = {POLY_DEGREES}")
    print(f"PHS order                = {PHS_ORDER}")
    print(f"SAT theta                = {SAT_THETA}")
    print(f"reference quadrature     = {REFERENCE_QUAD_ORDER} x {REFERENCE_QUAD_ORDER}")
    print(f"output folder            = {OUTPUT_DIR}")
    print("=" * 78)

    print("Computing high-order reference exact energy...")
    ref = reference_exact_energy(domain, REFERENCE_QUAD_ORDER)
    print(f"  E_exact_ref             = {ref['E_exact_ref']:.16e}")
    print(f"  reference area          = {ref['area_quad']:.16e}")
    print(f"  reference area error    = {ref['area_error']:.6e}")
    print(f"  reference triangles     = {ref['num_triangles']}")

    rows = []

    for p in POLY_DEGREES:
        print("\n" + "#" * 78)
        print(f"POLYNOMIAL DEGREE p={p}")
        print("#" * 78)

        for N_target in node_refinements:
            try:
                row = run_one_case(domain_type, domain, p, N_target, ref)
            except Exception as exc:
                print("-" * 78)
                print(f"FAILED: domain={domain_type}, p={p}, N={N_target}")
                print(f"  {exc}")
                row = failure_row(
                    domain_type,
                    domain.area,
                    ref["E_exact_ref"],
                    ref["area_error"],
                    p,
                    N_target,
                    exc,
                )

            rows.append(row)

    add_observed_rates(rows)
    summary = build_summary(rows, domain_type)

    raw_csv = OUTPUT_DIR / f"phase8_energy_{domain_type}_raw.csv"
    summary_csv = OUTPUT_DIR / f"phase8_energy_{domain_type}_summary.csv"

    raw_csv = write_csv(raw_csv, rows, RAW_FIELDS)
    summary_csv = write_csv(summary_csv, summary, SUMMARY_FIELDS)

    values_plot = plot_energy_values(rows, domain_type, ref["E_exact_ref"])
    err_N_plot = plot_energy_error_vs_N(rows, domain_type)
    err_h_plot = plot_energy_error_vs_h(rows, domain_type)
    err_sol_plot = plot_energy_vs_solution_error(rows, domain_type)

    print("\n" + "=" * 78)
    print("PHASE 8 ENERGY DIAGNOSTICS COMPLETE")
    print("=" * 78)
    print(f"raw table      -> {raw_csv}")
    print(f"summary table  -> {summary_csv}")
    print(f"values plot    -> {values_plot}")
    print(f"error vs N     -> {err_N_plot}")
    print(f"error vs h     -> {err_h_plot}")
    print(f"energy/sol err -> {err_sol_plot}")
    print("-" * 78)
    print("p | successful levels | finest N | E_num finest | energy err | solution err | order")
    print("-" * 78)

    for row in summary:
        print(
            f"{row['p']} | "
            f"{row['num_successful_refinements']} | "
            f"{row['N_finest']} | "
            f"{safe_float(row['E_num_finest']):.12e} | "
            f"{safe_float(row['energy_error_finest']):.3e} | "
            f"{safe_float(row['solution_error_finest']):.3e} | "
            f"{safe_float(row['fitted_energy_order']):.3f}"
        )

    print("-" * 78)


if __name__ == "__main__":
    main()
