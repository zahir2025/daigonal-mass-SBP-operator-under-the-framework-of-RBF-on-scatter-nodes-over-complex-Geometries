"""
Phase 7 - Time-Dependent Advection MMS Convergence By Node Refinement
=====================================================================

Manual workflow:
    1. Change DOMAIN_TYPE below to one of:
           "annulus"
           "box_minus_circle"
           "box_minus_airfoil"
    2. Optionally edit NODE_REFINEMENTS_BY_DOMAIN for that domain.
    3. Run this file.

This script reuses the Phase 4 Path A diagonal-mass SBP-SAT construction and
computes time-dependent advection MMS convergence for polynomial degrees
p = 1,...,7.

The MMS is the same as Phase 4:

    u_exact(x,y,t) = exp(x + y - t)
    lambda = [1, 1]
    u_t + lambda . grad(u) = f
    f = exp(x + y - t)

The default time step scales like dt ~ h^2. With RK4, this makes the temporal
error O(h^8), so the p=1,...,7 spatial convergence is less likely to be hidden
by time-integration error.

Outputs:
    outputs_phase7_time_MMS_convergence/
        phase7_time_convergence_<domain>_raw.csv
        phase7_time_convergence_<domain>_summary.csv
        phase7_time_convergence_<domain>_error_vs_N.png
        phase7_time_convergence_<domain>_error_vs_h.png
        phase7_time_convergence_<domain>_observed_orders.png
"""

from pathlib import Path
import csv
import math
from datetime import datetime

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shapely.ops import unary_union

import phase4_pathA_time_dependent_MMS as time_mms


# ============================================================
# Manual configuration
# ============================================================

# Change this manually, then rerun the script.
DOMAIN_TYPE = "annulus"

POLY_DEGREES = list(range(1, 8))

# The p=7 stencil has 126 nodes with the default stencil rule, so the coarsest
# N must stay comfortably larger than that. Increase these lists for a final
# high-resolution run if runtime allows.
NODE_REFINEMENTS_BY_DOMAIN = {
    "annulus": [400, 800, 1200, 3200],
    "box_minus_circle": [260, 380, 560, 820],
    "box_minus_airfoil": [450, 650, 950, 1350],
}

PHS_ORDER = 5
STENCIL_FACTOR = 3.5
STENCIL_MINIMUM = 25

SAT_THETA = 1.0

FINAL_TIME = 1.25

# dt = TIME_CFL * h**TIME_STEP_POWER / ||lambda||.
# TIME_STEP_POWER=2 is recommended for p=1,...,7 with RK4.
TIME_CFL = 0.08
TIME_STEP_POWER = 2.0

FIG_DPI = 180

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs_phase7_time_MMS_convergence"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


RAW_FIELDS = [
    "status",
    "message",
    "domain",
    "p",
    "N",
    "h",
    "dt",
    "num_steps",
    "final_time",
    "theta",
    "Np",
    "Nc",
    "phs_order",
    "rel_error_final",
    "max_error_over_time",
    "observed_rate",
    "energy_ratio",
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
    "max_fd_cond",
]

SUMMARY_FIELDS = [
    "domain",
    "p",
    "theta",
    "final_time",
    "num_successful_refinements",
    "N_coarsest",
    "N_finest",
    "h_coarsest",
    "h_finest",
    "dt_finest",
    "steps_finest",
    "error_coarsest",
    "error_finest",
    "max_error_finest",
    "last_step_rate",
    "fitted_order",
    "max_SBP_x",
    "max_poly_x",
    "max_fd_cond",
]


# ============================================================
# Helpers
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


def fitted_order(rows):
    good = [
        r for r in rows
        if r["status"] == "ok"
        and finite_positive(safe_float(r["h"]))
        and finite_positive(safe_float(r["rel_error_final"]))
    ]

    if len(good) < 2:
        return np.nan

    h = np.array([safe_float(r["h"]) for r in good], dtype=float)
    err = np.array([safe_float(r["rel_error_final"]) for r in good], dtype=float)

    slope, _ = np.polyfit(np.log(h), np.log(err), 1)
    return float(slope)


def compute_time_step(h_cloud):
    lambda_norm = max(np.linalg.norm(time_mms.LAMBDA_VEC), 1.0e-14)
    dt_raw = TIME_CFL * (h_cloud ** TIME_STEP_POWER) / lambda_norm
    num_steps = max(1, int(np.ceil(FINAL_TIME / dt_raw)))
    dt = FINAL_TIME / num_steps

    return float(dt), int(num_steps)


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


def failure_row(domain_type, domain_area, p, N_target, message):
    h_cloud = math.sqrt(domain_area / N_target) if domain_area > 0.0 else np.nan
    dt, num_steps = compute_time_step(h_cloud) if finite_positive(h_cloud) else (np.nan, 0)
    Np = time_mms.polynomial_term_count_2d(p)
    Nc = time_mms.choose_stencil_size(p, STENCIL_FACTOR, STENCIL_MINIMUM)

    row = {field: np.nan for field in RAW_FIELDS}
    row.update({
        "status": "failed",
        "message": str(message),
        "domain": domain_type,
        "p": p,
        "N": N_target,
        "h": h_cloud,
        "dt": dt,
        "num_steps": num_steps,
        "final_time": FINAL_TIME,
        "theta": SAT_THETA,
        "Np": Np,
        "Nc": Nc,
        "phs_order": PHS_ORDER,
    })

    return row


# ============================================================
# One MMS solve
# ============================================================

def run_one_case(domain_type, domain, p, N_target):
    points = time_mms.generate_nodes(domain_type, domain, N_target)

    cells = time_mms.clipped_voronoi(points, domain)
    weights = time_mms.lumped_mass_matrix(cells)

    if np.any(weights <= 0.0):
        raise RuntimeError("nonpositive Voronoi mass weight found")

    N = len(points)
    h_cloud = math.sqrt(domain.area / N)
    dt, num_steps = compute_time_step(h_cloud)

    Np = time_mms.polynomial_term_count_2d(p)
    Nc = time_mms.choose_stencil_size(p, STENCIL_FACTOR, STENCIL_MINIMUM)

    if Nc >= N:
        raise RuntimeError(f"Nc={Nc} >= N={N}; increase node refinement levels")

    area_error = abs(float(np.sum(weights)) - domain.area) / max(abs(domain.area), 1.0e-14)
    union_error = abs(unary_union(cells).area - domain.area) / max(abs(domain.area), 1.0e-14)

    print("-" * 78)
    print(
        f"domain={domain_type}, p={p}, N={N}, h={h_cloud:.6e}, "
        f"dt={dt:.6e}, steps={num_steps}, Np={Np}, Nc={Nc}"
    )

    Dx_raw, Dy_raw, conds = time_mms.build_rbf_fd_derivative_matrices(
        points,
        Nc,
        PHS_ORDER,
        p,
    )

    sbp = time_mms.build_pathA_sbp_operators(points, weights, Dx_raw, Dy_raw, p)

    Qx = sbp["Qx"]
    Qy = sbp["Qy"]
    Ex = sbp["Ex"]
    Ey = sbp["Ey"]
    P = sbp["P"]
    Px = sbp["Px"]
    Py = sbp["Py"]

    SBP_x = time_mms.relative_residual(Qx + Qx.T, Ex)
    SBP_y = time_mms.relative_residual(Qy + Qy.T, Ey)
    poly_x = time_mms.relative_residual(Qx @ P, time_mms.weighted_matrix(weights, Px))
    poly_y = time_mms.relative_residual(Qy @ P, time_mms.weighted_matrix(weights, Py))

    sat = time_mms.build_sat_matrices(Qx, Qy, Ex, Ey, time_mms.LAMBDA_VEC)

    A_theta = sat["Q_lambda"] + SAT_THETA * sat["B_minus"]
    min_eig_H = time_mms.min_eig_symmetric(A_theta)

    result = time_mms.run_time_mms(
        points,
        weights,
        sat,
        SAT_THETA,
        dt,
        num_steps,
    )

    print(
        f"  final_rel_error={result['rel_err_T']:.6e}, "
        f"max_error={result['max_err']:.6e}, "
        f"SBP_x={SBP_x:.3e}, poly_x={poly_x:.3e}"
    )

    return {
        "status": "ok",
        "message": "",
        "domain": domain_type,
        "p": p,
        "N": N,
        "h": float(h_cloud),
        "dt": float(dt),
        "num_steps": int(num_steps),
        "final_time": FINAL_TIME,
        "theta": SAT_THETA,
        "Np": Np,
        "Nc": Nc,
        "phs_order": PHS_ORDER,
        "rel_error_final": float(result["rel_err_T"]),
        "max_error_over_time": float(result["max_err"]),
        "observed_rate": np.nan,
        "energy_ratio": float(result["energy_ratio"]),
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
                row["observed_rate"] = np.nan
            else:
                row["observed_rate"] = convergence_rate(
                    safe_float(previous["rel_error_final"]),
                    safe_float(row["rel_error_final"]),
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
                "final_time": FINAL_TIME,
                "num_successful_refinements": 0,
                "N_coarsest": np.nan,
                "N_finest": np.nan,
                "h_coarsest": np.nan,
                "h_finest": np.nan,
                "dt_finest": np.nan,
                "steps_finest": np.nan,
                "error_coarsest": np.nan,
                "error_finest": np.nan,
                "max_error_finest": np.nan,
                "last_step_rate": np.nan,
                "fitted_order": np.nan,
                "max_SBP_x": np.nan,
                "max_poly_x": np.nan,
                "max_fd_cond": np.nan,
            })
            continue

        last_rate = good[-1]["observed_rate"] if len(good) >= 2 else np.nan

        summary.append({
            "domain": domain_type,
            "p": p,
            "theta": SAT_THETA,
            "final_time": FINAL_TIME,
            "num_successful_refinements": len(good),
            "N_coarsest": good[0]["N"],
            "N_finest": good[-1]["N"],
            "h_coarsest": good[0]["h"],
            "h_finest": good[-1]["h"],
            "dt_finest": good[-1]["dt"],
            "steps_finest": good[-1]["num_steps"],
            "error_coarsest": good[0]["rel_error_final"],
            "error_finest": good[-1]["rel_error_final"],
            "max_error_finest": good[-1]["max_error_over_time"],
            "last_step_rate": last_rate,
            "fitted_order": fitted_order(good),
            "max_SBP_x": float(np.nanmax([safe_float(r["SBP_x"]) for r in good])),
            "max_poly_x": float(np.nanmax([safe_float(r["poly_x"]) for r in good])),
            "max_fd_cond": float(np.nanmax([safe_float(r["max_fd_cond"]) for r in good])),
        })

    return summary


# ============================================================
# Plotting
# ============================================================

def plot_error_vs_N(rows, domain_type):
    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    fig.patch.set_facecolor("white")

    for p in POLY_DEGREES:
        group = [r for r in rows if r["p"] == p and r["status"] == "ok"]
        group.sort(key=lambda r: safe_float(r["N"]))

        if not group:
            continue

        N = np.array([safe_float(r["N"]) for r in group], dtype=float)
        err = np.array([safe_float(r["rel_error_final"]) for r in group], dtype=float)

        ax.loglog(N, err, "o-", linewidth=1.6, markersize=4.5, label=f"p={p}")

    ax.set_xlabel("number of nodes N")
    ax.set_ylabel("final relative M_L error")
    ax.set_title(f"Time-dependent advection MMS convergence vs N: {domain_type}")
    ax.grid(True, which="both", alpha=0.30)
    ax.legend(ncol=2, fontsize=8)

    fig.tight_layout()
    path = OUTPUT_DIR / f"phase7_time_convergence_{domain_type}_error_vs_N.png"
    path = save_figure(fig, path)
    plt.close(fig)
    return path


def plot_error_vs_h(rows, domain_type):
    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    fig.patch.set_facecolor("white")

    for p in POLY_DEGREES:
        group = [r for r in rows if r["p"] == p and r["status"] == "ok"]
        group.sort(key=lambda r: safe_float(r["h"]), reverse=True)

        if not group:
            continue

        h = np.array([safe_float(r["h"]) for r in group], dtype=float)
        err = np.array([safe_float(r["rel_error_final"]) for r in group], dtype=float)

        ax.loglog(h, err, "o-", linewidth=1.6, markersize=4.5, label=f"p={p}")

    ax.invert_xaxis()
    ax.set_xlabel("cloud spacing h = sqrt(|Omega|/N)")
    ax.set_ylabel("final relative M_L error")
    ax.set_title(f"Time-dependent advection MMS convergence vs h: {domain_type}")
    ax.grid(True, which="both", alpha=0.30)
    ax.legend(ncol=2, fontsize=8)

    fig.tight_layout()
    path = OUTPUT_DIR / f"phase7_time_convergence_{domain_type}_error_vs_h.png"
    path = save_figure(fig, path)
    plt.close(fig)
    return path


def plot_observed_orders(summary, domain_type):
    p = np.array([int(r["p"]) for r in summary], dtype=int)
    fitted = np.array([safe_float(r["fitted_order"]) for r in summary], dtype=float)
    last = np.array([safe_float(r["last_step_rate"]) for r in summary], dtype=float)

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    fig.patch.set_facecolor("white")

    ax.plot(p, fitted, "o-", linewidth=1.8, label="fit over all refinements")
    ax.plot(p, last, "s--", linewidth=1.6, label="last refinement step")
    ax.plot(p, p, "k:", linewidth=1.2, label="reference order p")

    ax.set_xlabel("polynomial degree p")
    ax.set_ylabel("observed order")
    ax.set_title(f"Observed time-dependent convergence orders: {domain_type}")
    ax.set_xticks(p)
    ax.grid(True, alpha=0.30)
    ax.legend(fontsize=8)

    fig.tight_layout()
    path = OUTPUT_DIR / f"phase7_time_convergence_{domain_type}_observed_orders.png"
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

    time_mms.LAMBDA_VEC = np.array([1.0, 1.0])

    domain = time_mms.build_domain(domain_type)
    node_refinements = NODE_REFINEMENTS_BY_DOMAIN[domain_type]

    print("=" * 78)
    print("PHASE 7 - TIME-DEPENDENT ADVECTION MMS CONVERGENCE")
    print("=" * 78)
    print(f"manual DOMAIN_TYPE       = {domain_type}")
    print(f"node refinements         = {node_refinements}")
    print(f"polynomial degrees       = {POLY_DEGREES}")
    print(f"PHS order                = {PHS_ORDER}")
    print(f"SAT theta                = {SAT_THETA}")
    print(f"final time               = {FINAL_TIME}")
    print(f"dt rule                  = {TIME_CFL} * h^{TIME_STEP_POWER} / ||lambda||")
    print(f"output folder            = {OUTPUT_DIR}")
    print("=" * 78)

    rows = []

    for p in POLY_DEGREES:
        print("\n" + "#" * 78)
        print(f"POLYNOMIAL DEGREE p={p}")
        print("#" * 78)

        for N_target in node_refinements:
            try:
                row = run_one_case(domain_type, domain, p, N_target)
            except Exception as exc:
                print("-" * 78)
                print(f"FAILED: domain={domain_type}, p={p}, N={N_target}")
                print(f"  {exc}")
                row = failure_row(domain_type, domain.area, p, N_target, exc)

            rows.append(row)

    add_observed_rates(rows)
    summary = build_summary(rows, domain_type)

    raw_csv = OUTPUT_DIR / f"phase7_time_convergence_{domain_type}_raw.csv"
    summary_csv = OUTPUT_DIR / f"phase7_time_convergence_{domain_type}_summary.csv"

    raw_csv = write_csv(raw_csv, rows, RAW_FIELDS)
    summary_csv = write_csv(summary_csv, summary, SUMMARY_FIELDS)

    path_N = plot_error_vs_N(rows, domain_type)
    path_h = plot_error_vs_h(rows, domain_type)
    path_orders = plot_observed_orders(summary, domain_type)

    print("\n" + "=" * 78)
    print("PHASE 7 TIME-DEPENDENT CONVERGENCE COMPLETE")
    print("=" * 78)
    print(f"raw table      -> {raw_csv}")
    print(f"summary table  -> {summary_csv}")
    print(f"plot vs N      -> {path_N}")
    print(f"plot vs h      -> {path_h}")
    print(f"order plot     -> {path_orders}")
    print("-" * 78)
    print("p | successful levels | finest N | finest rel error | fitted order | last rate")
    print("-" * 78)

    for row in summary:
        print(
            f"{row['p']} | "
            f"{row['num_successful_refinements']} | "
            f"{row['N_finest']} | "
            f"{safe_float(row['error_finest']):.3e} | "
            f"{safe_float(row['fitted_order']):.3f} | "
            f"{safe_float(row['last_step_rate']):.3f}"
        )

    print("-" * 78)


if __name__ == "__main__":
    main()
