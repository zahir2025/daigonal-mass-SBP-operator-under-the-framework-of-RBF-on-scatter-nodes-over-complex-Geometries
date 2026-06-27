"""
Phase 2d - Discrete Operator Order Test
=======================================

Alphabetic continuation of the closest corrected operator phase:

    Phase 2c -> Phase 2d

Phase 2c builds the Path A diagonal-mass SBP-compatible operators.  Phase 2d
keeps that same construction and tests the order of the discrete derivative
operator directly.  No PDE, SAT term, or time integrator is used here.

Theorem tested in this phase:

    Let X_h be a quasi-uniform node cloud on a bounded Lipschitz domain and let
    M_L = diag(w_i) be strictly positive diagonal Voronoi weights.  For a fixed
    polynomial degree p, let P contain all bivariate polynomials of total
    degree <= p and let P_x, P_y denote their exact derivatives at the nodes.
    Starting from raw RBF-FD matrices D_x^0,D_y^0, set Q_x^0=M_L D_x^0 and
    Q_y^0=M_L D_y^0.  The Path-A correction constructs symmetric matrices
    E_x,E_y with

        P.T E_x P = P_x.T M_L P + P.T M_L P_x,
        P.T E_y P = P_y.T M_L P + P.T M_L P_y,

    and then computes minimum-change matrices Q_x,Q_y such that

        Q_x + Q_x.T = E_x,     Q_x P = M_L P_x,
        Q_y + Q_y.T = E_y,     Q_y P = M_L P_y.

    Therefore D_x=M_L^{-1}Q_x and D_y=M_L^{-1}Q_y are diagonal-mass SBP
    operators and are exact on all polynomials of degree <= p.  If the raw
    RBF-FD family is stable on the refinement sequence and the correction is
    uniformly bounded in the M_L operator norm, then for every smooth u,

        ||D_x u - u_x||_{M_L} + ||D_y u - u_y||_{M_L} <= C h^p ||u||_{W^{p+1}}

    with C independent of h.  Hence the corrected operator is stable,
    pth-order accurate, and convergent in the M_L norm.

For each selected domain, refinement level, and polynomial degree p:

    1. build the Phase 2c raw PHS/RBF-FD derivative matrices,
    2. build the M_L-compatible Path A corrected Qx,Qy operators,
    3. form Dx = M_L^{-1} Qx and Dy = M_L^{-1} Qy,
    4. apply raw and corrected operators to a smooth manufactured field,
    5. fit the observed derivative order from the M_L-norm error vs h.

Run:

    python scripts/phase2d_discrete_operator_order.py

Useful PowerShell overrides:

    $env:PHASE2D_DOMAINS='annulus'
    $env:PHASE2D_POLY_DEGREES='1,2,3,4,5,6,7'
    $env:PHASE2D_N_TARGETS='annulus:160,240,360,520'
    python scripts/phase2d_discrete_operator_order.py
"""

from pathlib import Path
import csv
import math
import os
from datetime import datetime

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import phase2c_pathA_ML_compatible_Ex_Ey as phase2c


# ============================================================
# Configuration
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs_phase2d_discrete_operator_order"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PHS_ORDER = int(os.environ.get("PHASE2D_PHS_ORDER", "5"))
STENCIL_FACTOR = float(os.environ.get("PHASE2D_STENCIL_FACTOR", "3.5"))
STENCIL_MINIMUM = int(os.environ.get("PHASE2D_STENCIL_MINIMUM", "25"))
FIG_DPI = int(os.environ.get("PHASE2D_FIG_DPI", "190"))
CORRECTED_OPERATOR = "corrected_pathA_sbp"

DEFAULT_DOMAINS = [
    "annulus",
    "box_minus_circle",
    "box_minus_airfoil",
]

DEFAULT_N_TARGETS = {
    "annulus": [400, 800, 1600, 3200],
    "box_minus_circle": [400, 800, 1600, 3200],
    "box_minus_airfoil": [400, 800, 1600, 3200],
}


RAW_FIELDS = [
    "status",
    "message",
    "domain",
    "p",
    "operator",
    "N_target",
    "N",
    "h",
    "area",
    "Np",
    "Nc",
    "phs_order",
    "expected_order",
    "dx_rel_error",
    "dy_rel_error",
    "grad_rel_error",
    "observed_rate",
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
    "operator",
    "expected_order",
    "num_successful_refinements",
    "N_coarsest",
    "N_finest",
    "h_coarsest",
    "h_finest",
    "error_coarsest",
    "error_finest",
    "fitted_order",
    "last_step_rate",
    "max_SBP_x",
    "max_SBP_y",
    "max_poly_x",
    "max_poly_y",
    "max_mass_condition",
    "max_fd_cond",
]


CORRECTED_DETAIL_FIELDS = [
    "domain",
    "p",
    "N_target",
    "N",
    "h",
    "grad_rel_error",
    "observed_rate",
    "dx_rel_error",
    "dy_rel_error",
    "SBP_max",
    "SBP_x",
    "SBP_y",
    "poly_max",
    "poly_x",
    "poly_y",
    "Qx_relative_change",
    "Qy_relative_change",
    "mass_condition",
    "max_fd_cond",
]


CORRECTED_SUMMARY_FIELDS = [
    "domain",
    "p",
    "expected_order",
    "num_successful_refinements",
    "N_coarsest",
    "N_finest",
    "h_coarsest",
    "h_finest",
    "error_coarsest",
    "error_finest",
    "fitted_order",
    "last_step_rate",
    "max_SBP_residual",
    "max_SBP_x",
    "max_SBP_y",
    "max_poly_residual",
    "max_poly_x",
    "max_poly_y",
    "max_mass_condition",
    "max_fd_cond",
]


# ============================================================
# Parsing and small utilities
# ============================================================

def env_list(name, default_values):
    raw = os.environ.get(name, "")
    if not raw.strip():
        return list(default_values)
    return [item.strip() for item in raw.split(",") if item.strip()]


def env_int_list(name, default_values):
    raw = os.environ.get(name, "")
    if not raw.strip():
        return list(default_values)
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def parse_n_targets(default_targets):
    targets = {key: list(value) for key, value in default_targets.items()}
    raw = os.environ.get("PHASE2D_N_TARGETS", "").strip()
    if not raw:
        return targets

    for block in raw.split(";"):
        block = block.strip()
        if not block:
            continue

        name, values = block.split(":", 1)
        targets[name.strip()] = [
            int(item.strip()) for item in values.split(",") if item.strip()
        ]

    return targets


DOMAINS = env_list("PHASE2D_DOMAINS", DEFAULT_DOMAINS)
POLY_DEGREES = env_int_list("PHASE2D_POLY_DEGREES", [1, 2, 3, 4, 5, 6, 7])
N_TARGETS = parse_n_targets(DEFAULT_N_TARGETS)


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return np.nan


def max_finite(values):
    finite = [safe_float(value) for value in values if np.isfinite(safe_float(value))]
    if not finite:
        return np.nan
    return float(max(finite))


def finite_positive(value):
    return np.isfinite(value) and value > 0.0


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
        row for row in rows
        if row["status"] == "ok"
        and finite_positive(safe_float(row["h"]))
        and finite_positive(safe_float(row["grad_rel_error"]))
    ]

    if len(good) < 2:
        return np.nan

    h = np.array([safe_float(row["h"]) for row in good], dtype=float)
    err = np.array([safe_float(row["grad_rel_error"]) for row in good], dtype=float)
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
        fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
        return path
    except PermissionError:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback = path.with_name(f"{path.stem}_{stamp}{path.suffix}")
        fig.savefig(fallback, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
        print(f"WARNING: could not overwrite locked figure -> {path}")
        print(f"         saved this run instead as        -> {fallback}")
        return fallback


# ============================================================
# Manufactured operator test field
# ============================================================

def manufactured_field(points):
    x = points[:, 0]
    y = points[:, 1]

    a = 0.75 * x - 0.35 * y
    b = 2.10 * x + 1.70 * y
    c = 1.30 * x - 0.80 * y

    exp_a = np.exp(a)
    sin_b = np.sin(b)
    cos_b = np.cos(b)
    sin_c = np.sin(c)
    cos_c = np.cos(c)

    u = exp_a + 0.25 * sin_b + 0.10 * cos_c
    ux = 0.75 * exp_a + 0.25 * 2.10 * cos_b - 0.10 * 1.30 * sin_c
    uy = -0.35 * exp_a + 0.25 * 1.70 * cos_b + 0.10 * 0.80 * sin_c

    return u, ux, uy


def mass_norm(v, weights):
    return math.sqrt(max(float(np.sum(weights * v * v)), 0.0))


def relative_gradient_error(dx_num, dy_num, ux_exact, uy_exact, weights):
    dx_err = dx_num - ux_exact
    dy_err = dy_num - uy_exact

    num = math.sqrt(mass_norm(dx_err, weights) ** 2 + mass_norm(dy_err, weights) ** 2)
    den = math.sqrt(mass_norm(ux_exact, weights) ** 2 + mass_norm(uy_exact, weights) ** 2)

    return num / max(den, 1.0e-14)


def derivative_errors(points, weights, Dx, Dy):
    u, ux_exact, uy_exact = manufactured_field(points)

    dx_num = Dx @ u
    dy_num = Dy @ u

    dx_rel = phase2c.relative_mass_norm(dx_num - ux_exact, ux_exact, weights)
    dy_rel = phase2c.relative_mass_norm(dy_num - uy_exact, uy_exact, weights)
    grad_rel = relative_gradient_error(dx_num, dy_num, ux_exact, uy_exact, weights)

    return dx_rel, dy_rel, grad_rel


# ============================================================
# Path A operator construction
# ============================================================

def build_phase2c_operator(points, weights, poly_degree):
    npoly = phase2c.polynomial_term_count_2d(poly_degree)
    nc = phase2c.choose_stencil_size(poly_degree, STENCIL_FACTOR, STENCIL_MINIMUM)

    if nc >= len(points):
        raise RuntimeError(f"Nc={nc} must be smaller than N={len(points)}")

    dx_raw, dy_raw, fd_conds = phase2c.build_rbf_fd_derivative_matrices(
        points,
        nc,
        PHS_ORDER,
        poly_degree,
    )

    P, Px, Py, _, _, _, _ = phase2c.global_polynomial_matrices(points, poly_degree)

    qx_raw = phase2c.weighted_matrix(weights, dx_raw)
    qy_raw = phase2c.weighted_matrix(weights, dy_raw)

    ex_raw = qx_raw + qx_raw.T
    ey_raw = qy_raw + qy_raw.T

    target_x = phase2c.weighted_matrix(weights, Px)
    target_y = phase2c.weighted_matrix(weights, Py)

    b_ml_x = Px.T @ phase2c.weighted_matrix(weights, P) + P.T @ target_x
    b_ml_y = Py.T @ phase2c.weighted_matrix(weights, P) + P.T @ target_y

    ex = phase2c.project_symmetric_E_to_match_moments(ex_raw, P, b_ml_x)
    ey = phase2c.project_symmetric_E_to_match_moments(ey_raw, P, b_ml_y)

    qx, info_x = phase2c.minimum_change_correct_Q(qx_raw, ex, P, Px, weights)
    qy, info_y = phase2c.minimum_change_correct_Q(qy_raw, ey, P, Py, weights)

    dx_corrected = qx / weights[:, None]
    dy_corrected = qy / weights[:, None]

    raw_metrics = {
        "SBP_x": phase2c.relative_residual(qx_raw + qx_raw.T, ex),
        "SBP_y": phase2c.relative_residual(qy_raw + qy_raw.T, ey),
        "poly_x": phase2c.relative_residual(qx_raw @ P, target_x),
        "poly_y": phase2c.relative_residual(qy_raw @ P, target_y),
        "Qx_relative_change": np.nan,
        "Qy_relative_change": np.nan,
    }

    corrected_metrics = {
        "SBP_x": phase2c.relative_residual(qx + qx.T, ex),
        "SBP_y": phase2c.relative_residual(qy + qy.T, ey),
        "poly_x": phase2c.relative_residual(qx @ P, target_x),
        "poly_y": phase2c.relative_residual(qy @ P, target_y),
        "Qx_relative_change": float(info_x["relative_change"]),
        "Qy_relative_change": float(info_y["relative_change"]),
    }

    return {
        "Np": int(npoly),
        "Nc": int(nc),
        "Dx_raw": dx_raw,
        "Dy_raw": dy_raw,
        "Dx_corrected": dx_corrected,
        "Dy_corrected": dy_corrected,
        "max_fd_cond": float(np.nanmax(fd_conds)),
        "raw_metrics": raw_metrics,
        "corrected_metrics": corrected_metrics,
    }


def make_result_row(base, operator_name, metrics, errors):
    dx_rel, dy_rel, grad_rel = errors

    row = dict(base)
    row.update({
        "status": "ok",
        "message": "",
        "operator": operator_name,
        "dx_rel_error": float(dx_rel),
        "dy_rel_error": float(dy_rel),
        "grad_rel_error": float(grad_rel),
        "observed_rate": np.nan,
        "SBP_x": float(metrics["SBP_x"]),
        "SBP_y": float(metrics["SBP_y"]),
        "poly_x": float(metrics["poly_x"]),
        "poly_y": float(metrics["poly_y"]),
        "Qx_relative_change": safe_float(metrics["Qx_relative_change"]),
        "Qy_relative_change": safe_float(metrics["Qy_relative_change"]),
    })

    return row


def failure_rows(domain_type, poly_degree, n_target, message):
    rows = []
    for operator_name in ["raw_rbf_fd", "corrected_pathA_sbp"]:
        row = {field: np.nan for field in RAW_FIELDS}
        row.update({
            "status": "failed",
            "message": str(message),
            "domain": domain_type,
            "p": poly_degree,
            "operator": operator_name,
            "N_target": n_target,
            "expected_order": poly_degree,
            "phs_order": PHS_ORDER,
        })
        rows.append(row)
    return rows


def run_one_case(domain_type, poly_degree, n_target):
    print("\n" + "#" * 72)
    print(f"PHASE 2d CASE: domain={domain_type}, p={poly_degree}, N_target={n_target}")
    print("#" * 72)

    domain = phase2c.build_domain(domain_type)
    points = phase2c.generate_nodes(domain_type, domain, n_target)
    cells = phase2c.clipped_voronoi(points, domain)
    weights = phase2c.lumped_mass_matrix(cells)

    if np.any(weights <= 0.0):
        raise RuntimeError("nonpositive diagonal mass weight found")

    area_error = abs(np.sum(weights) - domain.area) / max(abs(domain.area), 1.0e-14)
    union_error = abs(phase2c.unary_union(cells).area - domain.area) / max(abs(domain.area), 1.0e-14)

    operator = build_phase2c_operator(points, weights, poly_degree)

    raw_errors = derivative_errors(points, weights, operator["Dx_raw"], operator["Dy_raw"])
    corrected_errors = derivative_errors(
        points,
        weights,
        operator["Dx_corrected"],
        operator["Dy_corrected"],
    )

    base = {
        "domain": domain_type,
        "p": poly_degree,
        "N_target": n_target,
        "N": int(len(points)),
        "h": math.sqrt(float(domain.area) / float(len(points))),
        "area": float(domain.area),
        "Np": int(operator["Np"]),
        "Nc": int(operator["Nc"]),
        "phs_order": PHS_ORDER,
        "expected_order": poly_degree,
        "min_diag": float(np.min(weights)),
        "max_diag": float(np.max(weights)),
        "mass_condition": float(np.max(weights) / np.min(weights)),
        "area_error": float(area_error),
        "union_error": float(union_error),
        "max_fd_cond": float(operator["max_fd_cond"]),
    }

    rows = [
        make_result_row(base, "raw_rbf_fd", operator["raw_metrics"], raw_errors),
        make_result_row(
            base,
            "corrected_pathA_sbp",
            operator["corrected_metrics"],
            corrected_errors,
        ),
    ]

    for row in rows:
        print(
            f"{row['operator']}: grad error={row['grad_rel_error']:.6e}, "
            f"SBP_x={row['SBP_x']:.3e}, poly_x={row['poly_x']:.3e}"
        )

    return rows


# ============================================================
# Summaries and plots
# ============================================================

def add_observed_rates(rows):
    for domain in sorted({row["domain"] for row in rows}):
        for p in sorted({row["p"] for row in rows if row["domain"] == domain}):
            for operator_name in sorted({row["operator"] for row in rows}):
                group = [
                    row for row in rows
                    if row["domain"] == domain
                    and row["p"] == p
                    and row["operator"] == operator_name
                    and row["status"] == "ok"
                ]
                group.sort(key=lambda row: safe_float(row["h"]), reverse=True)

                for prev, curr in zip(group[:-1], group[1:]):
                    curr["observed_rate"] = convergence_rate(
                        safe_float(prev["grad_rel_error"]),
                        safe_float(curr["grad_rel_error"]),
                        safe_float(prev["h"]),
                        safe_float(curr["h"]),
                    )


def build_summary(rows):
    summary = []

    keys = sorted({
        (row["domain"], row["p"], row["operator"])
        for row in rows
    })

    for domain, p, operator_name in keys:
        group = [
            row for row in rows
            if row["domain"] == domain
            and row["p"] == p
            and row["operator"] == operator_name
            and row["status"] == "ok"
        ]
        group.sort(key=lambda row: safe_float(row["h"]), reverse=True)

        if not group:
            continue

        summary.append({
            "domain": domain,
            "p": p,
            "operator": operator_name,
            "expected_order": p,
            "num_successful_refinements": len(group),
            "N_coarsest": int(group[0]["N"]),
            "N_finest": int(group[-1]["N"]),
            "h_coarsest": float(group[0]["h"]),
            "h_finest": float(group[-1]["h"]),
            "error_coarsest": float(group[0]["grad_rel_error"]),
            "error_finest": float(group[-1]["grad_rel_error"]),
            "fitted_order": fitted_order(group),
            "last_step_rate": safe_float(group[-1]["observed_rate"]),
            "max_SBP_x": float(np.nanmax([safe_float(row["SBP_x"]) for row in group])),
            "max_SBP_y": float(np.nanmax([safe_float(row["SBP_y"]) for row in group])),
            "max_poly_x": float(np.nanmax([safe_float(row["poly_x"]) for row in group])),
            "max_poly_y": float(np.nanmax([safe_float(row["poly_y"]) for row in group])),
            "max_mass_condition": float(np.nanmax([safe_float(row["mass_condition"]) for row in group])),
            "max_fd_cond": float(np.nanmax([safe_float(row["max_fd_cond"]) for row in group])),
        })

    return summary


def plot_error_vs_h(rows):
    paths = []
    ok_rows = [row for row in rows if row["status"] == "ok"]

    for domain in sorted({row["domain"] for row in ok_rows}):
        fig, ax = plt.subplots(figsize=(7.5, 5.2))
        fig.patch.set_facecolor("white")

        for operator_name, style in [
            ("raw_rbf_fd", "o-"),
            ("corrected_pathA_sbp", "s--"),
        ]:
            for p in sorted({row["p"] for row in ok_rows if row["domain"] == domain}):
                group = [
                    row for row in ok_rows
                    if row["domain"] == domain
                    and row["p"] == p
                    and row["operator"] == operator_name
                ]
                group.sort(key=lambda row: safe_float(row["h"]), reverse=True)
                if not group:
                    continue

                h = np.array([safe_float(row["h"]) for row in group])
                err = np.array([safe_float(row["grad_rel_error"]) for row in group])
                ax.loglog(h, err, style, linewidth=1.5, label=f"{operator_name}, p={p}")

        ax.set_xlabel("nominal h = sqrt(area / N)")
        ax.set_ylabel("relative M_L gradient error")
        ax.set_title(f"Phase 2d operator order: {domain}")
        ax.grid(True, which="both", alpha=0.28)
        ax.legend(fontsize=8)

        fig.tight_layout()
        path = save_figure(fig, OUTPUT_DIR / f"phase2d_operator_order_{domain}.png")
        paths.append(path)
        plt.close(fig)

    return paths


def plot_observed_orders(summary):
    if not summary:
        return None

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    fig.patch.set_facecolor("white")

    labels = [
        f"{row['domain']}\np={row['p']}\n{row['operator'].replace('_', ' ')}"
        for row in summary
    ]
    x = np.arange(len(summary))
    fitted = np.array([safe_float(row["fitted_order"]) for row in summary])
    expected = np.array([safe_float(row["expected_order"]) for row in summary])

    ax.bar(x, fitted, width=0.62, color="#2563EB", label="fitted order")
    ax.plot(x, expected, "k.", markersize=8, label="polynomial degree p")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("observed order")
    ax.set_title("Phase 2d fitted discrete-operator order")
    ax.grid(True, axis="y", alpha=0.28)
    ax.legend()

    fig.tight_layout()
    path = save_figure(fig, OUTPUT_DIR / "phase2d_fitted_orders.png")
    plt.close(fig)
    return path


def plot_algebra_residuals(rows):
    ok_rows = [
        row for row in rows
        if row["status"] == "ok" and row["operator"] == "corrected_pathA_sbp"
    ]
    if not ok_rows:
        return None

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    fig.patch.set_facecolor("white")

    for metric, style in [
        ("SBP_x", "o-"),
        ("poly_x", "s--"),
        ("Qx_relative_change", "^-"),
    ]:
        vals = []
        labels = []
        for row in ok_rows:
            labels.append(f"{row['domain']} p={row['p']} N={row['N']}")
            vals.append(safe_float(row[metric]))
        ax.semilogy(np.arange(len(vals)), vals, style, linewidth=1.4, label=metric)

    ax.set_xticks(np.arange(len(ok_rows)))
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=7)
    ax.set_ylabel("relative residual / change")
    ax.set_title("Phase 2d corrected operator algebra checks")
    ax.grid(True, axis="y", which="both", alpha=0.28)
    ax.legend()

    fig.tight_layout()
    path = save_figure(fig, OUTPUT_DIR / "phase2d_corrected_algebra_residuals.png")
    plt.close(fig)
    return path


def write_report(summary, paths):
    lines = []
    lines.append("Phase 2d - Discrete Operator Order Test")
    lines.append("=" * 44)
    lines.append("")
    lines.append("Closest corrected phase: Phase 2c.")
    lines.append("New alphabetic phase: Phase 2d.")
    lines.append("")
    lines.append("Purpose:")
    lines.append("  Test the order of the discrete derivative operator itself.")
    lines.append("  The test applies raw and corrected Dx,Dy to a smooth manufactured field")
    lines.append("  and fits the M_L-norm gradient error as h is refined.")
    lines.append("")
    lines.append(f"PHS order: {PHS_ORDER}")
    lines.append(f"Polynomial degrees: {', '.join(str(p) for p in POLY_DEGREES)}")
    lines.append(f"Domains: {', '.join(DOMAINS)}")
    lines.append("")
    lines.append("Summary:")

    for row in summary:
        lines.append(
            "  "
            f"{row['domain']}, p={row['p']}, {row['operator']}: "
            f"fitted={safe_float(row['fitted_order']):.6g}, "
            f"last={safe_float(row['last_step_rate']):.6g}, "
            f"finest error={safe_float(row['error_finest']):.6e}"
        )

    lines.append("")
    lines.append("Outputs:")
    for name, path in paths.items():
        if path is None:
            continue
        if isinstance(path, list):
            for item in path:
                lines.append(f"  {name}: {item}")
        else:
            lines.append(f"  {name}: {path}")

    report_path = OUTPUT_DIR / "phase2d_report.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
        f.write("\n")

    return report_path


# ============================================================
# Main
# ============================================================

def main():
    print("\n" + "=" * 72)
    print("PHASE 2d - DISCRETE OPERATOR ORDER TEST")
    print("=" * 72)
    print("Closest corrected phase : Phase 2c")
    print("New alphabetic phase    : Phase 2d")
    print(f"PHS order               : {PHS_ORDER}")
    print(f"Polynomial degrees      : {POLY_DEGREES}")
    print(f"Domains                 : {DOMAINS}")
    print(f"Output folder           : {OUTPUT_DIR}")
    print("=" * 72)

    rows = []

    for domain_type in DOMAINS:
        if domain_type not in N_TARGETS:
            raise ValueError(f"No PHASE2D_N_TARGETS entry for domain {domain_type}")

        for poly_degree in POLY_DEGREES:
            for n_target in N_TARGETS[domain_type]:
                try:
                    rows.extend(run_one_case(domain_type, poly_degree, n_target))
                except Exception as exc:
                    print(f"FAILED: domain={domain_type}, p={poly_degree}, N={n_target}: {exc}")
                    rows.extend(failure_rows(domain_type, poly_degree, n_target, exc))

    add_observed_rates(rows)
    summary = build_summary(rows)

    raw_csv = write_csv(
        OUTPUT_DIR / "phase2d_operator_order_raw.csv",
        rows,
        RAW_FIELDS,
    )
    summary_csv = write_csv(
        OUTPUT_DIR / "phase2d_operator_order_summary.csv",
        summary,
        SUMMARY_FIELDS,
    )

    paths = {
        "raw_csv": raw_csv,
        "summary_csv": summary_csv,
        "error_figures": plot_error_vs_h(rows),
        "orders_figure": plot_observed_orders(summary),
        "algebra_figure": plot_algebra_residuals(rows),
    }
    report = write_report(summary, paths)

    print("\n" + "=" * 72)
    print("PHASE 2d COMPLETE")
    print("=" * 72)
    print("domain | p | operator | fitted order | last rate | finest error")
    print("-" * 72)
    for row in summary:
        print(
            f"{row['domain']} | "
            f"{row['p']} | "
            f"{row['operator']} | "
            f"{safe_float(row['fitted_order']):.3f} | "
            f"{safe_float(row['last_step_rate']):.3f} | "
            f"{safe_float(row['error_finest']):.3e}"
        )
    print("-" * 72)
    print(f"report -> {report}")


if __name__ == "__main__":
    main()
