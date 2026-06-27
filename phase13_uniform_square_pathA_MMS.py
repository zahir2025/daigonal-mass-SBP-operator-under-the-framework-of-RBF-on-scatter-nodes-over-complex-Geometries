"""
Phase 13 - Uniform Unit-Square Nodes With The Current Path A SBP-SAT Scheme
==========================================================================

This phase keeps the numerical scheme from the steady Path A work unchanged:

    1. diagonal lumped mass from clipped Voronoi cells,
    2. raw PHS RBF-FD derivative matrices,
    3. Path A M_L-compatible SBP projection,
    4. full compatible SAT steady advection solve.

The only intended change is the test geometry:

    scattered physical nodes  ->  uniform cell-centered nodes,
    previous curved domains   ->  unit square [0,1]^2.

The MMS is the same as Phase 6:

    u_exact = exp(x+y)
    lambda = [1,1]
    lambda . grad(u) = f = 2 exp(x+y)

Outputs:
    outputs_phase13_uniform_square/
        phase13_uniform_square_raw.csv
        phase13_uniform_square_summary.csv
        phase13_uniform_square_geometry.png
        phase13_uniform_square_error_vs_N.png
        phase13_uniform_square_error_vs_h.png
        phase13_uniform_square_observed_orders.png
        phase13_uniform_square_diagnostics.png
        phase13_uniform_square_summary_table.png
        phase13_uniform_square_field_error.png
"""

from pathlib import Path
import csv
import math
from datetime import datetime

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from shapely.geometry import box
from shapely.ops import unary_union

import phase3c_pathA_minimal_SAT_steady_MMS as steady


# ============================================================
# Manual configuration
# ============================================================

DOMAIN_TYPE = "unit_square_uniform"

POLY_DEGREES = list(range(1, 8))

# Cell-centered tensor grids.  n_grid=32 gives N=1024 nodes, keeping the run
# useful but still moderate for the dense Path A correction.
N_GRID_VALUES = [16, 24, 32,64]

PHS_ORDER = 5
STENCIL_FACTOR = 3.5
STENCIL_MINIMUM = 25

# Same full compatible SAT used in the main convergence phase.
SAT_THETA = 1.0

FIELD_P = 5
FIELD_N_GRID = 32
GEOMETRY_N_GRID = 16

FIG_DPI = 190

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs_phase13_uniform_square"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

UNIT_SQUARE = box(0.0, 0.0, 1.0, 1.0)


RAW_FIELDS = [
    "status",
    "message",
    "domain",
    "p",
    "n_grid",
    "N",
    "h",
    "theta",
    "Np",
    "Nc",
    "phs_order",
    "rel_error",
    "abs_error",
    "observed_rate",
    "linear_residual",
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
    "num_successful_refinements",
    "n_grid_coarsest",
    "n_grid_finest",
    "N_coarsest",
    "N_finest",
    "h_coarsest",
    "h_finest",
    "error_coarsest",
    "error_finest",
    "last_step_rate",
    "fitted_order",
    "max_SBP_x",
    "max_SBP_y",
    "max_poly_x",
    "max_poly_y",
    "max_linear_residual",
    "max_mass_condition",
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
        row for row in rows
        if row["status"] == "ok"
        and finite_positive(safe_float(row["h"]))
        and finite_positive(safe_float(row["rel_error"]))
    ]

    if len(good) < 2:
        return np.nan

    h = np.array([safe_float(row["h"]) for row in good], dtype=float)
    err = np.array([safe_float(row["rel_error"]) for row in good], dtype=float)
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


def uniform_square_nodes(n_grid):
    coord = (np.arange(n_grid, dtype=float) + 0.5) / n_grid
    x, y = np.meshgrid(coord, coord, indexing="xy")

    return np.column_stack([x.ravel(), y.ravel()])


def cell_line_segments(cells):
    segments = []

    for cell in cells:
        for poly in steady.geometry_parts(cell):
            x, y = poly.exterior.xy
            pts = np.column_stack([x, y])
            segments.extend(np.stack([pts[:-1], pts[1:]], axis=1))

    return segments


def failure_row(p, n_grid, message):
    h = 1.0 / n_grid
    N = n_grid * n_grid
    Np = steady.polynomial_term_count_2d(p)
    Nc = steady.choose_stencil_size(p, STENCIL_FACTOR, STENCIL_MINIMUM)

    row = {field: np.nan for field in RAW_FIELDS}
    row.update({
        "status": "failed",
        "message": str(message),
        "domain": DOMAIN_TYPE,
        "p": p,
        "n_grid": n_grid,
        "N": N,
        "h": h,
        "theta": SAT_THETA,
        "Np": Np,
        "Nc": Nc,
        "phs_order": PHS_ORDER,
    })

    return row


# ============================================================
# One MMS solve
# ============================================================

def run_one_case(p, n_grid, keep_fields=False):
    points = uniform_square_nodes(n_grid)
    cells = steady.clipped_voronoi(points, UNIT_SQUARE)
    weights = steady.lumped_mass_matrix(cells)

    if np.any(weights <= 0.0):
        raise RuntimeError("nonpositive Voronoi mass weight found")

    N = points.shape[0]
    h = 1.0 / n_grid

    Np = steady.polynomial_term_count_2d(p)
    Nc = steady.choose_stencil_size(p, STENCIL_FACTOR, STENCIL_MINIMUM)

    if Nc >= N:
        raise RuntimeError(f"Nc={Nc} >= N={N}; increase N_GRID_VALUES")

    area_error = abs(float(np.sum(weights)) - UNIT_SQUARE.area) / UNIT_SQUARE.area
    union_error = abs(unary_union(cells).area - UNIT_SQUARE.area) / UNIT_SQUARE.area

    print("-" * 86)
    print(f"unit square uniform, p={p}, grid={n_grid}x{n_grid}, N={N}, h={h:.6e}, Nc={Nc}")

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

    print(
        f"  rel_error={sol['rel_err']:.6e}, "
        f"linear_res={sol['rel_res']:.3e}, "
        f"SBP_x={SBP_x:.3e}, poly_x={poly_x:.3e}, "
        f"mass_cond={np.max(weights) / np.min(weights):.3e}"
    )

    row = {
        "status": "ok",
        "message": "",
        "domain": DOMAIN_TYPE,
        "p": p,
        "n_grid": n_grid,
        "N": N,
        "h": h,
        "theta": SAT_THETA,
        "Np": Np,
        "Nc": Nc,
        "phs_order": PHS_ORDER,
        "rel_error": float(sol["rel_err"]),
        "abs_error": float(sol["abs_err"]),
        "observed_rate": np.nan,
        "linear_residual": float(sol["rel_res"]),
        "min_eig_H": float(sol["min_eig_H"]),
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

    fields = None
    if keep_fields:
        fields = {
            "points": points,
            "n_grid": n_grid,
            "p": p,
            "u_exact": sol["u_ex"],
            "u_num": sol["u_h"],
            "error": sol["err"],
            "rel_error": sol["rel_err"],
        }

    return row, fields


def add_observed_rates(rows):
    for p in POLY_DEGREES:
        group = [row for row in rows if row["p"] == p]
        group.sort(key=lambda row: safe_float(row["n_grid"]))

        previous = None
        for row in group:
            if row["status"] != "ok":
                continue

            if previous is None:
                row["observed_rate"] = np.nan
            else:
                row["observed_rate"] = convergence_rate(
                    safe_float(previous["rel_error"]),
                    safe_float(row["rel_error"]),
                    safe_float(previous["h"]),
                    safe_float(row["h"]),
                )

            previous = row


def build_summary(rows):
    summary = []

    for p in POLY_DEGREES:
        good = [row for row in rows if row["p"] == p and row["status"] == "ok"]
        good.sort(key=lambda row: safe_float(row["n_grid"]))

        if not good:
            summary.append({
                "domain": DOMAIN_TYPE,
                "p": p,
                "theta": SAT_THETA,
                "num_successful_refinements": 0,
                "n_grid_coarsest": np.nan,
                "n_grid_finest": np.nan,
                "N_coarsest": np.nan,
                "N_finest": np.nan,
                "h_coarsest": np.nan,
                "h_finest": np.nan,
                "error_coarsest": np.nan,
                "error_finest": np.nan,
                "last_step_rate": np.nan,
                "fitted_order": np.nan,
                "max_SBP_x": np.nan,
                "max_SBP_y": np.nan,
                "max_poly_x": np.nan,
                "max_poly_y": np.nan,
                "max_linear_residual": np.nan,
                "max_mass_condition": np.nan,
                "max_fd_cond": np.nan,
            })
            continue

        summary.append({
            "domain": DOMAIN_TYPE,
            "p": p,
            "theta": SAT_THETA,
            "num_successful_refinements": len(good),
            "n_grid_coarsest": good[0]["n_grid"],
            "n_grid_finest": good[-1]["n_grid"],
            "N_coarsest": good[0]["N"],
            "N_finest": good[-1]["N"],
            "h_coarsest": good[0]["h"],
            "h_finest": good[-1]["h"],
            "error_coarsest": good[0]["rel_error"],
            "error_finest": good[-1]["rel_error"],
            "last_step_rate": good[-1]["observed_rate"] if len(good) >= 2 else np.nan,
            "fitted_order": fitted_order(good),
            "max_SBP_x": float(np.nanmax([safe_float(row["SBP_x"]) for row in good])),
            "max_SBP_y": float(np.nanmax([safe_float(row["SBP_y"]) for row in good])),
            "max_poly_x": float(np.nanmax([safe_float(row["poly_x"]) for row in good])),
            "max_poly_y": float(np.nanmax([safe_float(row["poly_y"]) for row in good])),
            "max_linear_residual": float(np.nanmax([
                safe_float(row["linear_residual"]) for row in good
            ])),
            "max_mass_condition": float(np.nanmax([
                safe_float(row["mass_condition"]) for row in good
            ])),
            "max_fd_cond": float(np.nanmax([safe_float(row["max_fd_cond"]) for row in good])),
        })

    return summary


# ============================================================
# Plotting
# ============================================================

def plot_uniform_geometry():
    points = uniform_square_nodes(GEOMETRY_N_GRID)
    cells = steady.clipped_voronoi(points, UNIT_SQUARE)
    weights = steady.lumped_mass_matrix(cells)

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.5), constrained_layout=True)
    fig.patch.set_facecolor("white")

    axes[0].scatter(
        points[:, 0],
        points[:, 1],
        s=16,
        color="#0F766E",
        edgecolor="white",
        linewidth=0.25,
    )
    axes[0].set_title("Uniform cell-centered nodes", fontweight="bold")

    axes[1].add_collection(
        LineCollection(cell_line_segments(cells), colors="#2563EB", linewidths=0.35, alpha=0.8)
    )
    axes[1].scatter(points[:, 0], points[:, 1], s=5, color="#111827", alpha=0.7)
    axes[1].set_title("Clipped Voronoi mass cells", fontweight="bold")

    sc = axes[2].scatter(points[:, 0], points[:, 1], c=weights, s=18, cmap="viridis")
    axes[2].set_title("Diagonal mass weights", fontweight="bold")
    fig.colorbar(sc, ax=axes[2], shrink=0.82)

    for ax in axes:
        ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], color="#111827", linewidth=1.0)
        ax.set_aspect("equal")
        ax.set_xlim(-0.04, 1.04)
        ax.set_ylim(-0.04, 1.04)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True, alpha=0.18)

    fig.suptitle(
        f"Phase 13 geometry: unit square, {GEOMETRY_N_GRID}x{GEOMETRY_N_GRID} uniform nodes",
        fontsize=13,
        fontweight="bold",
    )

    path = OUTPUT_DIR / "phase13_uniform_square_geometry.png"
    path = save_figure(fig, path)
    plt.close(fig)

    return path


def plot_error_vs_N(rows):
    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    fig.patch.set_facecolor("white")

    for p in POLY_DEGREES:
        group = [row for row in rows if row["p"] == p and row["status"] == "ok"]
        group.sort(key=lambda row: safe_float(row["N"]))
        if not group:
            continue

        N = np.array([safe_float(row["N"]) for row in group], dtype=float)
        err = np.array([safe_float(row["rel_error"]) for row in group], dtype=float)

        ax.loglog(N, err, "o-", linewidth=1.6, markersize=4.5, label=f"p={p}")

    ax.set_xlabel("number of nodes N = n^2")
    ax.set_ylabel("relative M_L solution error")
    ax.set_title("Uniform unit-square steady MMS error vs N")
    ax.grid(True, which="both", alpha=0.30)
    ax.legend(ncol=2, fontsize=8)

    fig.tight_layout()
    path = OUTPUT_DIR / "phase13_uniform_square_error_vs_N.png"
    path = save_figure(fig, path)
    plt.close(fig)

    return path


def plot_error_vs_h(rows):
    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    fig.patch.set_facecolor("white")

    for p in POLY_DEGREES:
        group = [row for row in rows if row["p"] == p and row["status"] == "ok"]
        group.sort(key=lambda row: safe_float(row["h"]), reverse=True)
        if not group:
            continue

        h = np.array([safe_float(row["h"]) for row in group], dtype=float)
        err = np.array([safe_float(row["rel_error"]) for row in group], dtype=float)

        ax.loglog(h, err, "o-", linewidth=1.6, markersize=4.5, label=f"p={p}")

    ax.invert_xaxis()
    ax.set_xlabel("uniform spacing h = 1 / n")
    ax.set_ylabel("relative M_L solution error")
    ax.set_title("Uniform unit-square steady MMS error vs h")
    ax.grid(True, which="both", alpha=0.30)
    ax.legend(ncol=2, fontsize=8)

    fig.tight_layout()
    path = OUTPUT_DIR / "phase13_uniform_square_error_vs_h.png"
    path = save_figure(fig, path)
    plt.close(fig)

    return path


def plot_observed_orders(summary):
    p = np.array([int(row["p"]) for row in summary], dtype=int)
    fitted = np.array([safe_float(row["fitted_order"]) for row in summary], dtype=float)
    last = np.array([safe_float(row["last_step_rate"]) for row in summary], dtype=float)

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    fig.patch.set_facecolor("white")

    ax.plot(p, fitted, "o-", linewidth=1.8, label="fit over grids")
    ax.plot(p, last, "s--", linewidth=1.6, label="last refinement step")
    ax.plot(p, p, "k:", linewidth=1.2, label="reference order p")

    ax.set_xlabel("Path A polynomial degree p")
    ax.set_ylabel("observed order")
    ax.set_title("Observed uniform-square convergence orders")
    ax.set_xticks(p)
    ax.grid(True, alpha=0.30)
    ax.legend(fontsize=8)

    fig.tight_layout()
    path = OUTPUT_DIR / "phase13_uniform_square_observed_orders.png"
    path = save_figure(fig, path)
    plt.close(fig)

    return path


def plot_diagnostics(summary):
    p = np.array([int(row["p"]) for row in summary], dtype=int)
    sbp = np.array([
        max(safe_float(row["max_SBP_x"]), safe_float(row["max_SBP_y"]))
        for row in summary
    ])
    poly = np.array([
        max(safe_float(row["max_poly_x"]), safe_float(row["max_poly_y"]))
        for row in summary
    ])
    linear = np.array([safe_float(row["max_linear_residual"]) for row in summary])
    mass_cond = np.array([safe_float(row["max_mass_condition"]) - 1.0 for row in summary])

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)
    fig.patch.set_facecolor("white")

    axes[0].semilogy(p, np.maximum(sbp, 1.0e-18), "o-", linewidth=1.7, label="SBP residual")
    axes[0].semilogy(p, np.maximum(poly, 1.0e-18), "s-", linewidth=1.7, label="polynomial residual")
    axes[0].semilogy(
        p,
        np.maximum(linear, 1.0e-18),
        "^-",
        linewidth=1.7,
        label="linear solve residual",
    )
    axes[0].set_xlabel("p")
    axes[0].set_ylabel("relative residual")
    axes[0].set_title("Algebraic diagnostics", fontweight="bold")
    axes[0].grid(True, which="both", alpha=0.30)
    axes[0].legend(fontsize=8)

    axes[1].semilogy(
        p,
        np.maximum(np.abs(mass_cond), 1.0e-18),
        "o-",
        linewidth=1.7,
        color="#0F766E",
    )
    axes[1].set_xlabel("p")
    axes[1].set_ylabel("max(mass condition - 1)")
    axes[1].set_title("Uniform mass check", fontweight="bold")
    axes[1].grid(True, which="both", alpha=0.30)

    fig.suptitle("Phase 13 uniform-square scheme diagnostics", fontsize=13, fontweight="bold")

    path = OUTPUT_DIR / "phase13_uniform_square_diagnostics.png"
    path = save_figure(fig, path)
    plt.close(fig)

    return path


def plot_summary_table(summary):
    rows = []

    for row in summary:
        rows.append([
            f"{int(row['p'])}",
            f"{safe_float(row['N_finest']):.0f}",
            f"{safe_float(row['error_finest']):.2e}",
            f"{safe_float(row['fitted_order']):.2f}",
            f"{safe_float(row['last_step_rate']):.2f}",
            f"{safe_float(row['max_SBP_x']):.1e}",
            f"{safe_float(row['max_linear_residual']):.1e}",
        ])

    fig, ax = plt.subplots(figsize=(11.2, 3.9))
    fig.patch.set_facecolor("white")
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=[
            "p",
            "finest N",
            "finest error",
            "fit order",
            "last order",
            "max SBP_x",
            "max linear res.",
        ],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.35)

    for (i, j), cell in table.get_celld().items():
        cell.set_edgecolor("#CBD5E1")
        cell.set_linewidth(0.6)
        if i == 0:
            cell.set_facecolor("#E2E8F0")
            cell.set_text_props(weight="bold", color="#111827")
        else:
            cell.set_facecolor("#F8FAFC" if i % 2 == 0 else "white")

    ax.set_title(
        "Phase 13 uniform unit-square MMS error display",
        fontsize=14,
        fontweight="bold",
        pad=14,
    )

    path = OUTPUT_DIR / "phase13_uniform_square_summary_table.png"
    path = save_figure(fig, path)
    plt.close(fig)

    return path


def plot_field_error(fields):
    points = fields["points"]
    u_exact = fields["u_exact"]
    u_num = fields["u_num"]
    err = fields["error"]

    vmin = min(float(np.min(u_exact)), float(np.min(u_num)))
    vmax = max(float(np.max(u_exact)), float(np.max(u_num)))
    emax = max(float(np.max(np.abs(err))), 1.0e-16)

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.6), constrained_layout=True)
    fig.patch.set_facecolor("white")

    panels = [
        (u_exact, "Exact solution", "viridis", vmin, vmax),
        (u_num, "Path A SBP-SAT solution", "viridis", vmin, vmax),
        (err, "Nodal error", "coolwarm", -emax, emax),
    ]

    for ax, (values, title, cmap, lo, hi) in zip(axes, panels):
        sc = ax.scatter(
            points[:, 0],
            points[:, 1],
            c=values,
            s=15,
            cmap=cmap,
            vmin=lo,
            vmax=hi,
            linewidths=0.0,
        )
        ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], color="#111827", linewidth=0.9)
        ax.set_title(title, fontweight="bold")
        ax.set_aspect("equal")
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(sc, ax=ax, shrink=0.83)

    fig.suptitle(
        (
            "Uniform unit-square steady advection MMS "
            f"(p={fields['p']}, grid={fields['n_grid']}x{fields['n_grid']}, "
            f"rel. error={fields['rel_error']:.3e})"
        ),
        fontsize=13,
        fontweight="bold",
    )

    path = OUTPUT_DIR / "phase13_uniform_square_field_error.png"
    path = save_figure(fig, path)
    plt.close(fig)

    return path


# ============================================================
# Main
# ============================================================

def main():
    steady.LAMBDA_VEC = np.array([1.0, 1.0])

    print("=" * 86)
    print("PHASE 13 - UNIFORM UNIT-SQUARE NODES WITH CURRENT PATH A SBP-SAT")
    print("=" * 86)
    print("domain              = [0,1]^2")
    print("node type           = uniform cell-centered tensor grid")
    print(f"grid values         = {N_GRID_VALUES}")
    print(f"polynomial degrees  = {POLY_DEGREES}")
    print(f"PHS order           = {PHS_ORDER}")
    print(f"SAT theta           = {SAT_THETA}")
    print(f"lambda              = {steady.LAMBDA_VEC.tolist()}")
    print(f"output folder       = {OUTPUT_DIR}")
    print("=" * 86)

    geometry_plot = plot_uniform_geometry()

    rows = []
    field_data = None

    for p in POLY_DEGREES:
        print("\n" + "#" * 86)
        print(f"POLYNOMIAL DEGREE p={p}")
        print("#" * 86)

        for n_grid in N_GRID_VALUES:
            keep_fields = p == FIELD_P and n_grid == FIELD_N_GRID
            try:
                row, fields = run_one_case(p, n_grid, keep_fields=keep_fields)
                if fields is not None:
                    field_data = fields
            except Exception as exc:
                print("-" * 86)
                print(f"FAILED: p={p}, grid={n_grid}x{n_grid}")
                print(f"  {exc}")
                row = failure_row(p, n_grid, exc)

            rows.append(row)

    add_observed_rates(rows)
    summary = build_summary(rows)

    raw_csv = write_csv(
        OUTPUT_DIR / "phase13_uniform_square_raw.csv",
        rows,
        RAW_FIELDS,
    )
    summary_csv = write_csv(
        OUTPUT_DIR / "phase13_uniform_square_summary.csv",
        summary,
        SUMMARY_FIELDS,
    )

    error_N_plot = plot_error_vs_N(rows)
    error_h_plot = plot_error_vs_h(rows)
    orders_plot = plot_observed_orders(summary)
    diagnostics_plot = plot_diagnostics(summary)
    table_plot = plot_summary_table(summary)

    if field_data is None:
        field_plot = None
    else:
        field_plot = plot_field_error(field_data)

    print("\n" + "=" * 86)
    print("PHASE 13 UNIFORM UNIT-SQUARE RUN COMPLETE")
    print("=" * 86)
    print(f"raw table       -> {raw_csv}")
    print(f"summary table   -> {summary_csv}")
    print(f"geometry plot   -> {geometry_plot}")
    print(f"error vs N      -> {error_N_plot}")
    print(f"error vs h      -> {error_h_plot}")
    print(f"order plot      -> {orders_plot}")
    print(f"diagnostics     -> {diagnostics_plot}")
    print(f"error display   -> {table_plot}")
    print(f"field/error     -> {field_plot}")
    print("-" * 86)
    print("p | levels | finest grid | finest error | fitted order | last rate | max SBP | max res")
    print("-" * 86)

    for row in summary:
        print(
            f"{row['p']} | "
            f"{row['num_successful_refinements']} | "
            f"{row['n_grid_finest']}x{row['n_grid_finest']} | "
            f"{safe_float(row['error_finest']):.3e} | "
            f"{safe_float(row['fitted_order']):.3f} | "
            f"{safe_float(row['last_step_rate']):.3f} | "
            f"{safe_float(row['max_SBP_x']):.3e} | "
            f"{safe_float(row['max_linear_residual']):.3e}"
        )

    print("-" * 86)


if __name__ == "__main__":
    main()
