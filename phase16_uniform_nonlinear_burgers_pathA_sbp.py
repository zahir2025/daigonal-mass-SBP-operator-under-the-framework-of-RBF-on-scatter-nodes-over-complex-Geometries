"""
Phase 16 - Uniform-Node Nonlinear Burgers With Path A Entropy-Conservative SBP
=============================================================================

This phase follows the same construction trend as the earlier work:

    1. Generate uniform Cartesian nodes clipped to the selected physical domain.
    2. Build clipped Voronoi cells and the diagonal lumped mass M_L.
    3. Build raw RBF-FD derivative matrices.
    4. Project them to the Path A M_L-compatible SBP operators Qx, Qy.
    5. Test entropy-conservative Burgers flux differencing with our own
       operators.

The manufactured nonlinear spatial operator is

    u_t + d/dx (u^2/2) + d/dy (u^2/2) = s(x,y,t),

and this script checks the Path A SBP flux-differencing operator

    div_h(u)_i =
        2 / w_i sum_j Qx_ij f_ec(u_i, u_j)
      + 2 / w_i sum_j Qy_ij f_ec(u_i, u_j),

    f_ec(u_i, u_j) = (u_i^2 + u_i u_j + u_j^2) / 6,

against the exact divergence u (u_x + u_y).  This is a nonlinear verification
of the same diagonal-Voronoi Path A SBP method used in the rest of the project.

This phase is intentionally Phase 12 with only the scattered/cloud node
generation replaced by uniform clipped nodes.

Outputs:
    outputs_phase16_uniform_nonlinear_burgers/
        phase16_pathA_scheme_geometry_<domain>.png
        phase16_pathA_burgers_<domain>_raw.csv
        phase16_pathA_burgers_<domain>_summary.csv
        phase16_pathA_burgers_<domain>_error_vs_N.png
        phase16_pathA_burgers_<domain>_error_vs_h.png
        phase16_pathA_burgers_<domain>_observed_orders.png
        phase16_pathA_burgers_<domain>_conditioning.png
        phase16_pathA_burgers_<domain>_field_error.png
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
from shapely.geometry import Point
from shapely.ops import unary_union

import phase3c_pathA_minimal_SAT_steady_MMS as steady


# ============================================================
# Manual configuration
# ============================================================

# Phase 16 reports a uniform-node variant while using the same physical
# geometry as Phase 12.
BASE_DOMAIN_TYPE = "annulus"
DOMAIN_TYPE = "annulus_uniform"

POLY_DEGREES = list(range(1, 11))

# These match the refinement style used by Phases 6-8.  The p=7 stencil needs
# enough nodes, so keep the coarsest level comfortably above the stencil size.
NODE_REFINEMENTS_BY_DOMAIN = {
    "annulus_uniform": [400, 800, 1600, 3200, 6400],
}

PHS_ORDER = 5
STENCIL_FACTOR = 4.0
STENCIL_MINIMUM = 25

FIELD_P = 5
FIELD_N_TARGET = 1200
GEOMETRY_N_TARGET = 400

FIG_DPI = 180

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs_phase16_uniform_nonlinear_burgers"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


RAW_FIELDS = [
    "status",
    "message",
    "domain",
    "p",
    "N",
    "h",
    "Np",
    "Nc",
    "phs_order",
    "flux_rel_error",
    "flux_abs_error",
    "observed_rate",
    "exact_divergence_norm",
    "mass_balance_abs",
    "entropy_balance_abs",
    "SBP_x",
    "SBP_y",
    "poly_x",
    "poly_y",
    "constant_x",
    "constant_y",
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
    "num_successful_refinements",
    "N_coarsest",
    "N_finest",
    "h_coarsest",
    "h_finest",
    "error_coarsest",
    "error_finest",
    "last_step_rate",
    "fitted_order",
    "max_SBP_x",
    "max_poly_x",
    "max_constant_x",
    "max_constant_y",
    "max_mass_balance_abs",
    "max_entropy_balance_abs",
    "max_mass_condition",
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


def mass_norm(v, weights):
    return math.sqrt(max(float(np.sum(weights * v * v)), 0.0))


def relative_mass_norm(error, reference, weights):
    return mass_norm(error, weights) / max(mass_norm(reference, weights), 1.0e-14)


def fitted_order(rows):
    good = [
        r for r in rows
        if r["status"] == "ok"
        and finite_positive(safe_float(r["h"]))
        and finite_positive(safe_float(r["flux_rel_error"]))
    ]

    if len(good) < 2:
        return np.nan

    h = np.array([safe_float(r["h"]) for r in good], dtype=float)
    err = np.array([safe_float(r["flux_rel_error"]) for r in good], dtype=float)
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


def cell_line_segments(cells):
    segments = []
    for cell in cells:
        for poly in steady.geometry_parts(cell):
            x, y = poly.exterior.xy
            pts = np.column_stack([x, y])
            segments.extend(np.stack([pts[:-1], pts[1:]], axis=1))
    return segments


def uniform_nodes_in_domain(domain, N_target):
    h_target = math.sqrt(domain.area / N_target)
    minx, miny, maxx, maxy = domain.bounds
    width = maxx - minx
    height = maxy - miny

    nx = max(1, int(math.ceil(width / h_target)))
    ny = max(1, int(math.ceil(height / h_target)))
    h = min(width / nx, height / ny)

    xs = minx + (np.arange(nx, dtype=float) + 0.5) * h
    ys = miny + (np.arange(ny, dtype=float) + 0.5) * h
    candidates = np.array([
        (x, y)
        for y in ys
        for x in xs
        if x < maxx and y < maxy
    ])

    if candidates.size == 0:
        raise RuntimeError("uniform grid produced no candidate points")

    inside = np.array([
        domain.contains(Point(float(x), float(y)))
        for x, y in candidates
    ])
    points = candidates[inside]

    if points.shape[0] == 0:
        raise RuntimeError("uniform grid produced no interior points")

    return points, float(h), int(nx), int(ny)


# ============================================================
# Manufactured nonlinear Burgers field
# ============================================================

def exact_u_and_derivatives(points):
    x = points[:, 0]
    y = points[:, 1]

    A = np.pi * (x + 0.75 * y)
    B = np.pi * (2.0 * x - y)
    C = np.pi * (x * x + 0.50 * y)

    u = (
        1.0
        + 0.12 * np.sin(A)
        + 0.07 * np.cos(B)
        + 0.03 * np.sin(C)
    )

    ux = 0.12 * np.cos(A) * np.pi
    ux += -0.07 * np.sin(B) * (2.0 * np.pi)
    ux += 0.03 * np.cos(C) * (2.0 * np.pi * x)

    uy = 0.12 * np.cos(A) * (0.75 * np.pi)
    uy += 0.07 * np.sin(B) * np.pi
    uy += 0.03 * np.cos(C) * (0.50 * np.pi)

    return u, ux, uy


def exact_flux_divergence(points):
    u, ux, uy = exact_u_and_derivatives(points)
    return u * (ux + uy)


def burgers_entropy_conservative_flux(u_left, u_right):
    return (u_left * u_left + u_left * u_right + u_right * u_right) / 6.0


def nonlinear_pathA_flux_divergence(u, weights, Qx, Qy, chunk_size=256):
    """
    Path A SBP flux differencing with the Burgers Tadmor two-point flux.

    The dense Path A matrices are already available in this workflow.  The
    flux matrix is evaluated by row chunks so high-resolution runs do not need
    an extra full N-by-N dense allocation.
    """
    div = np.zeros_like(u)
    u_all = u[None, :]

    for start in range(0, len(u), chunk_size):
        stop = min(start + chunk_size, len(u))
        u_block = u[start:stop, None]
        F_block = burgers_entropy_conservative_flux(u_block, u_all)
        action = np.sum(Qx[start:stop, :] * F_block, axis=1)
        action += np.sum(Qy[start:stop, :] * F_block, axis=1)
        div[start:stop] = 2.0 * action / weights[start:stop]

    return div


def sbp_flux_balance_residuals(u, weights, Qx, Qy, Ex, Ey, div, chunk_size=256):
    """
    Conservative and entropy-conservative SBP balance residuals.

    On physical domains the boundary matrix E=Q+Q^T is generally nonzero, so
    the correct check is not zero production.  The correct check is that the
    volume flux differencing equals the pairwise SBP boundary flux.  If E=0,
    these same residuals reduce to closed/periodic mass and entropy
    conservation.
    """
    E = Ex + Ey
    psi = u * u * u / 6.0
    mass_boundary = 0.0
    entropy_boundary = 0.0
    u_all = u[None, :]
    psi_all = psi[None, :]

    for start in range(0, len(u), chunk_size):
        stop = min(start + chunk_size, len(u))
        u_block = u[start:stop, None]
        psi_block = psi[start:stop, None]
        F_block = burgers_entropy_conservative_flux(u_block, u_all)
        E_block = E[start:stop, :]

        mass_boundary += float(np.sum(E_block * F_block))
        entropy_boundary += float(np.sum(
            E_block * (
                0.5 * (u_block + u_all) * F_block
                - 0.5 * (psi_block + psi_all)
            )
        ))

    mass_volume = float(np.sum(weights * div))
    entropy_volume = float(np.sum(weights * u * div))

    return (
        abs(mass_volume - mass_boundary),
        abs(entropy_volume - entropy_boundary),
    )


# ============================================================
# One nonlinear spatial MMS case
# ============================================================

def failure_row(domain_type, domain_area, p, N_target, message):
    h_uniform = math.sqrt(domain_area / N_target) if domain_area > 0.0 else np.nan
    Np = steady.polynomial_term_count_2d(p)
    Nc = steady.choose_stencil_size(p, STENCIL_FACTOR, STENCIL_MINIMUM)

    row = {field: np.nan for field in RAW_FIELDS}
    row.update({
        "status": "failed",
        "message": str(message),
        "domain": domain_type,
        "p": p,
        "N": N_target,
        "h": h_uniform,
        "Np": Np,
        "Nc": Nc,
        "phs_order": PHS_ORDER,
    })
    return row


def build_pathA_case(domain_type, domain, p, N_target):
    points, h_uniform, nx, ny = uniform_nodes_in_domain(domain, N_target)
    cells = steady.clipped_voronoi(points, domain)
    weights = steady.lumped_mass_matrix(cells)

    if np.any(weights <= 0.0):
        raise RuntimeError("nonpositive Voronoi mass weight found")

    N = len(points)

    Np = steady.polynomial_term_count_2d(p)
    Nc = steady.choose_stencil_size(p, STENCIL_FACTOR, STENCIL_MINIMUM)

    if Nc >= N:
        raise RuntimeError(f"Nc={Nc} >= N={N}; increase node refinement levels")

    area_error = abs(float(np.sum(weights)) - domain.area) / max(abs(domain.area), 1.0e-14)
    union_error = abs(unary_union(cells).area - domain.area) / max(abs(domain.area), 1.0e-14)

    Dx_raw, Dy_raw, conds = steady.build_rbf_fd_derivative_matrices(
        points,
        Nc,
        PHS_ORDER,
        p,
    )

    sbp = steady.build_pathA_sbp_operators(points, weights, Dx_raw, Dy_raw, p)

    return {
        "points": points,
        "cells": cells,
        "weights": weights,
        "h": h_uniform,
        "nx": nx,
        "ny": ny,
        "Np": Np,
        "Nc": Nc,
        "area_error": area_error,
        "union_error": union_error,
        "max_fd_cond": float(np.nanmax(conds)),
        "sbp": sbp,
    }


def run_one_case(domain_type, domain, p, N_target):
    case = build_pathA_case(domain_type, domain, p, N_target)
    points = case["points"]
    weights = case["weights"]
    sbp = case["sbp"]

    Qx = sbp["Qx"]
    Qy = sbp["Qy"]
    Ex = sbp["Ex"]
    Ey = sbp["Ey"]
    P = sbp["P"]
    Px = sbp["Px"]
    Py = sbp["Py"]

    u, _, _ = exact_u_and_derivatives(points)
    div_num = nonlinear_pathA_flux_divergence(u, weights, Qx, Qy)
    div_exact = exact_flux_divergence(points)
    err = div_num - div_exact
    mass_balance_abs, entropy_balance_abs = sbp_flux_balance_residuals(
        u,
        weights,
        Qx,
        Qy,
        Ex,
        Ey,
        div_num,
    )

    one = np.ones(points.shape[0])
    weight_norm = max(np.linalg.norm(weights), 1.0e-14)

    SBP_x = steady.relative_residual(Qx + Qx.T, Ex)
    SBP_y = steady.relative_residual(Qy + Qy.T, Ey)
    poly_x = steady.relative_residual(Qx @ P, steady.weighted_matrix(weights, Px))
    poly_y = steady.relative_residual(Qy @ P, steady.weighted_matrix(weights, Py))
    constant_x = float(np.linalg.norm(Qx @ one) / weight_norm)
    constant_y = float(np.linalg.norm(Qy @ one) / weight_norm)

    flux_abs_error = mass_norm(err, weights)
    flux_rel_error = relative_mass_norm(err, div_exact, weights)
    exact_norm = mass_norm(div_exact, weights)

    print(
        f"  rel_flux_error={flux_rel_error:.6e}, "
        f"SBP_x={SBP_x:.3e}, poly_x={poly_x:.3e}, "
        f"mass_balance={mass_balance_abs:.3e}, "
        f"entropy_balance={entropy_balance_abs:.3e}, "
        f"mass_cond={np.max(weights) / np.min(weights):.3e}, "
        f"max_fd_cond={case['max_fd_cond']:.3e}"
    )

    return {
        "status": "ok",
        "message": "",
        "domain": domain_type,
        "p": p,
        "N": points.shape[0],
        "h": float(case["h"]),
        "Np": case["Np"],
        "Nc": case["Nc"],
        "phs_order": PHS_ORDER,
        "flux_rel_error": float(flux_rel_error),
        "flux_abs_error": float(flux_abs_error),
        "observed_rate": np.nan,
        "exact_divergence_norm": float(exact_norm),
        "mass_balance_abs": float(mass_balance_abs),
        "entropy_balance_abs": float(entropy_balance_abs),
        "SBP_x": float(SBP_x),
        "SBP_y": float(SBP_y),
        "poly_x": float(poly_x),
        "poly_y": float(poly_y),
        "constant_x": constant_x,
        "constant_y": constant_y,
        "Qx_relative_change": float(sbp["info_x"]["relative_change"]),
        "Qy_relative_change": float(sbp["info_y"]["relative_change"]),
        "min_diag": float(np.min(weights)),
        "max_diag": float(np.max(weights)),
        "mass_condition": float(np.max(weights) / np.min(weights)),
        "area_error": float(case["area_error"]),
        "union_error": float(case["union_error"]),
        "max_fd_cond": float(case["max_fd_cond"]),
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
                    safe_float(previous["flux_rel_error"]),
                    safe_float(row["flux_rel_error"]),
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
                "num_successful_refinements": 0,
                "N_coarsest": np.nan,
                "N_finest": np.nan,
                "h_coarsest": np.nan,
                "h_finest": np.nan,
                "error_coarsest": np.nan,
                "error_finest": np.nan,
                "last_step_rate": np.nan,
                "fitted_order": np.nan,
                "max_SBP_x": np.nan,
                "max_poly_x": np.nan,
                "max_constant_x": np.nan,
                "max_constant_y": np.nan,
                "max_mass_balance_abs": np.nan,
                "max_entropy_balance_abs": np.nan,
                "max_mass_condition": np.nan,
                "max_fd_cond": np.nan,
            })
            continue

        summary.append({
            "domain": domain_type,
            "p": p,
            "num_successful_refinements": len(good),
            "N_coarsest": good[0]["N"],
            "N_finest": good[-1]["N"],
            "h_coarsest": good[0]["h"],
            "h_finest": good[-1]["h"],
            "error_coarsest": good[0]["flux_rel_error"],
            "error_finest": good[-1]["flux_rel_error"],
            "last_step_rate": good[-1]["observed_rate"] if len(good) >= 2 else np.nan,
            "fitted_order": fitted_order(good),
            "max_SBP_x": float(np.nanmax([safe_float(r["SBP_x"]) for r in good])),
            "max_poly_x": float(np.nanmax([safe_float(r["poly_x"]) for r in good])),
            "max_constant_x": float(np.nanmax([safe_float(r["constant_x"]) for r in good])),
            "max_constant_y": float(np.nanmax([safe_float(r["constant_y"]) for r in good])),
            "max_mass_balance_abs": float(np.nanmax([
                safe_float(r["mass_balance_abs"]) for r in good
            ])),
            "max_entropy_balance_abs": float(np.nanmax([
                safe_float(r["entropy_balance_abs"]) for r in good
            ])),
            "max_mass_condition": float(np.nanmax([
                safe_float(r["mass_condition"]) for r in good
            ])),
            "max_fd_cond": float(np.nanmax([safe_float(r["max_fd_cond"]) for r in good])),
        })

    return summary


# ============================================================
# Plotting
# ============================================================

def plot_scheme_geometry(domain_type, domain):
    N_plot = min(GEOMETRY_N_TARGET, NODE_REFINEMENTS_BY_DOMAIN[domain_type][0])
    points, h_uniform, nx, ny = uniform_nodes_in_domain(domain, N_plot)
    cells = steady.clipped_voronoi(points, domain)
    weights = steady.lumped_mass_matrix(cells)

    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.8), constrained_layout=True)
    fig.patch.set_facecolor("white")

    for poly in steady.geometry_parts(domain):
        x, y = poly.exterior.xy
        axes[0].plot(x, y, color="#111827", linewidth=1.2)
        axes[1].plot(x, y, color="#111827", linewidth=1.0)
        for hole in poly.interiors:
            hx, hy = hole.xy
            axes[0].plot(hx, hy, color="#111827", linewidth=1.0)
            axes[1].plot(hx, hy, color="#111827", linewidth=0.9)

    axes[0].scatter(points[:, 0], points[:, 1], s=12, color="#0F766E", edgecolor="white", linewidth=0.25)
    axes[0].set_title("Uniform clipped nodes", fontweight="bold")

    segments = cell_line_segments(cells)
    axes[1].add_collection(LineCollection(segments, colors="#2563EB", linewidths=0.22, alpha=0.75))
    axes[1].scatter(points[:, 0], points[:, 1], s=4, color="#111827", alpha=0.65)
    axes[1].set_title("Clipped Voronoi mass cells", fontweight="bold")

    sc = axes[2].scatter(points[:, 0], points[:, 1], c=weights, s=16, cmap="viridis")
    axes[2].set_title("Diagonal mass weights", fontweight="bold")
    fig.colorbar(sc, ax=axes[2], shrink=0.83)

    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True, alpha=0.18)

    fig.suptitle(
        (
            f"Phase 16 Path A nonlinear Burgers geometry: {domain_type} "
            f"({nx}x{ny} grid, h={h_uniform:.3e})"
        ),
        fontsize=14,
        fontweight="bold",
    )

    path = OUTPUT_DIR / f"phase16_pathA_scheme_geometry_{domain_type}.png"
    path = save_figure(fig, path)
    plt.close(fig)
    return path


def plot_error_vs_N(rows, domain_type):
    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    fig.patch.set_facecolor("white")

    for p in POLY_DEGREES:
        group = [r for r in rows if r["p"] == p and r["status"] == "ok"]
        group.sort(key=lambda r: safe_float(r["N"]))
        if not group:
            continue

        N = np.array([safe_float(r["N"]) for r in group], dtype=float)
        err = np.array([safe_float(r["flux_rel_error"]) for r in group], dtype=float)
        ax.loglog(N, err, "o-", linewidth=1.6, markersize=4.5, label=f"p={p}")

    ax.set_xlabel("number of nodes N")
    ax.set_ylabel("relative M_L flux-divergence error")
    ax.set_title(f"Path A nonlinear Burgers MMS vs N: {domain_type}")
    ax.grid(True, which="both", alpha=0.30)
    ax.legend(ncol=2, fontsize=8)

    fig.tight_layout()
    path = OUTPUT_DIR / f"phase16_pathA_burgers_{domain_type}_error_vs_N.png"
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
        err = np.array([safe_float(r["flux_rel_error"]) for r in group], dtype=float)
        ax.loglog(h, err, "o-", linewidth=1.6, markersize=4.5, label=f"p={p}")

    ax.invert_xaxis()
    ax.set_xlabel("uniform background spacing h")
    ax.set_ylabel("relative M_L flux-divergence error")
    ax.set_title(f"Path A nonlinear Burgers MMS vs h: {domain_type}")
    ax.grid(True, which="both", alpha=0.30)
    ax.legend(ncol=2, fontsize=8)

    fig.tight_layout()
    path = OUTPUT_DIR / f"phase16_pathA_burgers_{domain_type}_error_vs_h.png"
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

    ax.set_xlabel("Path A polynomial degree p")
    ax.set_ylabel("observed order")
    ax.set_title(f"Observed nonlinear Burgers orders: {domain_type}")
    ax.set_xticks(p)
    ax.grid(True, alpha=0.30)
    ax.legend(fontsize=8)

    fig.tight_layout()
    path = OUTPUT_DIR / f"phase16_pathA_burgers_{domain_type}_observed_orders.png"
    path = save_figure(fig, path)
    plt.close(fig)
    return path


def plot_conditioning(summary, domain_type):
    p = np.array([int(r["p"]) for r in summary], dtype=int)
    mass_cond = np.array([safe_float(r["max_mass_condition"]) for r in summary], dtype=float)
    fd_cond = np.array([safe_float(r["max_fd_cond"]) for r in summary], dtype=float)

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    fig.patch.set_facecolor("white")

    ax.semilogy(p, mass_cond, "o-", linewidth=1.8, label="max mass condition")
    ax.semilogy(p, fd_cond, "s-", linewidth=1.8, label="max RBF-FD condition")

    ax.set_xlabel("Path A polynomial degree p")
    ax.set_ylabel("condition number")
    ax.set_title(f"Phase 16 conditioning by p: {domain_type}")
    ax.set_xticks(p)
    ax.grid(True, which="both", alpha=0.30)
    ax.legend(fontsize=8)

    fig.tight_layout()
    path = OUTPUT_DIR / f"phase16_pathA_burgers_{domain_type}_conditioning.png"
    path = save_figure(fig, path)
    plt.close(fig)
    return path


def plot_field_error(domain_type, domain):
    case = build_pathA_case(domain_type, domain, FIELD_P, FIELD_N_TARGET)
    points = case["points"]
    weights = case["weights"]
    sbp = case["sbp"]

    u, _, _ = exact_u_and_derivatives(points)
    div_exact = exact_flux_divergence(points)
    div_num = nonlinear_pathA_flux_divergence(u, weights, sbp["Qx"], sbp["Qy"])
    err = div_num - div_exact
    rel_err = relative_mass_norm(err, div_exact, weights)

    vmax = max(float(np.max(np.abs(div_exact))), float(np.max(np.abs(div_num))))
    emax = max(float(np.max(np.abs(err))), 1.0e-15)

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.7), constrained_layout=True)
    fig.patch.set_facecolor("white")

    panels = [
        (div_exact, "Exact nonlinear divergence", "viridis", -vmax, vmax),
        (div_num, "Path A SBP result", "viridis", -vmax, vmax),
        (err, "Nodal error", "coolwarm", -emax, emax),
    ]

    for ax, (data, title, cmap, vmin, vmax_plot) in zip(axes, panels):
        sc = ax.scatter(
            points[:, 0],
            points[:, 1],
            c=data,
            s=12,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax_plot,
            linewidths=0.0,
        )
        for poly in steady.geometry_parts(domain):
            x, y = poly.exterior.xy
            ax.plot(x, y, color="#111827", linewidth=0.8)
            for hole in poly.interiors:
                hx, hy = hole.xy
                ax.plot(hx, hy, color="#111827", linewidth=0.7)
        ax.set_title(title, fontweight="bold")
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(sc, ax=ax, shrink=0.82)

    fig.suptitle(
        (
            "Nonlinear Burgers entropy-conservative Path A SBP flux differencing "
            f"(p={FIELD_P}, N={points.shape[0]}, rel. error={rel_err:.3e})"
        ),
        fontsize=13,
        fontweight="bold",
    )

    path = OUTPUT_DIR / f"phase16_pathA_burgers_{domain_type}_field_error.png"
    path = save_figure(fig, path)
    plt.close(fig)
    return path, rel_err


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

    domain = steady.build_domain(BASE_DOMAIN_TYPE)
    node_refinements = NODE_REFINEMENTS_BY_DOMAIN[domain_type]

    print("=" * 94)
    print("PHASE 16 - UNIFORM-NODE NONLINEAR BURGERS WITH PATH A ENTROPY-CONSERVATIVE SBP")
    print("=" * 94)
    print("PDE          : u_t + (u^2/2)_x + (u^2/2)_y = s")
    print("Method       : Voronoi M_L + RBF-FD raw D + Path A SBP correction")
    print("Volume flux  : f_ec(u_i,u_j)=(u_i^2+u_i*u_j+u_j^2)/6")
    print("Discrete div : 2 M_L^{-1} ((Qx o F_ec)1 + (Qy o F_ec)1)")
    print(f"domain       : {domain_type} (physical geometry: {BASE_DOMAIN_TYPE})")
    print("node type    : uniform Cartesian cell centers clipped to domain")
    print(f"p values     : {POLY_DEGREES}")
    print(f"N targets    : {node_refinements}")
    print(f"output folder: {OUTPUT_DIR}")
    print("=" * 94)

    geometry_plot = plot_scheme_geometry(domain_type, domain)

    rows = []
    for p in POLY_DEGREES:
        print("\n" + "#" * 94)
        print(f"PATH A NONLINEAR BURGERS DEGREE p={p}")
        print("#" * 94)

        for N_target in node_refinements:
            print("-" * 94)
            print(f"domain={domain_type}, p={p}, N_target={N_target}")
            try:
                row = run_one_case(domain_type, domain, p, N_target)
            except Exception as exc:
                print(f"FAILED: domain={domain_type}, p={p}, N={N_target}")
                print(f"  {exc}")
                row = failure_row(domain_type, domain.area, p, N_target, exc)

            rows.append(row)

    add_observed_rates(rows)
    summary = build_summary(rows, domain_type)

    raw_csv = write_csv(
        OUTPUT_DIR / f"phase16_pathA_burgers_{domain_type}_raw.csv",
        rows,
        RAW_FIELDS,
    )
    summary_csv = write_csv(
        OUTPUT_DIR / f"phase16_pathA_burgers_{domain_type}_summary.csv",
        summary,
        SUMMARY_FIELDS,
    )

    path_N = plot_error_vs_N(rows, domain_type)
    path_h = plot_error_vs_h(rows, domain_type)
    path_orders = plot_observed_orders(summary, domain_type)
    path_conditioning = plot_conditioning(summary, domain_type)
    field_plot, field_rel_error = plot_field_error(domain_type, domain)

    print("\n" + "=" * 94)
    print("PHASE 16 PATH A ENTROPY-CONSERVATIVE NONLINEAR BURGERS COMPLETE")
    print("=" * 94)
    print(f"geometry plot     -> {geometry_plot}")
    print(f"raw table         -> {raw_csv}")
    print(f"summary table     -> {summary_csv}")
    print(f"plot vs N         -> {path_N}")
    print(f"plot vs h         -> {path_h}")
    print(f"order plot        -> {path_orders}")
    print(f"conditioning plot -> {path_conditioning}")
    print(f"field/error plot  -> {field_plot}")
    print("-" * 94)
    print(
        "p | levels | finest N | flux err | order | last | max SBP | "
        "mass bal | entropy bal | mass cond | max FD cond"
    )
    print("-" * 94)

    for row in summary:
        print(
            f"{row['p']} | "
            f"{row['num_successful_refinements']} | "
            f"{row['N_finest']} | "
            f"{safe_float(row['error_finest']):.3e} | "
            f"{safe_float(row['fitted_order']):.3f} | "
            f"{safe_float(row['last_step_rate']):.3f} | "
            f"{safe_float(row['max_SBP_x']):.3e} | "
            f"{safe_float(row['max_mass_balance_abs']):.3e} | "
            f"{safe_float(row['max_entropy_balance_abs']):.3e} | "
            f"{safe_float(row['max_mass_condition']):.3e} | "
            f"{safe_float(row['max_fd_cond']):.3e}"
        )

    print("-" * 94)
    print(
        f"field/error case: p={FIELD_P}, N={FIELD_N_TARGET}, "
        f"relative flux error={field_rel_error:.3e}"
    )
    print("=" * 94)


if __name__ == "__main__":
    main()
