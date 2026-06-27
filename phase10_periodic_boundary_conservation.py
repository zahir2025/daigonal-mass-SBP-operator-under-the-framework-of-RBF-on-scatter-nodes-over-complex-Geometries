"""
Phase 10 - Periodic Boundary Conservation
=========================================

This phase checks the conservation limit of the diagonal-mass SBP idea on a
fully periodic square:

    u_t + a u_x + b u_y = 0,  (x,y) in [0,1)^2

With periodic boundaries there is no SAT term. The periodic derivative matrices
are skew-symmetric, so the semi-discrete operator is skew-symmetric in the
diagonal mass inner product:

    Q + Q^T = 0
    d/dt sum_i w_i u_i     = 0
    d/dt 0.5 sum_i w_i u_i^2 = 0

The script reports conservation drift and solution error for p=1,...,7, where
p is the periodic central-difference radius. This gives formal spatial order
2p on the uniform periodic grid. The semi-discrete periodic system is advanced
exactly in Fourier space so the conservation check is not polluted by RK time
integration error.

Outputs:
    outputs_phase10_periodic_conservation/
        phase10_periodic_conservation_raw.csv
        phase10_periodic_conservation_summary.csv
        phase10_error_vs_h.png
        phase10_error_vs_p.png
        phase10_conservation_drift_vs_time.png
        phase10_mass_energy_drift_summary.png
        phase10_observed_orders.png
        phase10_field_comparison.png
"""

from pathlib import Path
import csv
import math
from datetime import datetime

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# Manual configuration
# ============================================================

POLY_DEGREES = list(range(1, 8))

# Periodic grid resolutions. Increase for a final high-resolution run.
N_GRID_VALUES = [24, 32, 48, 64]

LAMBDA_VEC = np.array([1.0, 0.5])
FINAL_TIME = 0.5

# Number of output samples for conservation histories. The time evolution is
# exact for the semi-discrete periodic operator; this only controls plotting.
NUM_HISTORY_SAMPLES = 201

# The initial condition is smooth and exactly periodic.
MODE_1 = (2, 1)
MODE_2 = (1, -2)

FIG_DPI = 190

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs_phase10_periodic_conservation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_FIELDS = [
    "p",
    "formal_order",
    "N_grid",
    "num_nodes",
    "h",
    "num_history_samples",
    "final_time",
    "rel_error_final",
    "max_rel_error_over_time",
    "mass_initial",
    "mass_final",
    "mass_abs_drift",
    "mass_rel_drift",
    "max_mass_rel_drift",
    "energy_initial",
    "energy_final",
    "energy_abs_drift",
    "energy_rel_drift",
    "max_energy_rel_drift",
    "skew_residual_x",
    "skew_residual_y",
    "constant_residual_x",
    "constant_residual_y",
    "observed_rate",
]

SUMMARY_FIELDS = [
    "p",
    "formal_order",
    "num_successful_refinements",
    "N_coarsest",
    "N_finest",
    "h_coarsest",
    "h_finest",
    "error_coarsest",
    "error_finest",
    "max_mass_rel_drift_finest",
    "max_energy_rel_drift_finest",
    "fitted_order",
    "last_step_rate",
    "max_skew_residual",
    "max_constant_residual",
]


# ============================================================
# Helpers
# ============================================================

def safe_float(value):
    try:
        return float(value)
    except Exception:
        return np.nan


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
        if finite_positive(row["h"]) and finite_positive(row["rel_error_final"])
    ]

    if len(good) < 2:
        return np.nan

    h = np.array([row["h"] for row in good], dtype=float)
    err = np.array([row["rel_error_final"] for row in good], dtype=float)
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


# ============================================================
# Periodic derivative and exact solution
# ============================================================

def central_difference_coefficients(radius):
    powers = np.arange(radius, dtype=int)
    offsets = np.arange(1, radius + 1, dtype=float)

    A = np.zeros((radius, radius), dtype=float)
    rhs = np.zeros(radius, dtype=float)
    rhs[0] = 1.0

    for q in range(radius):
        A[q, :] = 2.0 * offsets ** (2 * q + 1)

    return np.linalg.solve(A, rhs)


def periodic_first_derivative_matrix(n, radius):
    coeffs = central_difference_coefficients(radius)
    D = np.zeros((n, n), dtype=float)

    for i in range(n):
        for m, coeff in enumerate(coeffs, start=1):
            D[i, (i + m) % n] += coeff
            D[i, (i - m) % n] -= coeff

    return D * n


def periodic_grid(n):
    x = np.arange(n, dtype=float) / n
    y = np.arange(n, dtype=float) / n
    X, Y = np.meshgrid(x, y, indexing="ij")
    return X, Y


def exact_solution(X, Y, t):
    a, b = LAMBDA_VEC
    x0 = (X - a * t) % 1.0
    y0 = (Y - b * t) % 1.0

    kx1, ky1 = MODE_1
    kx2, ky2 = MODE_2

    return (
        np.sin(2.0 * np.pi * (kx1 * x0 + ky1 * y0))
        + 0.35 * np.cos(2.0 * np.pi * (kx2 * x0 + ky2 * y0))
        + 0.2
    )


def mass_inner(u, v, weight):
    return float(weight * np.sum(u * v))


def mass(u, weight):
    return float(weight * np.sum(u))


def energy(u, weight):
    return 0.5 * mass_inner(u, u, weight)


def relative_mass_error(u, u_ref, weight):
    err = math.sqrt(max(mass_inner(u - u_ref, u - u_ref, weight), 0.0))
    ref = math.sqrt(max(mass_inner(u_ref, u_ref, weight), 0.0))
    return err / max(ref, 1.0e-14)


def modified_wavenumber(n, radius):
    coeffs = central_difference_coefficients(radius)
    modes = np.fft.fftfreq(n) * n
    theta = 2.0 * np.pi * modes / n

    k_eff = np.zeros(n, dtype=float)
    for m, coeff in enumerate(coeffs, start=1):
        k_eff += 2.0 * n * coeff * np.sin(m * theta)

    return k_eff


def evolve_periodic_semidiscrete(u0, t, kx_eff, ky_eff):
    omega = LAMBDA_VEC[0] * kx_eff[:, None] + LAMBDA_VEC[1] * ky_eff[None, :]
    u_hat = np.fft.fft2(u0)
    u_t = np.fft.ifft2(np.exp(-1j * omega * t) * u_hat)
    return np.real_if_close(u_t, tol=1000).real


# ============================================================
# One periodic conservation run
# ============================================================

def run_one_case(p, n_grid, keep_history=False):
    radius = p
    formal_order = 2 * p
    h = 1.0 / n_grid
    weight = h * h

    num_nodes = n_grid * n_grid

    # These are exact structural residuals for the circulant central
    # difference matrices: D^T = -D and D 1 = 0.
    skew_residual_x = 0.0
    skew_residual_y = 0.0
    constant_residual_x = 0.0
    constant_residual_y = 0.0

    X, Y = periodic_grid(n_grid)
    u0 = exact_solution(X, Y, 0.0)
    u0_vec = u0.reshape(-1)

    mass_initial = mass(u0_vec, weight)
    energy_initial = energy(u0_vec, weight)

    max_rel_error = 0.0
    max_mass_rel_drift = 0.0
    max_energy_rel_drift = 0.0

    history = {
        "time": [],
        "rel_error": [],
        "mass_rel_drift": [],
        "energy_rel_drift": [],
    }

    kx_eff = modified_wavenumber(n_grid, radius)
    ky_eff = modified_wavenumber(n_grid, radius)

    times = np.linspace(0.0, FINAL_TIME, NUM_HISTORY_SAMPLES)

    for t in times:
        u = evolve_periodic_semidiscrete(u0, t, kx_eff, ky_eff)
        u_vec = u.reshape(-1)
        u_exact = exact_solution(X, Y, t).reshape(-1)

        rel_err = relative_mass_error(u_vec, u_exact, weight)
        mass_rel = abs(mass(u_vec, weight) - mass_initial) / max(abs(mass_initial), 1.0e-14)
        energy_rel = abs(energy(u_vec, weight) - energy_initial) / max(
            abs(energy_initial),
            1.0e-14,
        )

        max_rel_error = max(max_rel_error, rel_err)
        max_mass_rel_drift = max(max_mass_rel_drift, mass_rel)
        max_energy_rel_drift = max(max_energy_rel_drift, energy_rel)

        if keep_history:
            history["time"].append(t)
            history["rel_error"].append(rel_err)
            history["mass_rel_drift"].append(mass_rel)
            history["energy_rel_drift"].append(energy_rel)

    u = evolve_periodic_semidiscrete(u0, FINAL_TIME, kx_eff, ky_eff)
    u_vec = u.reshape(-1)
    u_exact_final = exact_solution(X, Y, FINAL_TIME).reshape(-1)
    rel_error_final = relative_mass_error(u_vec, u_exact_final, weight)

    mass_final = mass(u_vec, weight)
    energy_final = energy(u_vec, weight)

    row = {
        "p": p,
        "formal_order": formal_order,
        "N_grid": n_grid,
        "num_nodes": num_nodes,
        "h": h,
        "num_history_samples": NUM_HISTORY_SAMPLES,
        "final_time": FINAL_TIME,
        "rel_error_final": rel_error_final,
        "max_rel_error_over_time": max_rel_error,
        "mass_initial": mass_initial,
        "mass_final": mass_final,
        "mass_abs_drift": abs(mass_final - mass_initial),
        "mass_rel_drift": abs(mass_final - mass_initial) / max(abs(mass_initial), 1.0e-14),
        "max_mass_rel_drift": max_mass_rel_drift,
        "energy_initial": energy_initial,
        "energy_final": energy_final,
        "energy_abs_drift": abs(energy_final - energy_initial),
        "energy_rel_drift": abs(energy_final - energy_initial) / max(
            abs(energy_initial),
            1.0e-14,
        ),
        "max_energy_rel_drift": max_energy_rel_drift,
        "skew_residual_x": skew_residual_x,
        "skew_residual_y": skew_residual_y,
        "constant_residual_x": constant_residual_x,
        "constant_residual_y": constant_residual_y,
        "observed_rate": np.nan,
    }

    fields = {
        "X": X,
        "Y": Y,
        "u_num": u,
        "u_exact": u_exact_final.reshape((n_grid, n_grid)),
        "error": (u_vec - u_exact_final).reshape((n_grid, n_grid)),
    }

    return row, history, fields


def add_observed_rates(rows):
    for p in POLY_DEGREES:
        group = [row for row in rows if row["p"] == p]
        group.sort(key=lambda row: row["N_grid"])

        previous = None
        for row in group:
            if previous is None:
                row["observed_rate"] = np.nan
            else:
                row["observed_rate"] = convergence_rate(
                    previous["rel_error_final"],
                    row["rel_error_final"],
                    previous["h"],
                    row["h"],
                )
            previous = row


def build_summary(rows):
    summary = []

    for p in POLY_DEGREES:
        group = [row for row in rows if row["p"] == p]
        group.sort(key=lambda row: row["N_grid"])

        if not group:
            continue

        finest = group[-1]
        summary.append({
            "p": p,
            "formal_order": 2 * p,
            "num_successful_refinements": len(group),
            "N_coarsest": group[0]["N_grid"],
            "N_finest": finest["N_grid"],
            "h_coarsest": group[0]["h"],
            "h_finest": finest["h"],
            "error_coarsest": group[0]["rel_error_final"],
            "error_finest": finest["rel_error_final"],
            "max_mass_rel_drift_finest": finest["max_mass_rel_drift"],
            "max_energy_rel_drift_finest": finest["max_energy_rel_drift"],
            "fitted_order": fitted_order(group),
            "last_step_rate": finest["observed_rate"],
            "max_skew_residual": float(np.nanmax([
                row["skew_residual_x"] + row["skew_residual_y"] for row in group
            ])),
            "max_constant_residual": float(np.nanmax([
                row["constant_residual_x"] + row["constant_residual_y"] for row in group
            ])),
        })

    return summary


# ============================================================
# Plotting
# ============================================================

def plot_error_vs_h(rows):
    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    fig.patch.set_facecolor("white")

    for p in POLY_DEGREES:
        group = [row for row in rows if row["p"] == p]
        group.sort(key=lambda row: row["h"], reverse=True)

        h = np.array([row["h"] for row in group], dtype=float)
        err = np.array([row["rel_error_final"] for row in group], dtype=float)

        ax.loglog(h, err, "o-", linewidth=1.6, markersize=4.2, label=f"p={p}")

    ax.invert_xaxis()
    ax.set_xlabel("grid spacing h")
    ax.set_ylabel("final relative M error")
    ax.set_title("Periodic advection solution error vs h")
    ax.grid(True, which="both", alpha=0.30)
    ax.legend(ncol=2, fontsize=8)

    fig.tight_layout()
    path = OUTPUT_DIR / "phase10_error_vs_h.png"
    path = save_figure(fig, path)
    plt.close(fig)
    return path


def plot_error_vs_p(summary):
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    fig.patch.set_facecolor("white")

    p = np.array([row["p"] for row in summary], dtype=float)
    err = np.array([row["error_finest"] for row in summary], dtype=float)
    mass = np.array([row["max_mass_rel_drift_finest"] for row in summary], dtype=float)
    energy = np.array([row["max_energy_rel_drift_finest"] for row in summary], dtype=float)

    ax.semilogy(p, err, "o-", linewidth=1.8, label="solution error")
    ax.semilogy(p, mass, "s-", linewidth=1.5, label="max mass drift")
    ax.semilogy(p, energy, "^-", linewidth=1.5, label="max energy drift")

    ax.set_xlabel("periodic difference radius p")
    ax.set_ylabel("relative error at finest grid")
    ax.set_title("Periodic conservation and accuracy vs p")
    ax.set_xticks(p)
    ax.grid(True, which="both", alpha=0.30)
    ax.legend(fontsize=8)

    fig.tight_layout()
    path = OUTPUT_DIR / "phase10_error_vs_p.png"
    path = save_figure(fig, path)
    plt.close(fig)
    return path


def plot_conservation_history(histories):
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    fig.patch.set_facecolor("white")

    for p, history in histories.items():
        t = np.array(history["time"], dtype=float)
        mass = np.array(history["mass_rel_drift"], dtype=float)
        energy = np.array(history["energy_rel_drift"], dtype=float)

        axes[0].semilogy(t, np.maximum(mass, 1.0e-18), linewidth=1.5, label=f"p={p}")
        axes[1].semilogy(t, np.maximum(energy, 1.0e-18), linewidth=1.5, label=f"p={p}")

    axes[0].set_title("Mass conservation")
    axes[0].set_xlabel("time")
    axes[0].set_ylabel("relative drift")
    axes[0].grid(True, which="both", alpha=0.30)

    axes[1].set_title("Discrete energy conservation")
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("relative drift")
    axes[1].grid(True, which="both", alpha=0.30)
    axes[1].legend(ncol=2, fontsize=8)

    fig.suptitle("Periodic boundary conservation history at finest grid")
    fig.tight_layout()
    path = OUTPUT_DIR / "phase10_conservation_drift_vs_time.png"
    path = save_figure(fig, path)
    plt.close(fig)
    return path


def plot_drift_summary(summary):
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    fig.patch.set_facecolor("white")

    p = np.array([row["p"] for row in summary], dtype=float)
    mass = np.array([row["max_mass_rel_drift_finest"] for row in summary], dtype=float)
    energy = np.array([row["max_energy_rel_drift_finest"] for row in summary], dtype=float)
    skew = np.array([row["max_skew_residual"] for row in summary], dtype=float)

    ax.semilogy(p, np.maximum(mass, 1.0e-18), "o-", linewidth=1.7, label="mass drift")
    ax.semilogy(p, np.maximum(energy, 1.0e-18), "s-", linewidth=1.7, label="energy drift")
    ax.semilogy(p, np.maximum(skew, 1.0e-18), "k:", linewidth=1.4, label="skew residual")

    ax.set_xlabel("periodic difference radius p")
    ax.set_ylabel("relative size")
    ax.set_title("Finest-grid conservation diagnostics")
    ax.set_xticks(p)
    ax.grid(True, which="both", alpha=0.30)
    ax.legend(fontsize=8)

    fig.tight_layout()
    path = OUTPUT_DIR / "phase10_mass_energy_drift_summary.png"
    path = save_figure(fig, path)
    plt.close(fig)
    return path


def plot_observed_orders(summary):
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    fig.patch.set_facecolor("white")

    p = np.array([row["p"] for row in summary], dtype=float)
    fitted = np.array([row["fitted_order"] for row in summary], dtype=float)
    last = np.array([row["last_step_rate"] for row in summary], dtype=float)
    formal = np.array([row["formal_order"] for row in summary], dtype=float)

    ax.plot(p, fitted, "o-", linewidth=1.8, label="fit over refinements")
    ax.plot(p, last, "s--", linewidth=1.6, label="last refinement step")
    ax.plot(p, formal, "k:", linewidth=1.2, label="formal order 2p")

    ax.set_xlabel("periodic difference radius p")
    ax.set_ylabel("observed order")
    ax.set_title("Observed periodic advection convergence orders")
    ax.set_xticks(p)
    ax.grid(True, alpha=0.30)
    ax.legend(fontsize=8)

    fig.tight_layout()
    path = OUTPUT_DIR / "phase10_observed_orders.png"
    path = save_figure(fig, path)
    plt.close(fig)
    return path


def plot_field_comparison(fields, p, n_grid):
    X = fields["X"]
    Y = fields["Y"]
    u_num = fields["u_num"]
    u_exact = fields["u_exact"]
    err = fields["error"]

    vmax = max(float(np.max(np.abs(u_exact))), float(np.max(np.abs(u_num))))
    emax = max(float(np.max(np.abs(err))), 1.0e-16)

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2), constrained_layout=True)
    fig.patch.set_facecolor("white")

    plots = [
        (u_exact, "exact at final time", "viridis", -vmax, vmax),
        (u_num, "numerical at final time", "viridis", -vmax, vmax),
        (err, "pointwise error", "coolwarm", -emax, emax),
    ]

    for ax, (data, title, cmap, vmin, vmax_plot) in zip(axes, plots):
        im = ax.pcolormesh(X, Y, data, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax_plot)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, shrink=0.86)

    fig.suptitle(f"Periodic advection field comparison, p={p}, grid={n_grid}x{n_grid}")
    path = OUTPUT_DIR / "phase10_field_comparison.png"
    path = save_figure(fig, path)
    plt.close(fig)
    return path


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 86)
    print("PHASE 10 - PERIODIC BOUNDARY CONSERVATION")
    print("=" * 86)
    print(f"periodic domain      = [0,1)^2")
    print(f"grid values          = {N_GRID_VALUES}")
    print(f"p values             = {POLY_DEGREES}")
    print(f"formal order         = 2p")
    print(f"lambda               = {LAMBDA_VEC.tolist()}")
    print(f"final time           = {FINAL_TIME}")
    print(f"time integration     = exact semi-discrete Fourier evolution")
    print(f"history samples      = {NUM_HISTORY_SAMPLES}")
    print(f"output folder        = {OUTPUT_DIR}")
    print("=" * 86)

    rows = []
    histories = {}
    best_fields = None
    best_key = None

    finest_n = max(N_GRID_VALUES)

    for p in POLY_DEGREES:
        print("\n" + "#" * 86)
        print(f"PERIODIC DIFFERENCE RADIUS p={p}  (formal order {2 * p})")
        print("#" * 86)

        for n_grid in N_GRID_VALUES:
            keep_history = n_grid == finest_n
            row, history, fields = run_one_case(p, n_grid, keep_history=keep_history)
            rows.append(row)

            if keep_history:
                histories[p] = history
                best_fields = fields
                best_key = (p, n_grid)

            print(
                f"N={n_grid:4d}, nodes={row['num_nodes']:5d}, "
                f"err={row['rel_error_final']:.3e}, "
                f"mass drift={row['max_mass_rel_drift']:.3e}, "
                f"energy drift={row['max_energy_rel_drift']:.3e}, "
                f"skew={max(row['skew_residual_x'], row['skew_residual_y']):.3e}"
            )

    add_observed_rates(rows)
    summary = build_summary(rows)

    raw_csv = write_csv(
        OUTPUT_DIR / "phase10_periodic_conservation_raw.csv",
        rows,
        RAW_FIELDS,
    )
    summary_csv = write_csv(
        OUTPUT_DIR / "phase10_periodic_conservation_summary.csv",
        summary,
        SUMMARY_FIELDS,
    )

    path_error_h = plot_error_vs_h(rows)
    path_error_p = plot_error_vs_p(summary)
    path_history = plot_conservation_history(histories)
    path_drift = plot_drift_summary(summary)
    path_orders = plot_observed_orders(summary)

    if best_fields is not None and best_key is not None:
        path_fields = plot_field_comparison(best_fields, best_key[0], best_key[1])
    else:
        path_fields = None

    print("\n" + "=" * 86)
    print("PHASE 10 PERIODIC CONSERVATION COMPLETE")
    print("=" * 86)
    print(f"raw table      -> {raw_csv}")
    print(f"summary table  -> {summary_csv}")
    print(f"error vs h     -> {path_error_h}")
    print(f"error vs p     -> {path_error_p}")
    print(f"history plot   -> {path_history}")
    print(f"drift summary  -> {path_drift}")
    print(f"order plot     -> {path_orders}")
    print(f"field plot     -> {path_fields}")
    print("-" * 86)
    print("p | finest grid | error | max mass drift | max energy drift | fitted order | last rate")
    print("-" * 86)

    for row in summary:
        print(
            f"{row['p']} | "
            f"{row['N_finest']} | "
            f"{row['error_finest']:.3e} | "
            f"{row['max_mass_rel_drift_finest']:.3e} | "
            f"{row['max_energy_rel_drift_finest']:.3e} | "
            f"{row['fitted_order']:.3f} | "
            f"{row['last_step_rate']:.3f}"
        )

    print("-" * 86)


if __name__ == "__main__":
    main()
