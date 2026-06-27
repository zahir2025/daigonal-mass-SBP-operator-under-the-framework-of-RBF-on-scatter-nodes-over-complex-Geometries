"""
Phase 11 - Periodic Advection With and Without Dissipation
==========================================================

This phase uses the periodic conservation setting from Phase 10, fixes p=7,
and compares two semi-discrete periodic evolutions up to

    T = 2*pi.

Cases:
    1. no dissipation:
           u_t + a D_x u + b D_y u = 0

    2. added high-frequency dissipation:
           u_t + a D_x u + b D_y u = -epsilon * K u

The evolution is computed exactly in Fourier space for both cases, so the
energy difference comes from the artificial dissipation, not time stepping.

Outputs:
    outputs_phase11_periodic_dissipation/
        phase11_field_comparison_p7_T_2pi.png
        phase11_energy_with_without_dissipation_p7_T_2pi.png
        phase11_dissipation_summary.csv
"""

from pathlib import Path
import csv
import math
from datetime import datetime

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


# ============================================================
# Manual configuration
# ============================================================

P_DEGREE = 7
N_GRID = 256
FINAL_TIME = 2.0 * math.pi

LAMBDA_VEC = np.array([1.0, 0.5])

# High-frequency dissipation strength. Increase for stronger smoothing.
DISSIPATION_EPSILON = 35.0

# Dissipation order in the modal filter. The low-order filter damps the
# localized fine-scale packet clearly while preserving the broad waves.
DISSIPATION_POWER = 1

NUM_HISTORY_SAMPLES = 500
FIG_DPI = 256

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs_phase11_periodic_dissipation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Helpers
# ============================================================

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


def central_difference_coefficients(radius):
    powers = np.arange(radius, dtype=int)
    offsets = np.arange(1, radius + 1, dtype=float)

    A = np.zeros((radius, radius), dtype=float)
    rhs = np.zeros(radius, dtype=float)
    rhs[0] = 1.0

    for q in range(radius):
        A[q, :] = 2.0 * offsets ** (2 * q + 1)

    return np.linalg.solve(A, rhs)


def modified_wavenumber(n, radius):
    coeffs = central_difference_coefficients(radius)
    modes = np.fft.fftfreq(n) * n
    theta = 2.0 * np.pi * modes / n

    k_eff = np.zeros(n, dtype=float)
    for m, coeff in enumerate(coeffs, start=1):
        k_eff += 2.0 * n * coeff * np.sin(m * theta)

    return k_eff


def dissipation_symbol(n, power):
    modes = np.fft.fftfreq(n) * n
    theta = 2.0 * np.pi * modes / n
    return (2.0 * np.sin(0.5 * theta)) ** (2 * power)


def periodic_grid(n):
    x = np.arange(n, dtype=float) / n
    y = np.arange(n, dtype=float) / n
    X, Y = np.meshgrid(x, y, indexing="ij")
    return X, Y


def initial_condition(X, Y):
    smooth = (
        np.sin(2.0 * np.pi * (2.0 * X + 1.0 * Y))
        + 0.50 * np.cos(2.0 * np.pi * (1.0 * X - 3.0 * Y))
        + 0.28 * np.sin(2.0 * np.pi * (3.0 * X - 2.0 * Y))
    )

    dx = (X - 0.34 + 0.5) % 1.0 - 0.5
    dy = (Y - 0.62 + 0.5) % 1.0 - 0.5
    envelope = np.exp(-95.0 * (dx * dx + dy * dy))

    packet = 0.45 * envelope
    texture = envelope * (
        0.55 * np.sin(2.0 * np.pi * (38.0 * X - 27.0 * Y))
        + 0.28 * np.cos(2.0 * np.pi * (30.0 * X + 19.0 * Y))
    )

    return smooth + packet + texture


def energy(u, weight):
    return 0.5 * float(weight * np.sum(u * u))


def mass(u, weight):
    return float(weight * np.sum(u))


def evolve(u0, t, kx_eff, ky_eff, diss_x, diss_y, epsilon):
    omega = LAMBDA_VEC[0] * kx_eff[:, None] + LAMBDA_VEC[1] * ky_eff[None, :]
    damping = epsilon * (diss_x[:, None] + diss_y[None, :])
    u_hat = np.fft.fft2(u0)
    u_t = np.fft.ifft2(np.exp((-1j * omega - damping) * t) * u_hat)
    return np.real_if_close(u_t, tol=1000).real


# ============================================================
# Plotting
# ============================================================

def style_axis(ax):
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])


def plot_field_comparison(X, Y, u0, u_diss, u_clean):
    vmax = max(
        float(np.max(np.abs(u0))),
        float(np.max(np.abs(u_diss))),
        float(np.max(np.abs(u_clean))),
    )
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)

    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.7), constrained_layout=True)
    fig.patch.set_facecolor("white")

    panels = [
        (u0, "Initial solution"),
        (u_diss, "Final with dissipation"),
        (u_clean, "Final without dissipation"),
    ]

    for ax, (data, title) in zip(axes, panels):
        im = ax.imshow(
            data.T,
            origin="lower",
            extent=(0, 1, 0, 1),
            cmap="RdBu_r",
            norm=norm,
            interpolation="nearest",
            resample=False,
            aspect="equal",
        )
        ax.set_title(title, fontsize=12, fontweight="bold")
        style_axis(ax)

    cbar = fig.colorbar(im, ax=axes, shrink=0.86, pad=0.018)
    cbar.set_label("u(x,y)")

    fig.suptitle(
        f"Periodic advection, p={P_DEGREE}, T=2*pi, grid={N_GRID}x{N_GRID}",
        fontsize=14,
        fontweight="bold",
    )

    path = OUTPUT_DIR / "phase11_field_comparison_p7_T_2pi.png"
    path = save_figure(fig, path)
    plt.close(fig)
    return path


def plot_energy_history(times, energy_clean, energy_diss, mass_clean, mass_diss):
    E0 = energy_clean[0]
    M0 = mass_clean[0]

    rel_clean = np.abs(energy_clean - E0) / max(abs(E0), 1.0e-14)
    rel_diss_loss = np.maximum(E0 - energy_diss, 0.0) / max(abs(E0), 1.0e-14)
    mass_clean_rel = np.abs(mass_clean - M0) / max(abs(M0), 1.0e-14)
    mass_diss_rel = np.abs(mass_diss - M0) / max(abs(M0), 1.0e-14)

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), constrained_layout=True)
    fig.patch.set_facecolor("white")

    axes[0].plot(times, energy_clean, color="#2563EB", linewidth=2.0, label="without dissipation")
    axes[0].plot(times, energy_diss, color="#DC2626", linewidth=2.0, label="with dissipation")
    axes[0].set_xlabel("time")
    axes[0].set_ylabel("discrete energy")
    axes[0].set_title("Energy history", fontweight="bold")
    axes[0].grid(True, alpha=0.28)
    axes[0].legend(frameon=False)

    axes[1].semilogy(times, np.maximum(rel_clean, 1.0e-18), color="#2563EB", linewidth=2.0, label="energy drift, no diss.")
    axes[1].semilogy(times, np.maximum(rel_diss_loss, 1.0e-18), color="#DC2626", linewidth=2.0, label="energy loss, diss.")
    axes[1].semilogy(times, np.maximum(mass_clean_rel, 1.0e-18), color="#0F766E", linestyle="--", linewidth=1.5, label="mass, no diss.")
    axes[1].semilogy(times, np.maximum(mass_diss_rel, 1.0e-18), color="#9333EA", linestyle="--", linewidth=1.5, label="mass, diss.")
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("relative drift")
    axes[1].set_title("Energy loss and conservation", fontweight="bold")
    axes[1].grid(True, which="both", alpha=0.28)
    axes[1].legend(frameon=False, fontsize=8)

    fig.suptitle(
        f"Periodic p={P_DEGREE}: energy with and without dissipation",
        fontsize=14,
        fontweight="bold",
    )

    path = OUTPUT_DIR / "phase11_energy_with_without_dissipation_p7_T_2pi.png"
    path = save_figure(fig, path)
    plt.close(fig)
    return path


# ============================================================
# Main
# ============================================================

def main():
    h = 1.0 / N_GRID
    weight = h * h

    X, Y = periodic_grid(N_GRID)
    u0 = initial_condition(X, Y)

    k_eff = modified_wavenumber(N_GRID, P_DEGREE)
    diss = dissipation_symbol(N_GRID, DISSIPATION_POWER)

    # Scale the dissipation so the largest mode decays by exp(-epsilon*T).
    diss = diss / max(float(np.max(diss + diss[:, None])), 1.0e-14)

    times = np.linspace(0.0, FINAL_TIME, NUM_HISTORY_SAMPLES)

    energy_clean = []
    energy_diss = []
    mass_clean = []
    mass_diss = []

    for t in times:
        clean = evolve(u0, t, k_eff, k_eff, diss, diss, 0.0)
        damped = evolve(u0, t, k_eff, k_eff, diss, diss, DISSIPATION_EPSILON)

        energy_clean.append(energy(clean, weight))
        energy_diss.append(energy(damped, weight))
        mass_clean.append(mass(clean, weight))
        mass_diss.append(mass(damped, weight))

    energy_clean = np.array(energy_clean, dtype=float)
    energy_diss = np.array(energy_diss, dtype=float)
    mass_clean = np.array(mass_clean, dtype=float)
    mass_diss = np.array(mass_diss, dtype=float)

    u_clean_final = evolve(u0, FINAL_TIME, k_eff, k_eff, diss, diss, 0.0)
    u_diss_final = evolve(u0, FINAL_TIME, k_eff, k_eff, diss, diss, DISSIPATION_EPSILON)

    field_plot = plot_field_comparison(X, Y, u0, u_diss_final, u_clean_final)
    energy_plot = plot_energy_history(times, energy_clean, energy_diss, mass_clean, mass_diss)

    E0 = energy_clean[0]
    M0 = mass_clean[0]

    rows = [{
        "p": P_DEGREE,
        "N_grid": N_GRID,
        "num_nodes": N_GRID * N_GRID,
        "final_time": FINAL_TIME,
        "lambda_x": LAMBDA_VEC[0],
        "lambda_y": LAMBDA_VEC[1],
        "dissipation_epsilon": DISSIPATION_EPSILON,
        "dissipation_power": DISSIPATION_POWER,
        "energy_initial": E0,
        "energy_final_without_dissipation": energy_clean[-1],
        "energy_final_with_dissipation": energy_diss[-1],
        "energy_rel_drift_without_dissipation": abs(energy_clean[-1] - E0) / max(abs(E0), 1.0e-14),
        "energy_rel_drop_with_dissipation": (E0 - energy_diss[-1]) / max(abs(E0), 1.0e-14),
        "mass_initial": M0,
        "mass_final_without_dissipation": mass_clean[-1],
        "mass_final_with_dissipation": mass_diss[-1],
        "mass_rel_drift_without_dissipation": abs(mass_clean[-1] - M0) / max(abs(M0), 1.0e-14),
        "mass_rel_drift_with_dissipation": abs(mass_diss[-1] - M0) / max(abs(M0), 1.0e-14),
        "field_plot": str(field_plot),
        "energy_plot": str(energy_plot),
    }]

    summary_csv = write_csv(
        OUTPUT_DIR / "phase11_dissipation_summary.csv",
        rows,
        list(rows[0].keys()),
    )

    print("=" * 92)
    print("PHASE 11 - PERIODIC DISSIPATION COMPARISON")
    print("=" * 92)
    print(f"p                         = {P_DEGREE}")
    print(f"grid                      = {N_GRID} x {N_GRID}")
    print(f"final time                = 2*pi = {FINAL_TIME:.12e}")
    print(f"lambda                    = {LAMBDA_VEC.tolist()}")
    print(f"dissipation epsilon       = {DISSIPATION_EPSILON}")
    print(f"dissipation power         = {DISSIPATION_POWER}")
    print(f"initial energy            = {E0:.12e}")
    print(f"final energy no diss.     = {energy_clean[-1]:.12e}")
    print(f"final energy with diss.   = {energy_diss[-1]:.12e}")
    print(
        "energy rel drift no diss. = "
        f"{abs(energy_clean[-1] - E0) / max(abs(E0), 1.0e-14):.3e}"
    )
    print(
        "energy rel drop with diss.= "
        f"{(E0 - energy_diss[-1]) / max(abs(E0), 1.0e-14):.3e}"
    )
    print(f"field plot                = {field_plot}")
    print(f"energy plot               = {energy_plot}")
    print(f"summary CSV               = {summary_csv}")
    print("=" * 92)


if __name__ == "__main__":
    main()
