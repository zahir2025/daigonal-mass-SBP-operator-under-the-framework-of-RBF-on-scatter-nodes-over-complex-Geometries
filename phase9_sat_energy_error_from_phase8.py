"""
Phase 9 - SAT/Discrete Energy Error From Phase 8
================================================

This script reads the Phase 8 raw CSV and extracts the energy error that
isolates the SAT numerical solution from the diagonal-mass quadrature error:

    E_exact_ML = 0.5 * sum_i w_i u_exact(x_i)^2
    E_num_ML   = 0.5 * sum_i w_i u_h(x_i)^2

    SAT/discrete energy error = |E_num_ML - E_exact_ML| / |E_exact_ML|

This is the diagnostic that should approach roundoff as the steady compatible
SAT solution becomes exact in the lumped-mass norm. It is different from the
Phase 8 continuous-reference energy error

    |E_num_ML - E_exact_ref| / |E_exact_ref|,

which also contains the Voronoi lumped-mass quadrature error.

Outputs:
    outputs_phase9_sat_energy_error/
        phase9_sat_energy_error_<domain>_raw.csv
        phase9_sat_energy_error_<domain>_finest.csv
        phase9_sat_energy_error_<domain>_vs_p.png
        phase9_sat_energy_error_<domain>_vs_N.png
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

# Change this to:
#     "annulus"
#     "box_minus_circle"
#     "box_minus_airfoil"
DOMAIN_TYPE = "annulus"

# Use None to process every resolution available in the Phase 8 raw CSV.
# Use a list such as [400], [800, 1600], or [3200] to process selected
# resolutions only.
N_VALUES = None

# If True, print every selected (p, N) row. If False, print only the finest N
# for each p.
PRINT_ALL_RESOLUTIONS = True

POLY_DEGREES = list(range(1, 8))

SCRIPT_DIR = Path(__file__).resolve().parent
PHASE8_DIR = SCRIPT_DIR / "outputs_phase8_steady_energy"

OUTPUT_DIR = SCRIPT_DIR / "outputs_phase9_sat_energy_error"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FIG_DPI = 180

RAW_FIELDS = [
    "domain",
    "p",
    "N",
    "h",
    "theta",
    "E_exact_ref",
    "E_exact_ML",
    "E_num_ML",
    "sat_energy_abs_error",
    "sat_energy_rel_error",
    "continuous_energy_rel_error",
    "quadrature_energy_rel_error",
    "solution_rel_error",
    "linear_residual",
    "observed_sat_energy_rate",
    "SBP_x",
    "SBP_y",
    "poly_x",
    "poly_y",
    "max_fd_cond",
]

FINEST_FIELDS = [
    "domain",
    "p",
    "theta",
    "finest_N",
    "finest_h",
    "E_exact_ML",
    "E_num_ML",
    "sat_energy_rel_error",
    "solution_rel_error",
    "continuous_energy_rel_error",
    "quadrature_energy_rel_error",
    "last_step_sat_energy_rate",
    "fitted_sat_energy_order",
    "max_SBP_x",
    "max_poly_x",
    "max_fd_cond",
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


def fitted_order(rows, error_key):
    good = [
        row for row in rows
        if finite_positive(row["h"]) and finite_positive(row[error_key])
    ]

    if len(good) < 2:
        return np.nan

    h = np.array([row["h"] for row in good], dtype=float)
    err = np.array([row[error_key] for row in good], dtype=float)
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


def read_phase8_rows(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Phase 8 raw CSV not found:\n{path}\n"
            "Run phase8_steady_energy_compatible_SAT_manual_domain.py first."
        )

    with open(path, mode="r", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def phase8_raw_path(domain_type):
    return PHASE8_DIR / f"phase8_energy_{domain_type}_raw.csv"


def build_phase9_rows(phase8_rows, domain_type, n_values):
    rows = []
    n_filter = None if n_values is None else {int(n) for n in n_values}

    for src in phase8_rows:
        if src.get("status") != "ok":
            continue

        N = int(safe_float(src.get("N")))
        if n_filter is not None and N not in n_filter:
            continue

        E_exact_ML = safe_float(src.get("E_exact_ML"))
        E_num_ML = safe_float(src.get("E_num_ML"))
        sat_abs = abs(E_num_ML - E_exact_ML)
        sat_rel = sat_abs / max(abs(E_exact_ML), 1.0e-14)

        rows.append({
            "domain": src.get("domain", domain_type),
            "p": int(safe_float(src.get("p"))),
            "N": N,
            "h": safe_float(src.get("h")),
            "theta": safe_float(src.get("theta")),
            "E_exact_ref": safe_float(src.get("E_exact_ref")),
            "E_exact_ML": E_exact_ML,
            "E_num_ML": E_num_ML,
            "sat_energy_abs_error": sat_abs,
            "sat_energy_rel_error": sat_rel,
            "continuous_energy_rel_error": safe_float(src.get("energy_num_rel_error")),
            "quadrature_energy_rel_error": safe_float(src.get("energy_quad_rel_error")),
            "solution_rel_error": safe_float(src.get("solution_rel_error")),
            "linear_residual": safe_float(src.get("linear_residual")),
            "observed_sat_energy_rate": np.nan,
            "SBP_x": safe_float(src.get("SBP_x")),
            "SBP_y": safe_float(src.get("SBP_y")),
            "poly_x": safe_float(src.get("poly_x")),
            "poly_y": safe_float(src.get("poly_y")),
            "max_fd_cond": safe_float(src.get("max_fd_cond")),
        })

    for p in POLY_DEGREES:
        group = [row for row in rows if row["p"] == p]
        group.sort(key=lambda row: row["N"])

        previous = None
        for row in group:
            if previous is None:
                row["observed_sat_energy_rate"] = np.nan
            else:
                row["observed_sat_energy_rate"] = convergence_rate(
                    previous["sat_energy_rel_error"],
                    row["sat_energy_rel_error"],
                    previous["h"],
                    row["h"],
                )
            previous = row

    return rows


def build_finest_summary(rows):
    summary = []

    for p in POLY_DEGREES:
        group = [row for row in rows if row["p"] == p]
        group.sort(key=lambda row: row["N"])

        if not group:
            continue

        finest = group[-1]
        summary.append({
            "domain": finest["domain"],
            "p": p,
            "theta": finest["theta"],
            "finest_N": finest["N"],
            "finest_h": finest["h"],
            "E_exact_ML": finest["E_exact_ML"],
            "E_num_ML": finest["E_num_ML"],
            "sat_energy_rel_error": finest["sat_energy_rel_error"],
            "solution_rel_error": finest["solution_rel_error"],
            "continuous_energy_rel_error": finest["continuous_energy_rel_error"],
            "quadrature_energy_rel_error": finest["quadrature_energy_rel_error"],
            "last_step_sat_energy_rate": finest["observed_sat_energy_rate"],
            "fitted_sat_energy_order": fitted_order(group, "sat_energy_rel_error"),
            "max_SBP_x": float(np.nanmax([row["SBP_x"] for row in group])),
            "max_poly_x": float(np.nanmax([row["poly_x"] for row in group])),
            "max_fd_cond": float(np.nanmax([row["max_fd_cond"] for row in group])),
        })

    return summary


# ============================================================
# Plotting
# ============================================================

def plot_finest_vs_p(summary):
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    fig.patch.set_facecolor("white")

    p = np.array([row["p"] for row in summary], dtype=float)
    sat = np.array([row["sat_energy_rel_error"] for row in summary], dtype=float)
    sol = np.array([row["solution_rel_error"] for row in summary], dtype=float)
    cont = np.array([row["continuous_energy_rel_error"] for row in summary], dtype=float)
    quad = np.array([row["quadrature_energy_rel_error"] for row in summary], dtype=float)

    ax.semilogy(p, sat, "o-", linewidth=1.8, label="SAT/discrete energy")
    ax.semilogy(p, sol, "s-", linewidth=1.5, label="solution")
    ax.semilogy(p, cont, "^-", linewidth=1.2, label="continuous energy")
    ax.semilogy(p, quad, "k:", linewidth=1.3, label="quadrature floor")

    ax.set_xlabel("polynomial degree p")
    ax.set_ylabel("relative error at finest N")
    domain_type = summary[0]["domain"] if summary else DOMAIN_TYPE
    ax.set_title(f"Phase 9 SAT energy error at finest mesh: {domain_type}")
    ax.set_xticks(p)
    ax.grid(True, which="both", alpha=0.30)
    ax.legend(fontsize=8)

    fig.tight_layout()
    path = OUTPUT_DIR / f"phase9_sat_energy_error_{domain_type}_vs_p.png"
    path = save_figure(fig, path)
    plt.close(fig)
    return path


def plot_vs_N(rows, domain_type):
    fig, ax = plt.subplots(figsize=(8.0, 5.4))
    fig.patch.set_facecolor("white")

    for p in POLY_DEGREES:
        group = [row for row in rows if row["p"] == p]
        group.sort(key=lambda row: row["N"])

        if not group:
            continue

        N = np.array([row["N"] for row in group], dtype=float)
        sat = np.array([row["sat_energy_rel_error"] for row in group], dtype=float)
        ax.loglog(N, sat, "o-", linewidth=1.5, markersize=4.0, label=f"p={p}")

    ax.set_xlabel("number of nodes N")
    ax.set_ylabel("SAT/discrete relative energy error")
    ax.set_title(f"Phase 9 SAT energy error vs N: {domain_type}")
    ax.grid(True, which="both", alpha=0.30)
    ax.legend(ncol=2, fontsize=8)

    fig.tight_layout()
    path = OUTPUT_DIR / f"phase9_sat_energy_error_{domain_type}_vs_N.png"
    path = save_figure(fig, path)
    plt.close(fig)
    return path


# ============================================================
# Main
# ============================================================

def main():
    domain_type = DOMAIN_TYPE
    phase8_raw = phase8_raw_path(domain_type)

    phase8_rows = read_phase8_rows(phase8_raw)
    rows = build_phase9_rows(phase8_rows, domain_type, N_VALUES)

    if not rows:
        available = sorted({
            int(safe_float(row.get("N")))
            for row in phase8_rows
            if row.get("status") == "ok" and np.isfinite(safe_float(row.get("N")))
        })
        raise RuntimeError(
            f"No matching Phase 8 rows for DOMAIN_TYPE={domain_type!r} "
            f"and N_VALUES={N_VALUES!r}.\n"
            f"Available N values in {phase8_raw.name}: {available}"
        )

    summary = build_finest_summary(rows)

    raw_csv = write_csv(
        OUTPUT_DIR / f"phase9_sat_energy_error_{domain_type}_raw.csv",
        rows,
        RAW_FIELDS,
    )
    finest_csv = write_csv(
        OUTPUT_DIR / f"phase9_sat_energy_error_{domain_type}_finest.csv",
        summary,
        FINEST_FIELDS,
    )

    vs_p_plot = plot_finest_vs_p(summary)
    vs_N_plot = plot_vs_N(rows, domain_type)

    print("=" * 96)
    print("PHASE 9 - SAT/DISCRETE ENERGY ERROR FROM PHASE 8")
    print("=" * 96)
    print(f"domain       = {domain_type}")
    print(f"N filter     = {N_VALUES if N_VALUES is not None else 'all available'}")
    print(f"phase 8 raw  = {phase8_raw}")
    print(f"raw table    = {raw_csv}")
    print(f"finest table = {finest_csv}")
    print(f"plot vs p    = {vs_p_plot}")
    print(f"plot vs N    = {vs_N_plot}")
    print("-" * 96)

    if PRINT_ALL_RESOLUTIONS:
        print("p | N | SAT energy err | solution err | continuous energy err | quadrature floor")
        print("-" * 96)
        for row in sorted(rows, key=lambda item: (item["N"], item["p"])):
            print(
                f"{row['p']} | "
                f"{row['N']} | "
                f"{row['sat_energy_rel_error']:.3e} | "
                f"{row['solution_rel_error']:.3e} | "
                f"{row['continuous_energy_rel_error']:.3e} | "
                f"{row['quadrature_energy_rel_error']:.3e}"
            )
    else:
        print(
            "p | finest selected N | SAT energy err | solution err | "
            "continuous energy err | quadrature floor"
        )
        print("-" * 96)
        for row in summary:
            print(
                f"{row['p']} | "
                f"{row['finest_N']} | "
                f"{row['sat_energy_rel_error']:.3e} | "
                f"{row['solution_rel_error']:.3e} | "
                f"{row['continuous_energy_rel_error']:.3e} | "
                f"{row['quadrature_energy_rel_error']:.3e}"
            )

    print("-" * 96)


if __name__ == "__main__":
    main()
