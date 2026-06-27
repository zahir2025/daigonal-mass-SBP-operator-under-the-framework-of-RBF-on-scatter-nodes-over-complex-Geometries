"""
Phase 3b - Path A SAT Strength Scan
===================================

Purpose:
    Diagnose why Phase 3 energy decays too strongly.

We compare three operators:

1. Closed/skew operator:
       A_closed = Q_lambda - 0.5 E_lambda

   Then:
       A_closed + A_closed.T = 0

   So implicit midpoint should conserve energy to roundoff.
   This checks the interior SBP operator.

2. Full compatible SAT:
       B_abs    = |E_lambda|
       B_minus  = 0.5 (B_abs - E_lambda)
       A_full   = Q_lambda + B_minus

   Then:
       sym(A_full) = 0.5 B_abs >= 0

   This is guaranteed stable but can be too dissipative.

3. Reduced SAT scan:
       B_minus(theta) = theta * B_minus
       A_theta        = Q_lambda + B_minus(theta)

   We scan theta in [0,1]. Stability requires:
       sym(A_theta) >= 0

   This tells us how much SAT can be reduced while keeping energy stability.
"""

from pathlib import Path
import csv
import warnings

import numpy as np
import matplotlib

SHOW_FIGS = True
# matplotlib.use("Agg")

import matplotlib.pyplot as plt
from scipy.linalg import lu_factor, lu_solve
from scipy.spatial import Voronoi, cKDTree
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection, Point, box
from shapely.ops import unary_union

warnings.filterwarnings("ignore")


# ============================================================
# Configuration
# ============================================================

DOMAINS = [
    "annulus",
    "box_minus_circle",
    "box_minus_airfoil",
]

N_TARGETS = {
    "annulus": 300,
    "box_minus_circle": 400,
    "box_minus_airfoil": 900,
}

PHS_ORDER = 5
POLY_DEGREE = 3

STENCIL_FACTOR = 3.5
STENCIL_MINIMUM = 25

LAMBDA_VEC = np.array([1.0, 1.0])

FINAL_TIME = 0.5
CFL = 0.01

THETA_VALUES = np.linspace(0.0, 1.0, 21)

PSD_TOL = 1.0e-12

R_INNER = 0.3
R_OUTER = 1.0

CIRCLE_RESOLUTION = 256
AIRFOIL_POINTS = 900

FIG_DPI = 180

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs_phase3b_pathA_SAT_strength_scan"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_FILE = OUTPUT_DIR / "phase3b_pathA_SAT_strength_scan.csv"


# ============================================================
# Halton nodes
# ============================================================

def halton(n, base):
    seq = np.zeros(n)
    num, den = 0, 1

    for i in range(n):
        x = den - num

        if x == 1:
            num, den = 1, den * base
        else:
            y = den // base
            while x <= y:
                y //= base
            num = (base + 1) * y - x

        seq[i] = num / den

    return seq


def halton_2d(n):
    return np.column_stack([halton(n, 2), halton(n, 3)])


def nodes_on_annulus(N_target, r_in=0.3, r_out=1.0):
    factor = 4.5
    N_cand = int(N_target * factor)

    h = halton_2d(N_cand)
    pts = 2.0 * r_out * h - r_out

    r = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
    pts = pts[(r > r_in) & (r < r_out)]

    if len(pts) >= N_target:
        return pts[:N_target]

    raise RuntimeError(f"Not enough annulus candidates: got {len(pts)}, need {N_target}")


def nodes_in_domain(N_target, domain):
    minx, miny, maxx, maxy = domain.bounds

    box_area = (maxx - minx) * (maxy - miny)
    keep_fraction = max(domain.area / box_area, 1.0e-12)

    N_cand = int(np.ceil(1.8 * N_target / keep_fraction))
    N_cand = max(N_cand, 4 * N_target)

    while True:
        h = halton_2d(N_cand)

        pts = np.column_stack((
            minx + (maxx - minx) * h[:, 0],
            miny + (maxy - miny) * h[:, 1],
        ))

        inside = np.array([
            domain.contains(Point(float(x), float(y)))
            for x, y in pts
        ])

        pts = pts[inside]

        if len(pts) >= N_target:
            return pts[:N_target]

        N_cand *= 2


# ============================================================
# Domains
# ============================================================

def make_circle_polygon(cx, cy, r, n_pts=256):
    theta = np.linspace(0.0, 2.0 * np.pi, n_pts, endpoint=False)

    coords = list(zip(
        cx + r * np.cos(theta),
        cy + r * np.sin(theta),
    ))

    return Polygon(coords).buffer(0)


def make_annulus(r_in=0.3, r_out=1.0, n_pts=256):
    theta = np.linspace(0.0, 2.0 * np.pi, n_pts, endpoint=False)

    outer = list(zip(r_out * np.cos(theta), r_out * np.sin(theta)))
    inner = list(zip(r_in * np.cos(theta[::-1]), r_in * np.sin(theta[::-1])))

    return Polygon(outer, [inner]).buffer(0)


def naca0012_thickness(x, c=1.0, t=0.12):
    xc = np.clip(x / c, 0.0, 1.0)

    return 5.0 * t * c * (
        0.2969 * np.sqrt(xc)
        - 0.1260 * xc
        - 0.3516 * xc ** 2
        + 0.2843 * xc ** 3
        - 0.1036 * xc ** 4
    )


def get_airfoil_polygon(n_points=900):
    x = np.linspace(0.0, 1.0, n_points)
    yt = naca0012_thickness(x)

    upper = np.column_stack((x, yt))
    lower = np.column_stack((x[::-1], -yt[::-1]))

    coords = np.vstack((upper, lower, upper[0]))

    return Polygon(coords).buffer(0)


def build_domain(domain_type):
    if domain_type == "annulus":
        domain = make_annulus(R_INNER, R_OUTER, CIRCLE_RESOLUTION)

    elif domain_type == "box_minus_circle":
        outer_box = box(-1.0, -1.0, 1.0, 1.0)
        circle = make_circle_polygon(0.0, 0.0, 0.3, CIRCLE_RESOLUTION)
        domain = outer_box.difference(circle)

    elif domain_type == "box_minus_airfoil":
        outer_box = box(-1.0, -1.0, 2.0, 1.0)
        airfoil = get_airfoil_polygon(AIRFOIL_POINTS)
        domain = outer_box.difference(airfoil)

    else:
        raise ValueError(f"Unknown domain_type = {domain_type}")

    domain = domain.buffer(0)

    if not domain.is_valid:
        domain = domain.buffer(0)

    return domain


def generate_nodes(domain_type, domain, N_target):
    if domain_type == "annulus":
        return nodes_on_annulus(N_target, R_INNER, R_OUTER)

    return nodes_in_domain(N_target, domain)


# ============================================================
# Geometry helpers
# ============================================================

def geometry_parts(geom):
    if geom.is_empty:
        return []

    if isinstance(geom, Polygon):
        return [geom]

    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)

    if isinstance(geom, GeometryCollection):
        parts = []
        for g in geom.geoms:
            parts.extend(geometry_parts(g))
        return parts

    if hasattr(geom, "geoms"):
        parts = []
        for g in geom.geoms:
            parts.extend(geometry_parts(g))
        return parts

    return []


# ============================================================
# Voronoi lumped mass
# ============================================================

def clipped_voronoi(points, domain):
    N = len(points)

    minx, miny, maxx, maxy = domain.bounds

    length_scale = 0.5 * max(maxx - minx, maxy - miny)
    margin = 0.15 * length_scale

    left = minx - margin
    right = maxx + margin
    bottom = miny - margin
    top = maxy + margin

    mirrors = []

    for x, y in points:
        mirrors += [
            (2.0 * left - x, y),
            (2.0 * right - x, y),
            (x, 2.0 * bottom - y),
            (x, 2.0 * top - y),
        ]

    all_points = np.vstack([points, np.array(mirrors)])
    vor = Voronoi(all_points)

    fallback_half_width = 0.05 * length_scale

    cells = []

    for i in range(N):
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]

        if -1 in region or len(region) == 0:
            x, y = points[i]

            cell_poly = Polygon([
                (x - fallback_half_width, y - fallback_half_width),
                (x + fallback_half_width, y - fallback_half_width),
                (x + fallback_half_width, y + fallback_half_width),
                (x - fallback_half_width, y + fallback_half_width),
            ])
        else:
            verts = vor.vertices[region]
            cell_poly = Polygon(verts).buffer(0)

        try:
            clipped = cell_poly.intersection(domain)

            if clipped.is_empty or not clipped.is_valid:
                clipped = cell_poly.intersection(domain.buffer(1.0e-12))

        except Exception:
            clipped = Polygon()

        if not clipped.is_valid:
            clipped = clipped.buffer(0)

        cells.append(clipped)

    return cells


def lumped_mass_matrix(cells):
    return np.array([cell.area for cell in cells], dtype=float)


# ============================================================
# Polynomial basis
# ============================================================

def polynomial_powers_2d(poly_degree):
    powers = []

    for total_degree in range(poly_degree + 1):
        for a in range(total_degree + 1):
            b = total_degree - a
            powers.append((a, b))

    return powers


def polynomial_term_count_2d(poly_degree):
    return (poly_degree + 1) * (poly_degree + 2) // 2


def global_polynomial_matrices(points, poly_degree):
    powers = polynomial_powers_2d(poly_degree)

    xmin = np.min(points[:, 0])
    xmax = np.max(points[:, 0])
    ymin = np.min(points[:, 1])
    ymax = np.max(points[:, 1])

    xc = 0.5 * (xmin + xmax)
    yc = 0.5 * (ymin + ymax)

    scale = max(xmax - xmin, ymax - ymin)
    scale = max(scale, 1.0e-14)

    xs = (points[:, 0] - xc) / scale
    ys = (points[:, 1] - yc) / scale

    P = np.zeros((points.shape[0], len(powers)))
    Px = np.zeros_like(P)
    Py = np.zeros_like(P)

    for k, (a, b) in enumerate(powers):
        P[:, k] = (xs ** a) * (ys ** b)

        if a > 0:
            Px[:, k] = (a / scale) * (xs ** (a - 1)) * (ys ** b)

        if b > 0:
            Py[:, k] = (b / scale) * (xs ** a) * (ys ** (b - 1))

    return P, Px, Py


def local_polynomial_matrix(local_points, powers):
    x = local_points[:, 0]
    y = local_points[:, 1]

    P = np.zeros((local_points.shape[0], len(powers)))

    for k, (a, b) in enumerate(powers):
        P[:, k] = (x ** a) * (y ** b)

    return P


def local_polynomial_derivative_at_zero(powers, h, direction):
    rhs = np.zeros(len(powers))

    for k, (a, b) in enumerate(powers):
        if direction == "x" and a == 1 and b == 0:
            rhs[k] = 1.0 / h

        if direction == "y" and a == 0 and b == 1:
            rhs[k] = 1.0 / h

    return rhs


# ============================================================
# RBF-FD raw derivatives
# ============================================================

def choose_stencil_size(poly_degree, factor=3.5, minimum=25):
    Np = polynomial_term_count_2d(poly_degree)

    return max(minimum, int(np.ceil(factor * Np)))


def phs_matrix(local_points, phs_order):
    dx = local_points[:, 0][:, None] - local_points[:, 0][None, :]
    dy = local_points[:, 1][:, None] - local_points[:, 1][None, :]

    r = np.sqrt(dx ** 2 + dy ** 2)

    return r ** phs_order


def phs_derivative_rhs(local_points, h, phs_order, direction):
    sx = local_points[:, 0]
    sy = local_points[:, 1]

    rho = np.sqrt(sx ** 2 + sy ** 2)

    rho_power = np.zeros_like(rho)
    mask = rho > 0.0
    rho_power[mask] = rho[mask] ** (phs_order - 2)

    if direction == "x":
        return -phs_order * sx * rho_power / h

    if direction == "y":
        return -phs_order * sy * rho_power / h

    raise ValueError("direction must be x or y")


def solve_augmented_system(A, rhs):
    try:
        return np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, rhs, rcond=None)[0]


def build_rbf_fd_derivative_matrices(points, Nc, phs_order, poly_degree):
    N = points.shape[0]

    powers = polynomial_powers_2d(poly_degree)
    Np = len(powers)

    tree = cKDTree(points)

    Dx = np.zeros((N, N))
    Dy = np.zeros((N, N))
    conds = np.zeros(N)

    for i in range(N):
        distances, idx = tree.query(points[i], k=Nc)

        stencil_points = points[idx]
        h = max(np.max(distances), 1.0e-14)

        local_points = (stencil_points - points[i]) / h

        Phi = phs_matrix(local_points, phs_order)
        P_loc = local_polynomial_matrix(local_points, powers)

        A = np.zeros((Nc + Np, Nc + Np))
        A[:Nc, :Nc] = Phi
        A[:Nc, Nc:] = P_loc
        A[Nc:, :Nc] = P_loc.T

        rhs_x = np.zeros(Nc + Np)
        rhs_y = np.zeros(Nc + Np)

        rhs_x[:Nc] = phs_derivative_rhs(local_points, h, phs_order, "x")
        rhs_y[:Nc] = phs_derivative_rhs(local_points, h, phs_order, "y")

        rhs_x[Nc:] = local_polynomial_derivative_at_zero(powers, h, "x")
        rhs_y[Nc:] = local_polynomial_derivative_at_zero(powers, h, "y")

        sol_x = solve_augmented_system(A, rhs_x)
        sol_y = solve_augmented_system(A, rhs_y)

        Dx[i, idx] = sol_x[:Nc]
        Dy[i, idx] = sol_y[:Nc]

        try:
            conds[i] = np.linalg.cond(A)
        except Exception:
            conds[i] = np.nan

        if (i + 1) % 200 == 0:
            print(f"built RBF-FD rows {i + 1} / {N}")

    return Dx, Dy, conds


# ============================================================
# Path A SBP operators
# ============================================================

def weighted_matrix(weights, A):
    return weights[:, None] * A


def relative_residual(A, B):
    return np.linalg.norm(A - B, ord="fro") / max(np.linalg.norm(B, ord="fro"), 1.0e-14)


def project_symmetric_E_to_match_moments(E_start, P, moment_target):
    E_start = 0.5 * (E_start + E_start.T)
    moment_target = 0.5 * (moment_target + moment_target.T)

    U, R = np.linalg.qr(P, mode="reduced")

    current_small = U.T @ E_start @ U

    Y = np.linalg.solve(R.T, moment_target)
    target_small = np.linalg.solve(R.T, Y.T).T
    target_small = 0.5 * (target_small + target_small.T)

    E_new = E_start + U @ (target_small - current_small) @ U.T
    E_new = 0.5 * (E_new + E_new.T)

    return E_new


def minimal_skew_matrix_with_action(P, G):
    U, R = np.linalg.qr(P, mode="reduced")

    H = np.linalg.solve(R.T, G.T).T

    C = U.T @ H
    C_skew = 0.5 * (C - C.T)

    H = H + U @ (C_skew - C)

    S = H @ U.T - U @ H.T + U @ (H.T @ U) @ U.T
    S = 0.5 * (S - S.T)

    action_abs = np.linalg.norm(S @ P - G, ord="fro")
    action_rel = action_abs / max(np.linalg.norm(G, ord="fro"), 1.0e-14)

    return S, action_abs, action_rel


def minimum_change_correct_Q(Q_raw, E_compatible, P, P_deriv, weights):
    target = weighted_matrix(weights, P_deriv)

    E_raw = Q_raw + Q_raw.T
    F = E_compatible - E_raw
    F = 0.5 * (F + F.T)

    G = target - (Q_raw + 0.5 * F) @ P

    deltaS, action_abs, action_rel = minimal_skew_matrix_with_action(P, G)

    Q = Q_raw + 0.5 * F + deltaS

    relative_change = np.linalg.norm(Q - Q_raw, ord="fro") / max(
        np.linalg.norm(Q_raw, ord="fro"),
        1.0e-14,
    )

    return Q, {
        "relative_change": relative_change,
        "action_abs": action_abs,
        "action_rel": action_rel,
    }


def build_pathA_sbp_operators(points, weights, Dx_raw, Dy_raw):
    P, Px, Py = global_polynomial_matrices(points, POLY_DEGREE)

    Qx_raw = weighted_matrix(weights, Dx_raw)
    Qy_raw = weighted_matrix(weights, Dy_raw)

    Ex_raw = Qx_raw + Qx_raw.T
    Ey_raw = Qy_raw + Qy_raw.T

    B_ML_x = Px.T @ weighted_matrix(weights, P) + P.T @ weighted_matrix(weights, Px)
    B_ML_y = Py.T @ weighted_matrix(weights, P) + P.T @ weighted_matrix(weights, Py)

    Ex = project_symmetric_E_to_match_moments(Ex_raw, P, B_ML_x)
    Ey = project_symmetric_E_to_match_moments(Ey_raw, P, B_ML_y)

    Qx, info_x = minimum_change_correct_Q(Qx_raw, Ex, P, Px, weights)
    Qy, info_y = minimum_change_correct_Q(Qy_raw, Ey, P, Py, weights)

    return {
        "P": P,
        "Px": Px,
        "Py": Py,
        "Qx": Qx,
        "Qy": Qy,
        "Ex": Ex,
        "Ey": Ey,
        "info_x": info_x,
        "info_y": info_y,
    }


# ============================================================
# Operators and energy integration
# ============================================================

def symmetric_part(A):
    return 0.5 * (A + A.T)


def build_sat_matrices(Qx, Qy, Ex, Ey, lambda_vec):
    Q_lambda = lambda_vec[0] * Qx + lambda_vec[1] * Qy
    E_lambda = symmetric_part(lambda_vec[0] * Ex + lambda_vec[1] * Ey)

    eigvals, eigvecs = np.linalg.eigh(E_lambda)

    B_abs = (eigvecs * np.abs(eigvals)) @ eigvecs.T
    B_abs = symmetric_part(B_abs)

    B_minus = 0.5 * (B_abs - E_lambda)
    B_minus = symmetric_part(B_minus)

    A_closed = Q_lambda - 0.5 * E_lambda
    A_full = Q_lambda + B_minus

    closed_skew_res = np.linalg.norm(A_closed + A_closed.T, ord="fro") / max(
        np.linalg.norm(A_closed, ord="fro"),
        1.0e-14,
    )

    H_full = symmetric_part(A_full)

    sat_identity = 0.5 * E_lambda + B_minus - 0.5 * B_abs
    sat_identity_rel = np.linalg.norm(sat_identity, ord="fro") / max(
        np.linalg.norm(B_abs, ord="fro"),
        1.0e-14,
    )

    return {
        "Q_lambda": Q_lambda,
        "E_lambda": E_lambda,
        "B_abs": B_abs,
        "B_minus": B_minus,
        "A_closed": A_closed,
        "A_full": A_full,
        "closed_skew_res": closed_skew_res,
        "sat_identity_rel": sat_identity_rel,
        "min_eig_H_full": float(np.min(np.linalg.eigvalsh(H_full))),
    }


def compute_energy(u, weights):
    return 0.5 * float(np.sum(weights * u * u))


def initial_condition(points, domain):
    rp = domain.representative_point()
    x0, y0 = rp.x, rp.y

    x = points[:, 0]
    y = points[:, 1]

    sigma = 0.20

    return np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2)


def run_implicit_midpoint(weights, A, u0, dt, num_steps):
    M = np.diag(weights)

    left = M + 0.5 * dt * A
    right = M - 0.5 * dt * A

    lu_left, piv_left = lu_factor(left)

    u = u0.copy()
    energies = np.zeros(num_steps + 1)

    energies[0] = compute_energy(u, weights)

    for n in range(num_steps):
        u = lu_solve((lu_left, piv_left), right @ u)
        energies[n + 1] = compute_energy(u, weights)

    return u, energies


def energy_diagnostics(energies):
    E0 = float(energies[0])
    ET = float(energies[-1])

    jumps = np.diff(energies)

    max_jump = float(np.max(jumps))
    max_jump_rel = max_jump / max(E0, 1.0e-14)

    tol = 1.0e-10 * max(E0, 1.0)

    return {
        "energy_ratio": ET / max(E0, 1.0e-14),
        "max_jump_rel": max_jump_rel,
        "monotone": bool(np.all(jumps <= tol)),
        "max_abs_rel_energy_error": float(np.max(np.abs(energies / max(E0, 1.0e-14) - 1.0))),
    }


def min_eig_symmetric(A):
    return float(np.min(np.linalg.eigvalsh(symmetric_part(A))))


# ============================================================
# Plotting
# ============================================================

def plot_theta_scan(domain_type, theta_rows):
    theta = np.array([r["theta"] for r in theta_rows])
    energy = np.array([r["energy_ratio"] for r in theta_rows])
    min_eig = np.array([r["min_eig_H"] for r in theta_rows])
    stable = np.array([r["stable"] for r in theta_rows])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.patch.set_facecolor("white")

    axes[0].plot(theta, energy, "o-", color="#2563EB")
    axes[0].set_xlabel("theta")
    axes[0].set_ylabel("E(T) / E(0)")
    axes[0].set_title(f"Energy ratio vs SAT strength: {domain_type}")
    axes[0].grid(alpha=0.3)

    axes[1].plot(theta, min_eig, "o-", color="#DC2626")
    axes[1].axhline(0.0, color="k", lw=1.0)
    axes[1].axhline(-PSD_TOL, color="#6B7280", lw=1.0, ls="--")
    axes[1].scatter(theta[stable], min_eig[stable], color="#16A34A", zorder=4, label="PSD accepted")
    axes[1].set_xlabel("theta")
    axes[1].set_ylabel("min eig sym(A_theta)")
    axes[1].set_title("PSD stability check")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()

    fig_path = OUTPUT_DIR / f"phase3b_theta_scan_{domain_type}.png"
    fig.savefig(fig_path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")

    print(f"Figure saved -> {fig_path}")

    if SHOW_FIGS:
        plt.show()
    else:
        plt.close(fig)


def plot_summary(all_domain_rows):
    labels = sorted(set(r["domain"] for r in all_domain_rows))

    best = []
    closed = []
    full = []

    for domain in labels:
        rows = [r for r in all_domain_rows if r["domain"] == domain]

        closed_row = [r for r in rows if r["operator"] == "closed"][0]
        full_row = [r for r in rows if r["operator"] == "full_sat"][0]
        best_row = [r for r in rows if r["operator"] == "best_reduced_sat"][0]

        closed.append(closed_row["energy_ratio"])
        full.append(full_row["energy_ratio"])
        best.append(best_row["energy_ratio"])

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("white")

    ax.bar(x - width, closed, width, label="closed skew")
    ax.bar(x, best, width, label="best stable reduced SAT")
    ax.bar(x + width, full, width, label="full compatible SAT")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("E(T) / E(0)")
    ax.set_title("Phase 3b energy comparison")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    fig.tight_layout()

    fig_path = OUTPUT_DIR / "phase3b_energy_comparison_summary.png"
    fig.savefig(fig_path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")

    print(f"Figure saved -> {fig_path}")

    if SHOW_FIGS:
        plt.show()
    else:
        plt.close(fig)


# ============================================================
# One domain
# ============================================================

def run_one_domain(domain_type):
    print("\n" + "#" * 72)
    print(f"DOMAIN: {domain_type}")
    print("#" * 72)

    domain = build_domain(domain_type)
    points = generate_nodes(domain_type, domain, N_TARGETS[domain_type])

    cells = clipped_voronoi(points, domain)
    weights = lumped_mass_matrix(cells)

    if np.any(weights <= 0.0):
        raise RuntimeError(f"{domain_type}: nonpositive Voronoi mass weight found.")

    area_error = abs(np.sum(weights) - domain.area) / max(abs(domain.area), 1.0e-14)
    union_error = abs(unary_union(cells).area - domain.area) / max(abs(domain.area), 1.0e-14)

    N = len(points)
    h_cloud = np.sqrt(domain.area / N)

    dt_raw = CFL * h_cloud / max(np.linalg.norm(LAMBDA_VEC), 1.0e-14)
    num_steps = int(np.ceil(FINAL_TIME / dt_raw))
    dt = FINAL_TIME / num_steps

    Np = polynomial_term_count_2d(POLY_DEGREE)
    Nc = choose_stencil_size(POLY_DEGREE, STENCIL_FACTOR, STENCIL_MINIMUM)

    print(f"Generated nodes: {N}")
    print(f"Domain area    : {domain.area:.16e}")
    print(f"h              : {h_cloud:.6e}")
    print(f"p, Np, Nc      : {POLY_DEGREE}, {Np}, {Nc}")
    print(f"dt, steps      : {dt:.6e}, {num_steps}")

    print("Building raw RBF-FD derivative matrices...")

    Dx_raw, Dy_raw, fd_conds = build_rbf_fd_derivative_matrices(
        points,
        Nc,
        PHS_ORDER,
        POLY_DEGREE,
    )

    print("Building Path A SBP operators...")

    sbp = build_pathA_sbp_operators(points, weights, Dx_raw, Dy_raw)

    Qx = sbp["Qx"]
    Qy = sbp["Qy"]
    Ex = sbp["Ex"]
    Ey = sbp["Ey"]

    P = sbp["P"]
    Px = sbp["Px"]

    SBP_res_x = relative_residual(Qx + Qx.T, Ex)
    poly_rep_x = relative_residual(Qx @ P, weighted_matrix(weights, Px))

    print("Building SAT matrices...")

    sat = build_sat_matrices(Qx, Qy, Ex, Ey, LAMBDA_VEC)

    A_closed = sat["A_closed"]
    A_full = sat["A_full"]
    B_minus = sat["B_minus"]
    Q_lambda = sat["Q_lambda"]

    u0 = initial_condition(points, domain)

    rows = []
    theta_rows = []

    print("Running closed/skew conservation test...")

    _, energies_closed = run_implicit_midpoint(weights, A_closed, u0, dt, num_steps)
    diag_closed = energy_diagnostics(energies_closed)

    rows.append({
        "domain": domain_type,
        "operator": "closed",
        "theta": np.nan,
        "N": N,
        "p": POLY_DEGREE,
        "Nc": Nc,
        "area_error": area_error,
        "union_error": union_error,
        "SBP_res_x": SBP_res_x,
        "poly_rep_x": poly_rep_x,
        "closed_skew_res": sat["closed_skew_res"],
        "sat_identity_rel": sat["sat_identity_rel"],
        "min_eig_H": min_eig_symmetric(A_closed),
        "stable": True,
        "energy_ratio": diag_closed["energy_ratio"],
        "max_jump_rel": diag_closed["max_jump_rel"],
        "monotone": diag_closed["monotone"],
        "max_abs_rel_energy_error": diag_closed["max_abs_rel_energy_error"],
        "max_fd_cond": float(np.nanmax(fd_conds)),
    })

    print("Running full SAT test...")

    _, energies_full = run_implicit_midpoint(weights, A_full, u0, dt, num_steps)
    diag_full = energy_diagnostics(energies_full)

    rows.append({
        "domain": domain_type,
        "operator": "full_sat",
        "theta": 1.0,
        "N": N,
        "p": POLY_DEGREE,
        "Nc": Nc,
        "area_error": area_error,
        "union_error": union_error,
        "SBP_res_x": SBP_res_x,
        "poly_rep_x": poly_rep_x,
        "closed_skew_res": sat["closed_skew_res"],
        "sat_identity_rel": sat["sat_identity_rel"],
        "min_eig_H": min_eig_symmetric(A_full),
        "stable": True,
        "energy_ratio": diag_full["energy_ratio"],
        "max_jump_rel": diag_full["max_jump_rel"],
        "monotone": diag_full["monotone"],
        "max_abs_rel_energy_error": diag_full["max_abs_rel_energy_error"],
        "max_fd_cond": float(np.nanmax(fd_conds)),
    })

    print("Scanning reduced SAT theta values...")

    best_stable = None

    for theta in THETA_VALUES:
        A_theta = Q_lambda + theta * B_minus
        H_theta = symmetric_part(A_theta)
        min_eig_H = min_eig_symmetric(H_theta)

        stable = min_eig_H >= -PSD_TOL

        _, energies_theta = run_implicit_midpoint(weights, A_theta, u0, dt, num_steps)
        diag_theta = energy_diagnostics(energies_theta)

        theta_row = {
            "domain": domain_type,
            "theta": float(theta),
            "min_eig_H": float(min_eig_H),
            "stable": bool(stable),
            "energy_ratio": float(diag_theta["energy_ratio"]),
            "max_jump_rel": float(diag_theta["max_jump_rel"]),
            "monotone": bool(diag_theta["monotone"]),
            "max_abs_rel_energy_error": float(diag_theta["max_abs_rel_energy_error"]),
        }

        theta_rows.append(theta_row)

        if stable:
            if best_stable is None or theta < best_stable["theta"]:
                best_stable = theta_row

    if best_stable is None:
        best_stable = min(theta_rows, key=lambda r: r["min_eig_H"])

    rows.append({
        "domain": domain_type,
        "operator": "best_reduced_sat",
        "theta": best_stable["theta"],
        "N": N,
        "p": POLY_DEGREE,
        "Nc": Nc,
        "area_error": area_error,
        "union_error": union_error,
        "SBP_res_x": SBP_res_x,
        "poly_rep_x": poly_rep_x,
        "closed_skew_res": sat["closed_skew_res"],
        "sat_identity_rel": sat["sat_identity_rel"],
        "min_eig_H": best_stable["min_eig_H"],
        "stable": best_stable["stable"],
        "energy_ratio": best_stable["energy_ratio"],
        "max_jump_rel": best_stable["max_jump_rel"],
        "monotone": best_stable["monotone"],
        "max_abs_rel_energy_error": best_stable["max_abs_rel_energy_error"],
        "max_fd_cond": float(np.nanmax(fd_conds)),
    })

    theta_csv = OUTPUT_DIR / f"phase3b_theta_scan_{domain_type}.csv"

    with open(theta_csv, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(theta_rows[0].keys()))
        writer.writeheader()
        writer.writerows(theta_rows)

    plot_theta_scan(domain_type, theta_rows)

    print("\n" + "=" * 72)
    print(f"PHASE 3b RESULTS: {domain_type}")
    print("=" * 72)
    print(f"N nodes                         = {N}")
    print(f"area / union error              = {area_error:.6e} / {union_error:.6e}")
    print(f"SBP residual x                  = {SBP_res_x:.6e}")
    print(f"poly rep x                      = {poly_rep_x:.6e}")
    print(f"closed skew residual            = {sat['closed_skew_res']:.6e}")
    print(f"SAT identity residual           = {sat['sat_identity_rel']:.6e}")
    print("Energy comparison:")
    print(f"  closed E(T)/E(0)               = {diag_closed['energy_ratio']:.16e}")
    print(f"  closed max |E/E0-1|            = {diag_closed['max_abs_rel_energy_error']:.6e}")
    print(f"  full SAT E(T)/E(0)             = {diag_full['energy_ratio']:.16e}")
    print(f"  best theta                     = {best_stable['theta']:.6e}")
    print(f"  best theta min eig H           = {best_stable['min_eig_H']:.6e}")
    print(f"  best theta E(T)/E(0)           = {best_stable['energy_ratio']:.16e}")
    print(f"  best theta monotone            = {best_stable['monotone']}")
    print("=" * 72)

    np.savez(
        OUTPUT_DIR / f"phase3b_{domain_type}_data.npz",
        points=points,
        weights=weights,
        Qx=Qx,
        Qy=Qy,
        Ex=Ex,
        Ey=Ey,
        Q_lambda=Q_lambda,
        B_minus=B_minus,
        A_closed=A_closed,
        A_full=A_full,
        energies_closed=energies_closed,
        energies_full=energies_full,
        domain_type=domain_type,
        theta_values=np.array([r["theta"] for r in theta_rows]),
        theta_energy_ratios=np.array([r["energy_ratio"] for r in theta_rows]),
        theta_min_eigs=np.array([r["min_eig_H"] for r in theta_rows]),
        lambda_vec=LAMBDA_VEC,
    )

    return rows


# ============================================================
# Main
# ============================================================

def main():
    print("\n" + "=" * 72)
    print("PHASE 3b - PATH A SAT STRENGTH SCAN")
    print("=" * 72)
    print(f"PHS order          = {PHS_ORDER}")
    print(f"poly degree p      = {POLY_DEGREE}")
    print(f"lambda             = {LAMBDA_VEC}")
    print(f"final time         = {FINAL_TIME}")
    print(f"CFL                = {CFL}")
    print(f"theta values       = {THETA_VALUES}")
    print(f"output folder      = {OUTPUT_DIR}")
    print("=" * 72)

    all_rows = []

    for domain_type in DOMAINS:
        all_rows.extend(run_one_domain(domain_type))

    with open(CSV_FILE, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nCSV saved -> {CSV_FILE}")

    plot_summary(all_rows)

    print("\n" + "=" * 72)
    print("PHASE 3b COMPLETE")
    print("=" * 72)
    print("domain | operator | theta | minEigH | E(T)/E(0) | monotone | max energy error")
    print("-" * 72)

    for r in all_rows:
        print(
            f"{r['domain']} | "
            f"{r['operator']} | "
            f"{r['theta']} | "
            f"{r['min_eig_H']:.3e} | "
            f"{r['energy_ratio']:.6e} | "
            f"{r['monotone']} | "
            f"{r['max_abs_rel_energy_error']:.3e}"
        )

    print("-" * 72)


if __name__ == "__main__":
    main()