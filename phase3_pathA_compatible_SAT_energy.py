"""
Phase 3 - Path A Compatible SAT Energy Stability
================================================

This script uses the successful Path A construction:

    M_L = diag(Voronoi cell areas)

    Ex,Ey are M_L-compatible:
        P.T Ex P = Px.T M_L P + P.T M_L Px
        P.T Ey P = Py.T M_L P + P.T M_L Py

    Qx,Qy satisfy:
        Qx + Qx.T = Ex
        Qy + Qy.T = Ey
        Qx P = M_L Px
        Qy P = M_L Py

Then we build a compatible SAT for constant advection lambda:

    E_lambda = lambda_x Ex + lambda_y Ey

Use algebraic spectral splitting:

    E_lambda = V diag(mu) V.T
    B_abs    = V diag(abs(mu)) V.T
    B_minus  = 0.5 * (B_abs - E_lambda)

Then:

    0.5 * E_lambda + B_minus = 0.5 * B_abs >= 0

For the semi-discrete problem

    M_L u_t + (Q_lambda + B_minus) u = 0,

the energy is non-increasing.
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

RUN_RK4 = True
RUN_IMPLICIT_MIDPOINT = True

R_INNER = 0.3
R_OUTER = 1.0

CIRCLE_RESOLUTION = 256
AIRFOIL_POINTS = 900

FIG_DPI = 180

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs_phase3_pathA_compatible_SAT"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_FILE = OUTPUT_DIR / "phase3_pathA_compatible_SAT_energy.csv"


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
# Path A compatible Ex,Ey and Q correction
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

    compat_abs = np.linalg.norm(C + C.T, ord="fro")
    compat_rel = compat_abs / max(np.linalg.norm(H, ord="fro"), 1.0e-14)

    H = H + U @ (C_skew - C)

    S = H @ U.T - U @ H.T + U @ (H.T @ U) @ U.T
    S = 0.5 * (S - S.T)

    action_abs = np.linalg.norm(S @ P - G, ord="fro")
    action_rel = action_abs / max(np.linalg.norm(G, ord="fro"), 1.0e-14)

    return S, compat_abs, compat_rel, action_abs, action_rel


def minimum_change_correct_Q(Q_raw, E_compatible, P, P_deriv, weights):
    target = weighted_matrix(weights, P_deriv)

    E_raw = Q_raw + Q_raw.T
    F = E_compatible - E_raw
    F = 0.5 * (F + F.T)

    G = target - (Q_raw + 0.5 * F) @ P

    deltaS, compat_abs, compat_rel, action_abs, action_rel = minimal_skew_matrix_with_action(P, G)

    Q = Q_raw + 0.5 * F + deltaS

    relative_change = np.linalg.norm(Q - Q_raw, ord="fro") / max(
        np.linalg.norm(Q_raw, ord="fro"),
        1.0e-14,
    )

    return Q, {
        "relative_change": relative_change,
        "compat_abs": compat_abs,
        "compat_rel": compat_rel,
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
        "B_ML_x": B_ML_x,
        "B_ML_y": B_ML_y,
        "info_x": info_x,
        "info_y": info_y,
    }


# ============================================================
# Compatible SAT
# ============================================================

def symmetric_part(A):
    return 0.5 * (A + A.T)


def build_compatible_spectral_sat(Qx, Qy, Ex, Ey, lambda_vec):
    Q_lambda = lambda_vec[0] * Qx + lambda_vec[1] * Qy
    E_lambda = lambda_vec[0] * Ex + lambda_vec[1] * Ey

    E_lambda = symmetric_part(E_lambda)

    eigvals, eigvecs = np.linalg.eigh(E_lambda)

    abs_eigvals = np.abs(eigvals)

    B_abs = (eigvecs * abs_eigvals) @ eigvecs.T
    B_abs = symmetric_part(B_abs)

    B_minus = 0.5 * (B_abs - E_lambda)
    B_minus = symmetric_part(B_minus)

    B_plus = 0.5 * (B_abs + E_lambda)
    B_plus = symmetric_part(B_plus)

    A_adv = Q_lambda + B_minus

    H = symmetric_part(A_adv)

    sat_identity = 0.5 * E_lambda + B_minus - 0.5 * B_abs
    sat_identity_rel = np.linalg.norm(sat_identity, ord="fro") / max(
        np.linalg.norm(B_abs, ord="fro"),
        1.0e-14,
    )

    H_expected_rel = np.linalg.norm(H - 0.5 * B_abs, ord="fro") / max(
        np.linalg.norm(B_abs, ord="fro"),
        1.0e-14,
    )

    eig_H = np.linalg.eigvalsh(symmetric_part(H))
    eig_Babs = np.linalg.eigvalsh(symmetric_part(B_abs))
    eig_Bminus = np.linalg.eigvalsh(symmetric_part(B_minus))

    return {
        "Q_lambda": Q_lambda,
        "E_lambda": E_lambda,
        "B_abs": B_abs,
        "B_minus": B_minus,
        "B_plus": B_plus,
        "A_adv": A_adv,
        "H": H,
        "sat_identity_rel": sat_identity_rel,
        "H_expected_rel": H_expected_rel,
        "min_eig_H": float(np.min(eig_H)),
        "min_eig_Babs": float(np.min(eig_Babs)),
        "min_eig_Bminus": float(np.min(eig_Bminus)),
        "eig_E_lambda_min": float(np.min(eigvals)),
        "eig_E_lambda_max": float(np.max(eigvals)),
    }


# ============================================================
# Energy integration
# ============================================================

def compute_energy(u, weights):
    return 0.5 * float(np.sum(weights * u * u))


def compute_mass(u, weights):
    return float(np.sum(weights * u))


def initial_condition(points, domain):
    rp = domain.representative_point()
    x0, y0 = rp.x, rp.y

    x = points[:, 0]
    y = points[:, 1]

    sigma = 0.20

    return np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2)


def energy_diagnostics(energies):
    E0 = float(energies[0])
    ET = float(energies[-1])

    jumps = np.diff(energies)

    max_jump = float(np.max(jumps))
    max_jump_rel = max_jump / max(E0, 1.0e-14)

    tol = 1.0e-10 * max(E0, 1.0)

    monotone = bool(np.all(jumps <= tol))

    return {
        "energy_0": E0,
        "energy_T": ET,
        "energy_ratio": ET / max(E0, 1.0e-14),
        "max_jump": max_jump,
        "max_jump_rel": max_jump_rel,
        "monotone": monotone,
    }


def run_implicit_midpoint(weights, A, u0, dt, num_steps):
    M = np.diag(weights)

    left = M + 0.5 * dt * A
    right = M - 0.5 * dt * A

    lu_left, piv_left = lu_factor(left)

    u = u0.copy()

    energies = np.zeros(num_steps + 1)
    masses = np.zeros(num_steps + 1)

    energies[0] = compute_energy(u, weights)
    masses[0] = compute_mass(u, weights)

    for n in range(num_steps):
        u = lu_solve((lu_left, piv_left), right @ u)

        energies[n + 1] = compute_energy(u, weights)
        masses[n + 1] = compute_mass(u, weights)

    return u, energies, masses


def rk4_step(u, dt, rhs):
    k1 = rhs(u)
    k2 = rhs(u + 0.5 * dt * k1)
    k3 = rhs(u + 0.5 * dt * k2)
    k4 = rhs(u + dt * k3)

    return u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def run_rk4(weights, A, u0, dt, num_steps):
    def rhs(u):
        return -(A @ u) / weights

    u = u0.copy()

    energies = np.zeros(num_steps + 1)
    masses = np.zeros(num_steps + 1)

    energies[0] = compute_energy(u, weights)
    masses[0] = compute_mass(u, weights)

    for n in range(num_steps):
        u = rk4_step(u, dt, rhs)

        energies[n + 1] = compute_energy(u, weights)
        masses[n + 1] = compute_mass(u, weights)

    return u, energies, masses


# ============================================================
# Plotting
# ============================================================

def draw_domain_background(ax, domain):
    for poly in geometry_parts(domain):
        x, y = poly.exterior.xy
        ax.fill(x, y, color="#EFF6FF", zorder=0)

        for hole in poly.interiors:
            hx, hy = hole.xy
            ax.fill(hx, hy, color="white", zorder=1)

    for poly in geometry_parts(domain):
        x, y = poly.exterior.xy
        ax.plot(x, y, color="#1E3A5F", lw=1.3, zorder=8)

        for hole in poly.interiors:
            hx, hy = hole.xy
            ax.plot(hx, hy, color="#1E3A5F", lw=1.3, zorder=8)

    minx, miny, maxx, maxy = domain.bounds
    pad = 0.06 * max(maxx - minx, maxy - miny)

    ax.set_xlim(minx - pad, maxx + pad)
    ax.set_ylim(miny - pad, maxy + pad)
    ax.set_aspect("equal")
    ax.set_facecolor("#F8FAFC")
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def plot_energy_history(domain_type, times, energies_mid, energies_rk4):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("white")

    ax.plot(
        times,
        energies_mid / energies_mid[0],
        "k-",
        lw=2.0,
        label="implicit midpoint",
    )

    if energies_rk4 is not None:
        ax.plot(
            times,
            energies_rk4 / energies_rk4[0],
            "r--",
            lw=1.6,
            label="RK4",
        )

    ax.set_xlabel("time")
    ax.set_ylabel("E(t) / E(0)")
    ax.set_title(f"Phase 3 compatible SAT energy: {domain_type}")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()

    fig_path = OUTPUT_DIR / f"phase3_energy_history_{domain_type}.png"
    fig.savefig(fig_path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")

    print(f"Figure saved -> {fig_path}")

    if SHOW_FIGS:
        plt.show()
    else:
        plt.close(fig)


def plot_initial_final(domain_type, points, domain, u0, uT):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.patch.set_facecolor("white")

    vmin = min(float(np.min(u0)), float(np.min(uT)))
    vmax = max(float(np.max(u0)), float(np.max(uT)))

    for ax, values, title in zip(
        axes,
        [u0, uT],
        ["initial", "final implicit midpoint"],
    ):
        draw_domain_background(ax, domain)

        sc = ax.scatter(
            points[:, 0],
            points[:, 1],
            c=values,
            s=16,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            zorder=10,
            lw=0,
        )

        ax.set_title(title)

        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"Phase 3 compatible SAT solution field: {domain_type}")

    fig.tight_layout()

    fig_path = OUTPUT_DIR / f"phase3_initial_final_{domain_type}.png"
    fig.savefig(fig_path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")

    print(f"Figure saved -> {fig_path}")

    if SHOW_FIGS:
        plt.show()
    else:
        plt.close(fig)


def plot_summary(rows):
    labels = [r["domain"] for r in rows]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("white")

    ax.bar(x, [r["energy_ratio_mid"] for r in rows], color="#2563EB", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("E(T) / E(0)")
    ax.set_title("Phase 3 implicit midpoint energy ratios")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()

    fig_path = OUTPUT_DIR / "phase3_energy_ratio_summary.png"
    fig.savefig(fig_path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")

    print(f"Figure saved -> {fig_path}")

    if SHOW_FIGS:
        plt.show()
    else:
        plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(9, 5))
    fig2.patch.set_facecolor("white")

    ax2.plot(labels, [r["sat_identity_rel"] for r in rows], "o-", label="SAT identity")
    ax2.plot(labels, [r["H_expected_rel"] for r in rows], "s-", label="H - 0.5 B_abs")
    ax2.plot(labels, [r["SBP_res_x"] for r in rows], "^-", label="SBP x")
    ax2.plot(labels, [r["poly_rep_x"] for r in rows], "d-", label="poly rep x")

    ax2.set_yscale("log")
    ax2.set_ylabel("relative residual")
    ax2.set_title("Phase 3 algebraic residuals")
    ax2.grid(True, axis="y", which="both", alpha=0.3)
    ax2.legend()

    fig2.tight_layout()

    fig_path2 = OUTPUT_DIR / "phase3_algebraic_residuals_summary.png"
    fig2.savefig(fig_path2, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")

    print(f"Figure saved -> {fig_path2}")

    if SHOW_FIGS:
        plt.show()
    else:
        plt.close(fig2)


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

    lambda_norm = np.linalg.norm(LAMBDA_VEC)
    dt_raw = CFL * h_cloud / max(lambda_norm, 1.0e-14)
    num_steps = int(np.ceil(FINAL_TIME / dt_raw))
    dt = FINAL_TIME / num_steps
    times = np.linspace(0.0, FINAL_TIME, num_steps + 1)

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

    print("Building Path A M_L-compatible SBP operators...")

    sbp = build_pathA_sbp_operators(points, weights, Dx_raw, Dy_raw)

    Qx = sbp["Qx"]
    Qy = sbp["Qy"]
    Ex = sbp["Ex"]
    Ey = sbp["Ey"]
    P = sbp["P"]
    Px = sbp["Px"]

    target_x = weighted_matrix(weights, Px)

    SBP_res_x = relative_residual(Qx + Qx.T, Ex)
    SBP_res_y = relative_residual(Qy + Qy.T, Ey)
    poly_rep_x = relative_residual(Qx @ P, target_x)

    print("Building compatible spectral SAT...")

    sat = build_compatible_spectral_sat(Qx, Qy, Ex, Ey, LAMBDA_VEC)
    A_adv = sat["A_adv"]

    u0 = initial_condition(points, domain)

    u_mid = None
    energies_mid = None
    masses_mid = None
    diag_mid = None

    if RUN_IMPLICIT_MIDPOINT:
        print("Running implicit midpoint energy test...")

        u_mid, energies_mid, masses_mid = run_implicit_midpoint(
            weights,
            A_adv,
            u0,
            dt,
            num_steps,
        )

        diag_mid = energy_diagnostics(energies_mid)

    u_rk4 = None
    energies_rk4 = None
    masses_rk4 = None
    diag_rk4 = None

    if RUN_RK4:
        print("Running RK4 energy test...")

        u_rk4, energies_rk4, masses_rk4 = run_rk4(
            weights,
            A_adv,
            u0,
            dt,
            num_steps,
        )

        diag_rk4 = energy_diagnostics(energies_rk4)

    row = {
        "domain": domain_type,
        "N": N,
        "p": POLY_DEGREE,
        "Np": Np,
        "Nc": Nc,
        "phs_order": PHS_ORDER,
        "min_diag": float(np.min(weights)),
        "max_diag": float(np.max(weights)),
        "mass_condition": float(np.max(weights) / np.min(weights)),
        "area_error": float(area_error),
        "union_error": float(union_error),
        "h": float(h_cloud),
        "dt": float(dt),
        "num_steps": int(num_steps),
        "SBP_res_x": float(SBP_res_x),
        "SBP_res_y": float(SBP_res_y),
        "poly_rep_x": float(poly_rep_x),
        "Qx_relative_change": float(sbp["info_x"]["relative_change"]),
        "Qy_relative_change": float(sbp["info_y"]["relative_change"]),
        "sat_identity_rel": float(sat["sat_identity_rel"]),
        "H_expected_rel": float(sat["H_expected_rel"]),
        "min_eig_H": float(sat["min_eig_H"]),
        "min_eig_Babs": float(sat["min_eig_Babs"]),
        "min_eig_Bminus": float(sat["min_eig_Bminus"]),
        "eig_E_lambda_min": float(sat["eig_E_lambda_min"]),
        "eig_E_lambda_max": float(sat["eig_E_lambda_max"]),
        "energy_ratio_mid": float(diag_mid["energy_ratio"]) if diag_mid else np.nan,
        "max_jump_rel_mid": float(diag_mid["max_jump_rel"]) if diag_mid else np.nan,
        "monotone_mid": bool(diag_mid["monotone"]) if diag_mid else False,
        "mass_change_mid": float(masses_mid[-1] - masses_mid[0]) if masses_mid is not None else np.nan,
        "energy_ratio_rk4": float(diag_rk4["energy_ratio"]) if diag_rk4 else np.nan,
        "max_jump_rel_rk4": float(diag_rk4["max_jump_rel"]) if diag_rk4 else np.nan,
        "monotone_rk4": bool(diag_rk4["monotone"]) if diag_rk4 else False,
        "max_fd_cond": float(np.nanmax(fd_conds)),
    }

    print("\n" + "=" * 72)
    print(f"PHASE 3 PATH A COMPATIBLE SAT RESULTS: {domain_type}")
    print("=" * 72)
    print(f"N nodes                         = {row['N']}")
    print(f"min diag(M_L)                   = {row['min_diag']:.16e}")
    print(f"area / union error              = {row['area_error']:.6e} / {row['union_error']:.6e}")
    print("Path A SBP checks:")
    print(f"  SBP residual x/y               = {row['SBP_res_x']:.6e} / {row['SBP_res_y']:.6e}")
    print(f"  polynomial rep x               = {row['poly_rep_x']:.6e}")
    print(f"  Qx/Qy relative change          = {row['Qx_relative_change']:.6e} / {row['Qy_relative_change']:.6e}")
    print("Compatible SAT checks:")
    print(f"  SAT identity residual          = {row['sat_identity_rel']:.6e}")
    print(f"  H - 0.5 B_abs residual         = {row['H_expected_rel']:.6e}")
    print(f"  min eig H                      = {row['min_eig_H']:.6e}")
    print(f"  min eig B_abs                  = {row['min_eig_Babs']:.6e}")
    print(f"  min eig B_minus                = {row['min_eig_Bminus']:.6e}")
    print("Energy integration:")
    print(f"  midpoint E(T)/E(0)             = {row['energy_ratio_mid']:.16e}")
    print(f"  midpoint monotone              = {row['monotone_mid']}")
    print(f"  midpoint max jump rel          = {row['max_jump_rel_mid']:.6e}")
    print(f"  midpoint mass change           = {row['mass_change_mid']:.6e}")
    print(f"  RK4 E(T)/E(0)                  = {row['energy_ratio_rk4']:.16e}")
    print(f"  RK4 monotone                   = {row['monotone_rk4']}")
    print(f"max FD cond                      = {row['max_fd_cond']:.6e}")
    print("=" * 72)

    np.savez(
        OUTPUT_DIR / f"phase3_pathA_SAT_{domain_type}_data.npz",
        points=points,
        weights=weights,
        Qx=Qx,
        Qy=Qy,
        Ex=Ex,
        Ey=Ey,
        E_lambda=sat["E_lambda"],
        B_abs=sat["B_abs"],
        B_minus=sat["B_minus"],
        A_adv=A_adv,
        u0=u0,
        u_mid=u_mid,
        energies_mid=energies_mid,
        masses_mid=masses_mid,
        energies_rk4=energies_rk4,
        masses_rk4=masses_rk4,
        domain_type=domain_type,
        poly_degree=POLY_DEGREE,
        phs_order=PHS_ORDER,
        lambda_vec=LAMBDA_VEC,
    )

    if RUN_IMPLICIT_MIDPOINT:
        plot_initial_final(domain_type, points, domain, u0, u_mid)

    plot_energy_history(domain_type, times, energies_mid, energies_rk4)

    return row


# ============================================================
# Main
# ============================================================

def main():
    print("\n" + "=" * 72)
    print("PHASE 3 - PATH A COMPATIBLE SAT ENERGY STABILITY")
    print("=" * 72)
    print(f"PHS order          = {PHS_ORDER}")
    print(f"poly degree p      = {POLY_DEGREE}")
    print(f"lambda             = {LAMBDA_VEC}")
    print(f"final time         = {FINAL_TIME}")
    print(f"CFL                = {CFL}")
    print(f"output folder      = {OUTPUT_DIR}")
    print("=" * 72)

    rows = []

    for domain_type in DOMAINS:
        rows.append(run_one_domain(domain_type))

    with open(CSV_FILE, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCSV saved -> {CSV_FILE}")

    plot_summary(rows)

    print("\n" + "=" * 72)
    print("PHASE 3 COMPLETE")
    print("=" * 72)
    print("domain | N | SBP | SAT id | minEigH | E_mid | mono_mid | E_rk4 | mono_rk4")
    print("-" * 72)

    for r in rows:
        print(
            f"{r['domain']} | "
            f"{r['N']} | "
            f"{r['SBP_res_x']:.3e} | "
            f"{r['sat_identity_rel']:.3e} | "
            f"{r['min_eig_H']:.3e} | "
            f"{r['energy_ratio_mid']:.6e} | "
            f"{r['monotone_mid']} | "
            f"{r['energy_ratio_rk4']:.6e} | "
            f"{r['monotone_rk4']}"
        )

    print("-" * 72)


if __name__ == "__main__":
    main()