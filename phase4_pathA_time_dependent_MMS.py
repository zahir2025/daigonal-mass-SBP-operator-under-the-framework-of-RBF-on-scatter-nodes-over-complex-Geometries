"""
Phase 4 - Path A Time-Dependent MMS
===================================

This script tests the Path A diagonal Voronoi SBP-SAT method on a
time-dependent manufactured solution.

Exact solution:
    u(x,y,t) = exp(x + y - t)

Advection:
    u_t + lambda . grad u = f

With lambda = [1,1]:

    u_t = -u
    u_x = u
    u_y = u

    f = u_t + u_x + u_y = (-1 + 1 + 1) u = u

Semi-discrete form:
    M_L u_t + Q_lambda u + theta B_minus (u - u_in) = M_L f

or

    M_L u_t = M_L f - (Q_lambda + theta B_minus) u
              + theta B_minus u_exact(t)

We compare:
    theta = 0.5  minimal stable SAT
    theta = 1.0  full compatible SAT

Time stepping:
    RK4 for accuracy.

Diagnostics:
    SBP residual
    polynomial reproduction residual
    min eig symmetric part
    relative final solution error
    time history error
"""

from pathlib import Path
import csv
import warnings

import numpy as np
import matplotlib

SHOW_FIGS = True
# matplotlib.use("Agg")

import matplotlib.pyplot as plt
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

POLY_DEGREES = [2, 3, 4, 5]

STENCIL_FACTOR = 3.5
STENCIL_MINIMUM = 25

LAMBDA_VEC = np.array([1.0, 1.0])

FINAL_TIME = 0.25
CFL = 0.004

THETA_VALUES = [0.5, 1.0]

R_INNER = 0.3
R_OUTER = 1.0

CIRCLE_RESOLUTION = 256
AIRFOIL_POINTS = 900

FIG_DPI = 180

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs_phase4_pathA_time_MMS"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_FILE = OUTPUT_DIR / "phase4_pathA_time_MMS.csv"


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


def build_pathA_sbp_operators(points, weights, Dx_raw, Dy_raw, poly_degree):
    P, Px, Py = global_polynomial_matrices(points, poly_degree)

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
# SAT matrices
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

    return {
        "Q_lambda": Q_lambda,
        "E_lambda": E_lambda,
        "B_abs": B_abs,
        "B_minus": B_minus,
    }


def min_eig_symmetric(A):
    return float(np.min(np.linalg.eigvalsh(symmetric_part(A))))


# ============================================================
# Time-dependent MMS
# ============================================================

def exact_u(points, t):
    x = points[:, 0]
    y = points[:, 1]

    return np.exp(x + y - t)


def exact_f(points, t):
    # u_t + u_x + u_y = -u + u + u = u
    return exact_u(points, t)


def mass_norm(v, weights):
    return np.sqrt(max(float(np.sum(weights * v * v)), 0.0))


def relative_mass_norm(error, reference, weights):
    return mass_norm(error, weights) / max(mass_norm(reference, weights), 1.0e-14)


def rhs_function(u, t, points, weights, sat, theta):
    Q_lambda = sat["Q_lambda"]
    B_minus = sat["B_minus"]

    f = exact_f(points, t)
    u_ex = exact_u(points, t)

    rhs_mass = weights * f - Q_lambda @ u - theta * (B_minus @ (u - u_ex))

    return rhs_mass / weights


def rk4_step(u, t, dt, rhs):
    k1 = rhs(u, t)
    k2 = rhs(u + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = rhs(u + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = rhs(u + dt * k3, t + dt)

    return u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def run_time_mms(points, weights, sat, theta, dt, num_steps):
    u = exact_u(points, 0.0)

    times = np.zeros(num_steps + 1)
    errors = np.zeros(num_steps + 1)
    energies = np.zeros(num_steps + 1)

    u_ex0 = exact_u(points, 0.0)

    errors[0] = relative_mass_norm(u - u_ex0, u_ex0, weights)
    energies[0] = 0.5 * float(np.sum(weights * u * u))

    def rhs(u_local, t_local):
        return rhs_function(u_local, t_local, points, weights, sat, theta)

    t = 0.0

    for n in range(num_steps):
        u = rk4_step(u, t, dt, rhs)
        t += dt

        u_ex = exact_u(points, t)

        times[n + 1] = t
        errors[n + 1] = relative_mass_norm(u - u_ex, u_ex, weights)
        energies[n + 1] = 0.5 * float(np.sum(weights * u * u))

    u_exact_T = exact_u(points, times[-1])
    err_T = u - u_exact_T

    return {
        "u": u,
        "u_exact_T": u_exact_T,
        "err_T": err_T,
        "times": times,
        "errors": errors,
        "energies": energies,
        "rel_err_T": float(errors[-1]),
        "max_err": float(np.max(errors)),
        "energy_ratio": float(energies[-1] / max(energies[0], 1.0e-14)),
    }


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
        ax.plot(x, y, color="#1E3A5F", lw=1.2, zorder=8)

        for hole in poly.interiors:
            hx, hy = hole.xy
            ax.plot(hx, hy, color="#1E3A5F", lw=1.2, zorder=8)

    minx, miny, maxx, maxy = domain.bounds
    pad = 0.06 * max(maxx - minx, maxy - miny)

    ax.set_xlim(minx - pad, maxx + pad)
    ax.set_ylim(miny - pad, maxy + pad)
    ax.set_aspect("equal")
    ax.set_facecolor("#F8FAFC")
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def plot_final_fields(domain_type, p, points, domain, result_min, result_full):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor("white")

    u_exact = result_min["u_exact_T"]
    u_min = result_min["u"]
    err_min = result_min["err_T"]

    vmin = min(float(np.min(u_exact)), float(np.min(u_min)))
    vmax = max(float(np.max(u_exact)), float(np.max(u_min)))

    emax = max(float(np.max(np.abs(err_min))), 1.0e-16)

    plot_data = [
        (u_exact, "exact final", "viridis", vmin, vmax),
        (u_min, "theta=0.5 final", "viridis", vmin, vmax),
        (err_min, "theta=0.5 error", "coolwarm", -emax, emax),
    ]

    for ax, (values, title, cmap, lo, hi) in zip(axes, plot_data):
        draw_domain_background(ax, domain)

        sc = ax.scatter(
            points[:, 0],
            points[:, 1],
            c=values,
            s=16,
            cmap=cmap,
            vmin=lo,
            vmax=hi,
            zorder=10,
            lw=0,
        )

        ax.set_title(title)
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Time MMS final field, {domain_type}, p={p}, "
        f"err theta=0.5 {result_min['rel_err_T']:.3e}, "
        f"err theta=1 {result_full['rel_err_T']:.3e}"
    )

    fig.tight_layout()

    fig_path = OUTPUT_DIR / f"phase4_time_MMS_fields_{domain_type}_p{p}.png"
    fig.savefig(fig_path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")

    print(f"Figure saved -> {fig_path}")

    if SHOW_FIGS:
        plt.show()
    else:
        plt.close(fig)


def plot_error_history(domain_type, p, result_min, result_full):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("white")

    ax.semilogy(
        result_min["times"],
        result_min["errors"],
        "o-",
        markevery=max(1, len(result_min["times"]) // 20),
        label="theta=0.5",
    )

    ax.semilogy(
        result_full["times"],
        result_full["errors"],
        "s--",
        markevery=max(1, len(result_full["times"]) // 20),
        label="theta=1.0",
    )

    ax.set_xlabel("time")
    ax.set_ylabel("relative M_L error")
    ax.set_title(f"Time MMS error history, {domain_type}, p={p}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    fig.tight_layout()

    fig_path = OUTPUT_DIR / f"phase4_time_MMS_error_history_{domain_type}_p{p}.png"
    fig.savefig(fig_path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")

    print(f"Figure saved -> {fig_path}")

    if SHOW_FIGS:
        plt.show()
    else:
        plt.close(fig)


def plot_summary(rows):
    domains = sorted(set(r["domain"] for r in rows))

    for domain in domains:
        rows_d = sorted([r for r in rows if r["domain"] == domain], key=lambda r: r["p"])

        p = np.array([r["p"] for r in rows_d])
        err_min = np.array([r["rel_err_theta_min"] for r in rows_d])
        err_full = np.array([r["rel_err_theta_full"] for r in rows_d])

        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor("white")

        ax.semilogy(p, err_min, "o-", label="theta=0.5")
        ax.semilogy(p, err_full, "s--", label="theta=1.0")

        ax.set_xlabel("polynomial degree p")
        ax.set_ylabel("relative final error")
        ax.set_title(f"Phase 4 time MMS final error: {domain}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()

        fig.tight_layout()

        fig_path = OUTPUT_DIR / f"phase4_time_MMS_summary_{domain}.png"
        fig.savefig(fig_path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")

        print(f"Figure saved -> {fig_path}")

        if SHOW_FIGS:
            plt.show()
        else:
            plt.close(fig)


# ============================================================
# One case
# ============================================================

def run_one_case(domain_type, p):
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

    Np = polynomial_term_count_2d(p)
    Nc = choose_stencil_size(p, STENCIL_FACTOR, STENCIL_MINIMUM)

    if Nc >= N:
        raise RuntimeError(f"Nc={Nc} >= N={N}. Increase N_TARGETS.")

    print("\n" + "-" * 72)
    print(f"Time MMS case: domain={domain_type}, p={p}")
    print("-" * 72)
    print(f"N, Np, Nc      = {N}, {Np}, {Nc}")
    print(f"h, dt, steps   = {h_cloud:.6e}, {dt:.6e}, {num_steps}")
    print(f"area error     = {area_error:.6e}")

    print("Building raw RBF-FD derivative matrices...")

    Dx_raw, Dy_raw, conds = build_rbf_fd_derivative_matrices(
        points,
        Nc,
        PHS_ORDER,
        p,
    )

    print("Building Path A SBP operators...")

    sbp = build_pathA_sbp_operators(points, weights, Dx_raw, Dy_raw, p)

    Qx = sbp["Qx"]
    Qy = sbp["Qy"]
    Ex = sbp["Ex"]
    Ey = sbp["Ey"]

    P = sbp["P"]
    Px = sbp["Px"]

    SBP_x = relative_residual(Qx + Qx.T, Ex)
    poly_x = relative_residual(Qx @ P, weighted_matrix(weights, Px))

    sat = build_sat_matrices(Qx, Qy, Ex, Ey, LAMBDA_VEC)

    H_min = sat["Q_lambda"] + THETA_VALUES[0] * sat["B_minus"]
    H_full = sat["Q_lambda"] + THETA_VALUES[1] * sat["B_minus"]

    min_eig_min = min_eig_symmetric(H_min)
    min_eig_full = min_eig_symmetric(H_full)

    print("Running time MMS theta=0.5...")

    result_min = run_time_mms(
        points,
        weights,
        sat,
        THETA_VALUES[0],
        dt,
        num_steps,
    )

    print("Running time MMS theta=1.0...")

    result_full = run_time_mms(
        points,
        weights,
        sat,
        THETA_VALUES[1],
        dt,
        num_steps,
    )

    row = {
        "domain": domain_type,
        "N": N,
        "p": p,
        "Np": Np,
        "Nc": Nc,
        "phs_order": PHS_ORDER,
        "min_diag": float(np.min(weights)),
        "area_error": float(area_error),
        "union_error": float(union_error),
        "h": float(h_cloud),
        "dt": float(dt),
        "num_steps": int(num_steps),
        "SBP_x": float(SBP_x),
        "poly_x": float(poly_x),
        "Qx_relative_change": float(sbp["info_x"]["relative_change"]),
        "max_fd_cond": float(np.nanmax(conds)),
        "min_eig_H_theta_min": float(min_eig_min),
        "min_eig_H_theta_full": float(min_eig_full),
        "rel_err_theta_min": float(result_min["rel_err_T"]),
        "max_err_theta_min": float(result_min["max_err"]),
        "energy_ratio_theta_min": float(result_min["energy_ratio"]),
        "rel_err_theta_full": float(result_full["rel_err_T"]),
        "max_err_theta_full": float(result_full["max_err"]),
        "energy_ratio_theta_full": float(result_full["energy_ratio"]),
    }

    print("Results:")
    print(f"  SBP x                         = {row['SBP_x']:.6e}")
    print(f"  poly x                        = {row['poly_x']:.6e}")
    print(f"  min eig H theta=0.5            = {row['min_eig_H_theta_min']:.6e}")
    print(f"  min eig H theta=1.0            = {row['min_eig_H_theta_full']:.6e}")
    print(f"  final error theta=0.5          = {row['rel_err_theta_min']:.6e}")
    print(f"  final error theta=1.0          = {row['rel_err_theta_full']:.6e}")
    print(f"  energy ratio theta=0.5         = {row['energy_ratio_theta_min']:.6e}")
    print(f"  energy ratio theta=1.0         = {row['energy_ratio_theta_full']:.6e}")

    plot_final_fields(domain_type, p, points, domain, result_min, result_full)
    plot_error_history(domain_type, p, result_min, result_full)

    np.savez(
        OUTPUT_DIR / f"phase4_time_MMS_{domain_type}_p{p}.npz",
        points=points,
        weights=weights,
        u_theta_min=result_min["u"],
        u_theta_full=result_full["u"],
        u_exact_T=result_min["u_exact_T"],
        err_theta_min=result_min["err_T"],
        err_theta_full=result_full["err_T"],
        times=result_min["times"],
        errors_theta_min=result_min["errors"],
        errors_theta_full=result_full["errors"],
        Qx=Qx,
        Qy=Qy,
        Ex=Ex,
        Ey=Ey,
        domain_type=domain_type,
        p=p,
        phs_order=PHS_ORDER,
        lambda_vec=LAMBDA_VEC,
    )

    return row


# ============================================================
# Main
# ============================================================

def main():
    print("\n" + "=" * 72)
    print("PHASE 4 - PATH A TIME-DEPENDENT MMS")
    print("=" * 72)
    print(f"PHS order          = {PHS_ORDER}")
    print(f"poly degrees       = {POLY_DEGREES}")
    print(f"lambda             = {LAMBDA_VEC}")
    print(f"final time         = {FINAL_TIME}")
    print(f"CFL                = {CFL}")
    print(f"theta values       = {THETA_VALUES}")
    print(f"output folder      = {OUTPUT_DIR}")
    print("=" * 72)

    rows = []

    for domain_type in DOMAINS:
        print("\n" + "#" * 72)
        print(f"DOMAIN: {domain_type}")
        print("#" * 72)

        for p in POLY_DEGREES:
            rows.append(run_one_case(domain_type, p))

    with open(CSV_FILE, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCSV saved -> {CSV_FILE}")

    plot_summary(rows)

    print("\n" + "=" * 72)
    print("PHASE 4 COMPLETE")
    print("=" * 72)
    print("domain | p | N | SBP | poly | err theta=0.5 | err theta=1.0 | E theta=0.5 | E theta=1.0")
    print("-" * 72)

    for r in rows:
        print(
            f"{r['domain']} | "
            f"{r['p']} | "
            f"{r['N']} | "
            f"{r['SBP_x']:.3e} | "
            f"{r['poly_x']:.3e} | "
            f"{r['rel_err_theta_min']:.3e} | "
            f"{r['rel_err_theta_full']:.3e} | "
            f"{r['energy_ratio_theta_min']:.3e} | "
            f"{r['energy_ratio_theta_full']:.3e}"
        )

    print("-" * 72)


if __name__ == "__main__":
    main()