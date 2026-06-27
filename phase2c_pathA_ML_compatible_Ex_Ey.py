"""
Phase 2c - Path A: M_L-Compatible Ex,Ey + Physical Boundary Diagnostic
======================================================================

Path A decision:
    Keep the positive diagonal Voronoi mass

        M_L = diag(|Omega_i|)

    and enforce SBP compatibility with M_L, not with independently assembled
    physical boundary matrices.

Actual operators used for the next SAT phase:
    Ex,Ey are projected so that

        P.T Ex P = Px.T M_L P + P.T M_L Px
        P.T Ey P = Py.T M_L P + P.T M_L Py

    Then Qx,Qy are corrected by minimum change from raw RBF-FD:

        Qx + Qx.T = Ex
        Qy + Qy.T = Ey
        Qx P = M_L Px
        Qy P = M_L Py

Physical boundary quadrature is computed only as a diagnostic:

        Bphys_x = integral_boundary P P.T n_x ds
        Bphys_y = integral_boundary P P.T n_y ds

The mismatch

        ||Bphys_x - B_ML_x|| / ||Bphys_x||

tells us how far pure Voronoi area weights are from high-order physical
volume-boundary compatibility.
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
POLY_DEGREE = 3

STENCIL_FACTOR = 3.5
STENCIL_MINIMUM = 25

BOUNDARY_QUAD_ORDER = 8

R_INNER = 0.3
R_OUTER = 1.0

CIRCLE_RESOLUTION = 256
AIRFOIL_POINTS = 900

FIG_DPI = 180

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs_phase2c_pathA_ML_compatible"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_FILE = OUTPUT_DIR / "phase2c_pathA_ML_compatible_diagnostics.csv"


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


def ring_signed_area(coords):
    xy = np.asarray(coords)

    x = xy[:, 0]
    y = xy[:, 1]

    return 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])


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


def global_polynomial_setup(points, poly_degree):
    powers = polynomial_powers_2d(poly_degree)

    xmin = np.min(points[:, 0])
    xmax = np.max(points[:, 0])
    ymin = np.min(points[:, 1])
    ymax = np.max(points[:, 1])

    xc = 0.5 * (xmin + xmax)
    yc = 0.5 * (ymin + ymax)

    scale = max(xmax - xmin, ymax - ymin)
    scale = max(scale, 1.0e-14)

    return powers, xc, yc, scale


def evaluate_global_polynomials(points, powers, xc, yc, scale):
    xs = (points[:, 0] - xc) / scale
    ys = (points[:, 1] - yc) / scale

    P = np.zeros((points.shape[0], len(powers)))

    for k, (a, b) in enumerate(powers):
        P[:, k] = (xs ** a) * (ys ** b)

    return P


def evaluate_global_polynomial_derivatives(points, powers, xc, yc, scale):
    xs = (points[:, 0] - xc) / scale
    ys = (points[:, 1] - yc) / scale

    Px = np.zeros((points.shape[0], len(powers)))
    Py = np.zeros((points.shape[0], len(powers)))

    for k, (a, b) in enumerate(powers):
        if a > 0:
            Px[:, k] = (a / scale) * (xs ** (a - 1)) * (ys ** b)

        if b > 0:
            Py[:, k] = (b / scale) * (xs ** a) * (ys ** (b - 1))

    return Px, Py


def global_polynomial_matrices(points, poly_degree):
    powers, xc, yc, scale = global_polynomial_setup(points, poly_degree)

    P = evaluate_global_polynomials(points, powers, xc, yc, scale)
    Px, Py = evaluate_global_polynomial_derivatives(points, powers, xc, yc, scale)

    return P, Px, Py, powers, xc, yc, scale


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
# Physical boundary moment diagnostic
# ============================================================

def segment_outward_normal(p0, p1, domain, ring_coords, length_scale):
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)

    tangent = p1 - p0
    length = np.linalg.norm(tangent)

    if length <= 0.0:
        return np.array([0.0, 0.0])

    tangent = tangent / length

    left_normal = np.array([-tangent[1], tangent[0]])
    right_normal = np.array([tangent[1], -tangent[0]])

    mid = 0.5 * (p0 + p1)
    eps = 1.0e-7 * length_scale

    left_inside = domain.contains(Point(*(mid + eps * left_normal)))
    right_inside = domain.contains(Point(*(mid + eps * right_normal)))

    if left_inside and not right_inside:
        return right_normal

    if right_inside and not left_inside:
        return left_normal

    signed = ring_signed_area(ring_coords)

    if signed > 0.0:
        return right_normal

    return left_normal


def boundary_quadrature(domain, order):
    xi, wi = np.polynomial.legendre.leggauss(order)

    t_nodes = 0.5 * (xi + 1.0)
    t_weights = 0.5 * wi

    minx, miny, maxx, maxy = domain.bounds
    length_scale = max(maxx - minx, maxy - miny)

    q_points = []
    q_weights = []
    q_normals = []

    for poly in geometry_parts(domain):
        rings = [poly.exterior] + list(poly.interiors)

        for ring in rings:
            coords = np.asarray(ring.coords, dtype=float)

            for k in range(len(coords) - 1):
                p0 = coords[k]
                p1 = coords[k + 1]

                edge = p1 - p0
                length = np.linalg.norm(edge)

                if length <= 0.0:
                    continue

                normal = segment_outward_normal(p0, p1, domain, coords, length_scale)

                for t, wt in zip(t_nodes, t_weights):
                    q = (1.0 - t) * p0 + t * p1

                    q_points.append(q)
                    q_weights.append(wt * length)
                    q_normals.append(normal)

    return (
        np.asarray(q_points, dtype=float),
        np.asarray(q_weights, dtype=float),
        np.asarray(q_normals, dtype=float),
    )


def physical_boundary_moments(domain, powers, xc, yc, scale, quad_order):
    q_points, q_weights, q_normals = boundary_quadrature(domain, quad_order)

    Pq = evaluate_global_polynomials(q_points, powers, xc, yc, scale)

    Np = len(powers)

    Bx = np.zeros((Np, Np))
    By = np.zeros((Np, Np))

    for m in range(len(q_points)):
        p = Pq[m, :]
        pp = np.outer(p, p)

        Bx += q_weights[m] * q_normals[m, 0] * pp
        By += q_weights[m] * q_normals[m, 1] * pp

    return Bx, By, q_points, q_weights, q_normals


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


# ============================================================
# Accuracy diagnostics
# ============================================================

def mass_norm(v, weights):
    return np.sqrt(max(float(np.sum(weights * v * v)), 0.0))


def relative_mass_norm(error, reference, weights):
    return mass_norm(error, weights) / max(mass_norm(reference, weights), 1.0e-14)


def derivative_mms_errors(points, weights, Dx, Dy):
    x = points[:, 0]
    y = points[:, 1]

    u = np.exp(x + y)

    ux_exact = u.copy()
    uy_exact = u.copy()

    ux_num = Dx @ u
    uy_num = Dy @ u

    rel_ux = relative_mass_norm(ux_num - ux_exact, ux_exact, weights)
    rel_uy = relative_mass_norm(uy_num - uy_exact, uy_exact, weights)

    return rel_ux, rel_uy, ux_num, uy_num, ux_exact, uy_exact


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
        ax.plot(x, y, color="#1E3A5F", lw=1.4, zorder=8)

        for hole in poly.interiors:
            hx, hy = hole.xy
            ax.plot(hx, hy, color="#1E3A5F", lw=1.4, zorder=8)

    minx, miny, maxx, maxy = domain.bounds
    pad = 0.06 * max(maxx - minx, maxy - miny)

    ax.set_xlim(minx - pad, maxx + pad)
    ax.set_ylim(miny - pad, maxy + pad)
    ax.set_aspect("equal")
    ax.set_facecolor("#F8FAFC")
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def plot_boundary_quadrature(domain_type, domain, q_points, q_normals):
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("white")

    draw_domain_background(ax, domain)

    ax.scatter(
        q_points[:, 0],
        q_points[:, 1],
        s=5,
        c="#DC2626",
        zorder=10,
        lw=0,
        label="boundary quadrature",
    )

    step = max(1, len(q_points) // 120)

    ax.quiver(
        q_points[::step, 0],
        q_points[::step, 1],
        q_normals[::step, 0],
        q_normals[::step, 1],
        color="#111827",
        width=0.003,
        scale=35,
        zorder=11,
    )

    ax.set_title(f"Physical boundary diagnostic: {domain_type}")
    ax.legend(fontsize=8)

    fig.tight_layout()

    fig_path = OUTPUT_DIR / f"phase2c_boundary_quadrature_{domain_type}.png"
    fig.savefig(fig_path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")

    print(f"Figure saved -> {fig_path}")

    if SHOW_FIGS:
        plt.show()
    else:
        plt.close(fig)


def plot_derivative_error(domain_type, points, domain, err):
    emax = max(np.max(np.abs(err)), 1.0e-16)

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("white")

    draw_domain_background(ax, domain)

    sc = ax.scatter(
        points[:, 0],
        points[:, 1],
        c=err,
        s=16,
        cmap="coolwarm",
        vmin=-emax,
        vmax=emax,
        zorder=10,
        lw=0,
    )

    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("corrected Dx exp(x+y) - exp(x+y)", fontsize=8)

    ax.set_title(f"Path A corrected derivative error: {domain_type}")

    fig.tight_layout()

    fig_path = OUTPUT_DIR / f"phase2c_derivative_error_{domain_type}.png"
    fig.savefig(fig_path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")

    print(f"Figure saved -> {fig_path}")

    if SHOW_FIGS:
        plt.show()
    else:
        plt.close(fig)


def plot_summary(rows):
    labels = [r["domain"] for r in rows]
    x = np.arange(len(labels))
    width = 0.35

    raw = np.array([r["raw_deriv_ux"] for r in rows])
    corrected = np.array([r["corrected_deriv_ux"] for r in rows])

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("white")

    ax.bar(x - 0.5 * width, raw, width, label="raw RBF-FD ux")
    ax.bar(x + 0.5 * width, corrected, width, label="Path A corrected ux")

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("relative M_L error")
    ax.set_title("Path A derivative accuracy")
    ax.grid(True, axis="y", which="both", alpha=0.3)
    ax.legend()

    fig.tight_layout()

    fig_path = OUTPUT_DIR / "phase2c_derivative_accuracy_summary.png"
    fig.savefig(fig_path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")

    print(f"Figure saved -> {fig_path}")

    if SHOW_FIGS:
        plt.show()
    else:
        plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(9, 5))
    fig2.patch.set_facecolor("white")

    ax2.plot(labels, [r["ML_moment_res_x"] for r in rows], "o-", label="ML moment residual x")
    ax2.plot(labels, [r["SBP_res_x"] for r in rows], "s-", label="SBP residual x")
    ax2.plot(labels, [r["poly_rep_x"] for r in rows], "^-", label="poly reproduction x")
    ax2.plot(labels, [r["physical_vs_ML_moment_x"] for r in rows], "d-", label="physical vs ML moment x")

    ax2.set_yscale("log")
    ax2.set_ylabel("relative residual / mismatch")
    ax2.set_title("Path A algebraic residuals and physical mismatch")
    ax2.grid(True, axis="y", which="both", alpha=0.3)
    ax2.legend()

    fig2.tight_layout()

    fig_path2 = OUTPUT_DIR / "phase2c_residuals_and_physical_mismatch.png"
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

    Np = polynomial_term_count_2d(POLY_DEGREE)
    Nc = choose_stencil_size(POLY_DEGREE, STENCIL_FACTOR, STENCIL_MINIMUM)

    print(f"Generated nodes: {len(points)}")
    print(f"Domain area    : {domain.area:.16e}")
    print(f"p, Np, Nc      : {POLY_DEGREE}, {Np}, {Nc}")

    print("Building raw RBF-FD derivative matrices...")

    Dx_raw, Dy_raw, fd_conds = build_rbf_fd_derivative_matrices(
        points,
        Nc,
        PHS_ORDER,
        POLY_DEGREE,
    )

    print("Building M_L-compatible Ex,Ey...")

    P, Px, Py, powers, xc, yc, scale = global_polynomial_matrices(points, POLY_DEGREE)

    Qx_raw = weighted_matrix(weights, Dx_raw)
    Qy_raw = weighted_matrix(weights, Dy_raw)

    Ex_raw = Qx_raw + Qx_raw.T
    Ey_raw = Qy_raw + Qy_raw.T

    B_ML_x = Px.T @ weighted_matrix(weights, P) + P.T @ weighted_matrix(weights, Px)
    B_ML_y = Py.T @ weighted_matrix(weights, P) + P.T @ weighted_matrix(weights, Py)

    Ex = project_symmetric_E_to_match_moments(Ex_raw, P, B_ML_x)
    Ey = project_symmetric_E_to_match_moments(Ey_raw, P, B_ML_y)

    print("Computing physical boundary moment diagnostic...")

    B_phys_x, B_phys_y, q_points, q_weights, q_normals = physical_boundary_moments(
        domain,
        powers,
        xc,
        yc,
        scale,
        BOUNDARY_QUAD_ORDER,
    )

    physical_vs_ML_x = relative_residual(B_phys_x, B_ML_x)
    physical_vs_ML_y = relative_residual(B_phys_y, B_ML_y)

    Qx, info_x = minimum_change_correct_Q(Qx_raw, Ex, P, Px, weights)
    Qy, info_y = minimum_change_correct_Q(Qy_raw, Ey, P, Py, weights)

    Dx_corrected = Qx / weights[:, None]
    Dy_corrected = Qy / weights[:, None]

    target_x = weighted_matrix(weights, Px)
    target_y = weighted_matrix(weights, Py)

    SBP_x = relative_residual(Qx + Qx.T, Ex)
    SBP_y = relative_residual(Qy + Qy.T, Ey)

    poly_rep_x = relative_residual(Qx @ P, target_x)
    poly_rep_y = relative_residual(Qy @ P, target_y)

    ML_moment_res_x = relative_residual(P.T @ Ex @ P, B_ML_x)
    ML_moment_res_y = relative_residual(P.T @ Ey @ P, B_ML_y)

    physical_moment_res_x = relative_residual(P.T @ Ex @ P, B_phys_x)
    physical_moment_res_y = relative_residual(P.T @ Ey @ P, B_phys_y)

    raw_ux, raw_uy, _, _, _, _ = derivative_mms_errors(points, weights, Dx_raw, Dy_raw)

    corrected_ux, corrected_uy, ux_num, uy_num, ux_exact, uy_exact = derivative_mms_errors(
        points,
        weights,
        Dx_corrected,
        Dy_corrected,
    )

    row = {
        "domain": domain_type,
        "N": len(points),
        "p": POLY_DEGREE,
        "Np": Np,
        "Nc": Nc,
        "phs_order": PHS_ORDER,
        "boundary_quad_order": BOUNDARY_QUAD_ORDER,
        "min_diag": float(np.min(weights)),
        "max_diag": float(np.max(weights)),
        "mass_condition": float(np.max(weights) / np.min(weights)),
        "area_error": float(area_error),
        "union_error": float(union_error),
        "ML_moment_res_x": float(ML_moment_res_x),
        "ML_moment_res_y": float(ML_moment_res_y),
        "physical_vs_ML_moment_x": float(physical_vs_ML_x),
        "physical_vs_ML_moment_y": float(physical_vs_ML_y),
        "physical_moment_res_x": float(physical_moment_res_x),
        "physical_moment_res_y": float(physical_moment_res_y),
        "SBP_res_x": float(SBP_x),
        "SBP_res_y": float(SBP_y),
        "poly_rep_x": float(poly_rep_x),
        "poly_rep_y": float(poly_rep_y),
        "raw_deriv_ux": float(raw_ux),
        "raw_deriv_uy": float(raw_uy),
        "corrected_deriv_ux": float(corrected_ux),
        "corrected_deriv_uy": float(corrected_uy),
        "Qx_relative_change": float(info_x["relative_change"]),
        "Qy_relative_change": float(info_y["relative_change"]),
        "skew_action_abs_x": float(info_x["action_abs"]),
        "skew_action_abs_y": float(info_y["action_abs"]),
        "skew_action_rel_x": float(info_x["action_rel"]),
        "skew_action_rel_y": float(info_y["action_rel"]),
        "max_fd_cond": float(np.nanmax(fd_conds)),
        "num_boundary_quadrature_points": int(len(q_points)),
    }

    print("\n" + "=" * 72)
    print(f"PHASE 2c PATH A DIAGNOSTICS: {domain_type}")
    print("=" * 72)
    print(f"N nodes                         = {row['N']}")
    print(f"boundary quadrature points       = {row['num_boundary_quadrature_points']}")
    print(f"min diag(M_L)                   = {row['min_diag']:.16e}")
    print(f"area / union error              = {row['area_error']:.6e} / {row['union_error']:.6e}")
    print("Actual Path A algebra:")
    print(f"  ML moment residual x/y         = {row['ML_moment_res_x']:.6e} / {row['ML_moment_res_y']:.6e}")
    print(f"  SBP residual x/y               = {row['SBP_res_x']:.6e} / {row['SBP_res_y']:.6e}")
    print(f"  polynomial rep x/y             = {row['poly_rep_x']:.6e} / {row['poly_rep_y']:.6e}")
    print("Physical boundary diagnostic only:")
    print(f"  physical vs ML moment x/y       = {row['physical_vs_ML_moment_x']:.6e} / {row['physical_vs_ML_moment_y']:.6e}")
    print(f"  Ex moment vs physical x/y       = {row['physical_moment_res_x']:.6e} / {row['physical_moment_res_y']:.6e}")
    print("Correction size:")
    print(f"  Qx/Qy relative change           = {row['Qx_relative_change']:.6e} / {row['Qy_relative_change']:.6e}")
    print("Derivative MMS, u=exp(x+y):")
    print(f"  raw ux/uy                      = {row['raw_deriv_ux']:.6e} / {row['raw_deriv_uy']:.6e}")
    print(f"  corrected ux/uy                = {row['corrected_deriv_ux']:.6e} / {row['corrected_deriv_uy']:.6e}")
    print(f"max FD cond                      = {row['max_fd_cond']:.6e}")
    print("=" * 72)

    np.savez(
        OUTPUT_DIR / f"phase2c_pathA_{domain_type}_data.npz",
        points=points,
        weights=weights,
        Dx_raw=Dx_raw,
        Dy_raw=Dy_raw,
        Qx=Qx,
        Qy=Qy,
        Ex=Ex,
        Ey=Ey,
        B_ML_x=B_ML_x,
        B_ML_y=B_ML_y,
        B_phys_x=B_phys_x,
        B_phys_y=B_phys_y,
        q_points=q_points,
        q_weights=q_weights,
        q_normals=q_normals,
        domain_type=domain_type,
        poly_degree=POLY_DEGREE,
        phs_order=PHS_ORDER,
    )

    plot_boundary_quadrature(domain_type, domain, q_points, q_normals)
    plot_derivative_error(domain_type, points, domain, ux_num - ux_exact)

    return row


# ============================================================
# Main
# ============================================================

def main():
    print("\n" + "=" * 72)
    print("PHASE 2c - PATH A: M_L-COMPATIBLE Ex,Ey BEFORE SAT")
    print("=" * 72)
    print(f"PHS order          = {PHS_ORDER}")
    print(f"poly degree p      = {POLY_DEGREE}")
    print(f"boundary quad      = {BOUNDARY_QUAD_ORDER}")
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
    print("PHASE 2c COMPLETE")
    print("=" * 72)
    print("domain | N | ML moment | physical mismatch | SBP | poly | raw ux | corr ux | dQx")
    print("-" * 72)

    for r in rows:
        print(
            f"{r['domain']} | "
            f"{r['N']} | "
            f"{r['ML_moment_res_x']:.3e} | "
            f"{r['physical_vs_ML_moment_x']:.3e} | "
            f"{r['SBP_res_x']:.3e} | "
            f"{r['poly_rep_x']:.3e} | "
            f"{r['raw_deriv_ux']:.3e} | "
            f"{r['corrected_deriv_ux']:.3e} | "
            f"{r['Qx_relative_change']:.3e}"
        )

    print("-" * 72)


if __name__ == "__main__":
    main()