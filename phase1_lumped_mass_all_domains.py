"""
Phase 1 - Lumped Mass Matrix For Three Domains
==============================================

Domains:
  1. annulus
  2. box_minus_circle
  3. box_minus_airfoil

This script follows the original annulus-only construction:
  - Halton nodes
  - mirror-point Voronoi closure ( this is important to avoid infinite Voronoi cells at the boundary and mirror points are needed for the clipped Voronoi construction 
  the mirroring is done around the bounding box of the domain, with a margin to ensure good closure even for domains with long thin features)
  - clipped Voronoi cells ( the detail of clipping is important to get accurate areas for the lumped mass, especially near boundaries. the clipping ensures that each Voronoi cell is properly intersected with the domain geometry, resulting in accurate cell areas that reflect the true portion of the domain associated with each node. this is crucial for the accuracy of the lumped mass matrix, as the diagonal entries are based on these clipped areas)
  - lumped diagonal mass M_L = diag(|Omega_i|), where Omega_i is the clipped Voronoi cell associated with node i. the formula is M_L_ii = |Omega_i|, where |Omega_i| is the area of the clipped Voronoi cell for node i. in the matrix M_L, the diagonal entry M_L_ii is equal to the area of the clipped Voronoi cell Omega_i associated with node i. this means that M_L is a diagonal matrix where each diagonal entry corresponds to the area of the respective clipped Voronoi cell.
  how to compute the lumped mass matrix M_L: 1. compute the Voronoi tessellation of the nodes, including mirror points to ensure proper closure of cells at the boundaries. 2. clip each Voronoi cell with the domain geometry to get the clipped Voronoi cells Omega_i. 3. compute the area of each clipped Voronoi cell Omega_i and set M_L_ii = |Omega_i| for each node i. this results in a lumped mass matrix M_L that is diagonal, where each diagonal entry corresponds to the area of the respective clipped Voronoi cell.


Important:
  If a clipped cell is a MultiPolygon, we keep all pieces.
"""

from pathlib import Path # for file paths and directories
import csv # for writing diagnostics to a CSV file
import warnings # for filtering warnings

import numpy as np # for numerical operations and array handling
import matplotlib # for plotting

SHOW_FIGS = True # set to False to save figures without displaying them

# Keep this commented if you want plots to appear immediately.
# matplotlib.use("Agg") # Use the Agg backend for matplotlib to allow saving figures without displaying them

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection

from scipy.spatial import Voronoi
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection, Point, box
from shapely.ops import unary_union

warnings.filterwarnings("ignore") # Ignore warnings from Shapely about invalid geometries, which can occur during Voronoi clipping but are handled by buffering with a small tolerance.


# ============================================================
# Configuration
# ============================================================

DOMAINS = [          # List of domain types to process - annulus, box with a circular hole, and box with an airfoil-shaped hole - these represent different geometric complexities for testing the lumped mass matrix construction 
    "annulus",
    "box_minus_circle",
    "box_minus_airfoil",
]

N_TARGETS = {   # Target number of nodes for each domain type - these values are chosen to provide a good balance between accuracy and computational cost for the lumped mass matrix construction. the annulus has fewer nodes due to its simpler geometry, while the box with an airfoil hole has more nodes to capture the complex geometry accurately.
    "annulus": 300,
    "box_minus_circle": 400,
    "box_minus_airfoil": 900,
}

R_INNER = 0.3 # Inner radius for the annulus domain - this defines the size of the inner hole in the annulus, which affects the geometry and the distribution of nodes and Voronoi cells in that domain.
R_OUTER = 1.0 # Outer radius for the annulus domain - this defines the size of the outer boundary in the annulus, which affects the geometry and the distribution of nodes and Voronoi cells in that domain.

CIRCLE_RESOLUTION = 256 # Number of points to use when constructing the circular boundaries for the annulus and box_minus_circle domains - a higher resolution results in smoother boundaries, which can lead to more accurate Voronoi cell clipping and area calculations for the lumped mass matrix.
AIRFOIL_POINTS = 900 # Number of points to use when constructing the airfoil shape for the box_minus_airfoil domain - a higher number of points allows for a more accurate representation of the airfoil geometry, which is important for the Voronoi cell clipping and area calculations in that domain.

FIG_DPI = 180 # DPI for saving figures - this controls the resolution of the saved images, with a higher DPI resulting in sharper images that can better show the details of the nodes, Voronoi cells, and lumped mass weights.

SCRIPT_DIR = Path(__file__).resolve().parent # Directory where the script is located - this is used as the base directory for saving output figures and diagnostics, ensuring that all outputs are organized in a consistent location relative to the script.
OUTPUT_DIR = SCRIPT_DIR / "outputs_phase1_all_domains" # Directory for saving output figures and diagnostics - this directory will contain all the generated figures and the CSV file with diagnostics for the lumped mass matrix construction across the different domains. it is created if it does not already exist, ensuring that all outputs are stored in a structured manner.
OUTPUT_DIR.mkdir(parents=True, exist_ok=True) # Create the output directory if it doesn't exist, allowing the script to save figures and diagnostics without errors related to missing directories. the 'parents=True' argument allows for the creation of any necessary parent directories, and 'exist_ok=True' prevents errors if the directory already exists. this ensures that the script can run smoothly and save all outputs as intended.

CSV_FILE = OUTPUT_DIR / "phase1_lumped_mass_diagnostics.csv" # Path to the CSV file where diagnostics will be saved - this file will contain a summary of the diagnostics for the lumped mass matrix construction for each domain type, including metrics such as area errors, minimum and maximum diagonal entries, condition number, and whether the matrix is strictly positive definite. this allows for easy analysis and comparison of the results across different domains.


# ============================================================
# Colours
# ============================================================

COL_NODE = "#2563EB"
COL_BNODE = "#DC2626"
COL_EDGE = "#EF4444"
COL_DOMAIN = "#1E3A5F"
COL_FILL = "#DBEAFE"
COL_BFILL = "#FEE2E2"


# ============================================================
# 1. Halton sequence
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

    mask = (r > r_in) & (r < r_out)
    pts = pts[mask]

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
# 2. Domain construction
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

    yt = 5.0 * t * c * (
        0.2969 * np.sqrt(xc)
        - 0.1260 * xc
        - 0.3516 * xc ** 2
        + 0.2843 * xc ** 3
        - 0.1036 * xc ** 4
    )

    return yt


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
# 3. Geometry helpers
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
# 4. Clipped Voronoi tessellation
# ============================================================

def clipped_voronoi(pts, domain_poly):
    """
    Same idea as the first annulus code:
    use mirror points around the domain bounding box to close Voronoi cells.
    """
    N = len(pts)

    minx, miny, maxx, maxy = domain_poly.bounds

    length_scale = 0.5 * max(maxx - minx, maxy - miny)
    margin = 0.15 * length_scale

    left = minx - margin
    right = maxx + margin
    bottom = miny - margin
    top = maxy + margin

    mirrors = []

    for x, y in pts:
        mirrors += [
            (2.0 * left - x, y),
            (2.0 * right - x, y),
            (x, 2.0 * bottom - y),
            (x, 2.0 * top - y),
        ]

    all_pts = np.vstack([pts, np.array(mirrors)])

    vor = Voronoi(all_pts)

    fallback_half_width = 0.05 * length_scale

    cells = []

    for i in range(N):
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]

        if -1 in region or len(region) == 0:
            x, y = pts[i]

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
            clipped = cell_poly.intersection(domain_poly)

            if clipped.is_empty or not clipped.is_valid:
                clipped = cell_poly.intersection(domain_poly.buffer(1.0e-12))

        except Exception:
            clipped = Polygon()

        if not clipped.is_valid:
            clipped = clipped.buffer(0)

        # Important:
        # Keep the full clipped geometry.
        # Do not replace MultiPolygon by only the largest polygon.
        cells.append(clipped)

    return cells


# ============================================================
# 5. Lumped mass and diagnostics
# ============================================================

def lumped_mass_matrix(cells):
    return np.array([cell.area for cell in cells], dtype=float)


def diagnostics(domain_type, pts, domain, cells, areas):
    Omega = domain.area
    total = areas.sum()

    lmin = areas.min()
    lmax = areas.max()

    area_err = abs(total - Omega) / max(abs(Omega), 1.0e-14)

    cell_union = unary_union(cells)
    union_area = cell_union.area
    union_area_err = abs(union_area - Omega) / max(abs(Omega), 1.0e-14)

    n_empty = sum(1 for c in cells if c.is_empty)
    n_nonpositive = int(np.sum(areas <= 0.0))

    row = {
        "domain": domain_type,
        "N": len(pts),
        "area_domain": Omega,
        "area_sum_weights": total,
        "relative_area_error": area_err,
        "union_area_error": union_area_err,
        "min_diag": lmin,
        "max_diag": lmax,
        "condition_number": lmax / lmin if lmin > 0.0 else np.inf,
        "empty_cells": n_empty,
        "nonpositive_diagonal_entries": n_nonpositive,
        "is_spd": bool(lmin > 0.0 and n_nonpositive == 0),
    }

    print("=" * 72)
    print(f"PHASE 1 LUMPED MASS DIAGNOSTICS: {domain_type}")
    print("=" * 72)
    print(f"N nodes                         = {row['N']}")
    print(f"empty cells                     = {n_empty}")
    print(f"|Omega|                         = {Omega:.16e}")
    print(f"1^T M_L 1                       = {total:.16e}")
    print(f"relative area error             = {area_err:.6e}")
    print(f"union area error                = {union_area_err:.6e}")
    print(f"min diag(M_L)                   = {lmin:.16e}")
    print(f"max diag(M_L)                   = {lmax:.16e}")
    print(f"condition number                = {row['condition_number']:.6e}")
    print(f"nonpositive diagonal entries    = {n_nonpositive}")
    print("diagonal by construction         = YES")
    print(f"strictly positive definite       = {'YES' if row['is_spd'] else 'NO'}")
    print("=" * 72)

    return row


# ============================================================
# 6. Plotting
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
        ax.plot(x, y, color=COL_DOMAIN, lw=1.6, zorder=8)

        for hole in poly.interiors:
            hx, hy = hole.xy
            ax.plot(hx, hy, color=COL_DOMAIN, lw=1.6, zorder=8)

    minx, miny, maxx, maxy = domain.bounds
    width = maxx - minx
    height = maxy - miny
    pad = 0.06 * max(width, height)

    ax.set_xlim(minx - pad, maxx + pad)
    ax.set_ylim(miny - pad, maxy + pad)
    ax.set_aspect("equal")
    ax.set_facecolor("#F8FAFC")
    ax.tick_params(labelsize=8)
    ax.set_xlabel("$x$", fontsize=9)
    ax.set_ylabel("$y$", fontsize=9)


def is_boundary_cell(cell, domain, tol=1.0e-3):
    if cell.is_empty:
        return False

    return cell.boundary.distance(domain.boundary) < tol


def cell_patches(cells, values=None):
    patches = []
    colors = []

    for i, cell in enumerate(cells):
        for part in geometry_parts(cell):
            try:
                coords = np.array(part.exterior.coords)
            except Exception:
                continue

            patches.append(MplPolygon(coords, closed=True))

            if values is not None:
                colors.append(values[i])

    return patches, colors


def plot_domain(domain_type, pts, domain, cells, areas):
    fig = plt.figure(figsize=(18, 6))
    fig.patch.set_facecolor("white")

    minx, miny, maxx, maxy = domain.bounds
    length_scale = max(maxx - minx, maxy - miny)
    boundary_tol_nodes = 0.02 * length_scale

    # ------------------------------------------------------------
    # Panel A: nodes
    # ------------------------------------------------------------
    ax1 = fig.add_subplot(131)
    draw_domain_background(ax1, domain)

    boundary_nodes = np.array([
        Point(float(x), float(y)).distance(domain.boundary) < boundary_tol_nodes
        for x, y in pts
    ])

    ax1.scatter(
        pts[boundary_nodes, 0],
        pts[boundary_nodes, 1],
        s=20,
        c=COL_BNODE,
        zorder=10,
        label="Boundary-near nodes",
        lw=0,
    )

    ax1.scatter(
        pts[~boundary_nodes, 0],
        pts[~boundary_nodes, 1],
        s=13,
        c=COL_NODE,
        zorder=9,
        label="Interior nodes",
        lw=0,
    )

    ax1.set_title(f"(a) Halton nodes  (N={len(pts)})", fontsize=11, pad=8)
    ax1.legend(fontsize=8, loc="upper right", framealpha=0.9)

    # ------------------------------------------------------------
    # Panel B: cells
    # ------------------------------------------------------------
    ax2 = fig.add_subplot(132)
    draw_domain_background(ax2, domain)

    interior_cells = []
    boundary_cells = []

    for cell in cells:
        if is_boundary_cell(cell, domain):
            boundary_cells.append(cell)
        else:
            interior_cells.append(cell)

    patches_int, _ = cell_patches(interior_cells)
    patches_bnd, _ = cell_patches(boundary_cells)

    pc_int = PatchCollection(
        patches_int,
        facecolor=COL_FILL,
        edgecolor=COL_EDGE,
        linewidth=0.45,
        alpha=0.90,
        zorder=3,
    )

    pc_bnd = PatchCollection(
        patches_bnd,
        facecolor=COL_BFILL,
        edgecolor=COL_EDGE,
        linewidth=0.45,
        alpha=0.95,
        zorder=4,
    )

    ax2.add_collection(pc_int)
    ax2.add_collection(pc_bnd)

    ax2.scatter(
        pts[:, 0],
        pts[:, 1],
        s=7,
        c=COL_NODE,
        zorder=10,
        lw=0,
    )

    p_int = mpatches.Patch(
        facecolor=COL_FILL,
        edgecolor=COL_EDGE,
        label="Interior cells",
    )

    p_bnd = mpatches.Patch(
        facecolor=COL_BFILL,
        edgecolor=COL_EDGE,
        label="Boundary cells",
    )

    ax2.legend(handles=[p_int, p_bnd], fontsize=8, loc="upper right", framealpha=0.9)
    ax2.set_title("(b) Clipped Voronoi tessellation", fontsize=11, pad=8)

    # ------------------------------------------------------------
    # Panel C: weights
    # ------------------------------------------------------------
    ax3 = fig.add_subplot(133)
    draw_domain_background(ax3, domain)

    patches_c, colors_c = cell_patches(cells, values=areas)

    norm = plt.Normalize(vmin=areas.min(), vmax=areas.max())

    pc_col = PatchCollection(
        patches_c,
        cmap=plt.cm.plasma,
        norm=norm,
        edgecolor="white",
        linewidth=0.25,
        zorder=3,
    )

    pc_col.set_array(np.array(colors_c))
    ax3.add_collection(pc_col)

    ax3.scatter(
        pts[:, 0],
        pts[:, 1],
        s=5,
        c="white",
        zorder=10,
        lw=0,
        alpha=0.75,
    )

    cbar = plt.colorbar(pc_col, ax=ax3, fraction=0.046, pad=0.04)
    cbar.set_label("Cell area |Omega_i| = (M_L)_ii", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ax3.set_title("(c) Lumped mass weights", fontsize=11, pad=8)

    fig.suptitle(
        f"Phase 1 - {domain_type}: Halton nodes, clipped Voronoi cells, lumped mass",
        fontsize=12,
        fontweight="bold",
        y=1.01,
    )

    plt.tight_layout()

    fig_path = OUTPUT_DIR / f"phase1_{domain_type}.png"
    plt.savefig(fig_path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")

    print(f"Figure saved -> {fig_path}")

    if SHOW_FIGS:
        plt.show()
    else:
        plt.close(fig)


def plot_histograms(rows, weights_by_domain):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("white")

    for row in rows:
        domain_type = row["domain"]
        weights = weights_by_domain[domain_type]

        ax.hist(
            weights,
            bins=32,
            alpha=0.45,
            density=True,
            label=domain_type,
        )

    ax.set_xlabel("Diagonal mass weight |Omega_i|")
    ax.set_ylabel("Density")
    ax.set_title("Lumped mass weight distributions")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    plt.tight_layout()

    fig_path = OUTPUT_DIR / "phase1_area_weights_histogram.png"
    plt.savefig(fig_path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")

    print(f"Figure saved -> {fig_path}")

    if SHOW_FIGS:
        plt.show()
    else:
        plt.close(fig)


# ============================================================
# 7. Main
# ============================================================

def run_one_domain(domain_type):
    print("\n" + "#" * 72)
    print(f"DOMAIN: {domain_type}")
    print("#" * 72)

    domain = build_domain(domain_type)

    N_target = N_TARGETS[domain_type]

    pts = generate_nodes(domain_type, domain, N_target)

    print(f"Generated nodes: {len(pts)}")
    print(f"Domain area    : {domain.area:.16e}")

    cells = clipped_voronoi(pts, domain)

    n_empty = sum(1 for c in cells if c.is_empty)

    print(f"Voronoi cells  : {len(cells) - n_empty} non-empty, {n_empty} empty")

    areas = lumped_mass_matrix(cells)

    row = diagnostics(domain_type, pts, domain, cells, areas)

    np.savez(
        OUTPUT_DIR / f"phase1_{domain_type}_data.npz",
        points=pts,
        areas=areas,
        domain_type=domain_type,
    )

    plot_domain(domain_type, pts, domain, cells, areas)

    return row, areas


def main():
    print("\n" + "=" * 72)
    print("PHASE 1 - LUMPED VORONOI DIAGONAL MASS")
    print("=" * 72)
    print(f"output folder = {OUTPUT_DIR}")
    print("=" * 72)

    rows = []
    weights_by_domain = {}

    for domain_type in DOMAINS:
        row, areas = run_one_domain(domain_type)
        rows.append(row)
        weights_by_domain[domain_type] = areas

    with open(CSV_FILE, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCSV saved -> {CSV_FILE}")

    plot_histograms(rows, weights_by_domain)

    print("\n" + "=" * 60)
    print("PHASE 1 COMPLETE")
    print("=" * 60)
    print("domain | N | min diag | area error | union error | cond | SPD")
    print("-" * 60)

    for row in rows:
        print(
            f"{row['domain']} | "
            f"{row['N']} | "
            f"{row['min_diag']:.3e} | "
            f"{row['relative_area_error']:.3e} | "
            f"{row['union_area_error']:.3e} | "
            f"{row['condition_number']:.3e} | "
            f"{row['is_spd']}"
        )

    print("-" * 60)


if __name__ == "__main__":
    main()