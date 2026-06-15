"""
DBSCAN FEA Stress Hotspot Clustering — Demo
============================================
Demonstrates the methodology used in a production tool built at BSH Hausgeräte GmbH
(Bosch Group) to automatically detect and rank structural stress hotspots from FEA results.

In production: nodal data is read from Ansys result databases via PyAnsys.
Here: synthetic stress field data is generated to demonstrate the algorithm.

Dependencies: numpy, scikit-learn, matplotlib
Run: python dbscan_fea_clustering_demo.py

Author: Mohd Aaquib Khan
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN


# ── 1. GENERATE SYNTHETIC FEA STRESS FIELD ───────────────────────────────────
# Simulates a flat plate-like mesh with 3 stress concentration zones
# (e.g. around holes, notches, or weld toes) plus background noise.

np.random.seed(42)

def stress_field(n_background=2000, hotspot_params=None):
    """
    Generate synthetic nodal stress data mimicking an FEA result.
    Returns: array of shape (N, 3) — columns [x, y, von_mises_stress_MPa]
    """
    if hotspot_params is None:
        # Each hotspot: (x_centre, y_centre, spread_x, spread_y, peak_stress, n_nodes)
        hotspot_params = [
            (30,  70,  4,  3, 320, 180),   # Hotspot A — notch / stress riser
            (75,  40,  3,  4, 280, 150),   # Hotspot B — weld toe
            (60,  85,  2,  2, 245, 100),   # Hotspot C — hole edge
        ]

    rows = []

    # Background: low stress, scattered across a 100x100 domain
    x_bg = np.random.uniform(0, 100, n_background)
    y_bg = np.random.uniform(0, 100, n_background)
    s_bg = np.random.uniform(10, 80, n_background)   # background 10–80 MPa
    rows.append(np.column_stack([x_bg, y_bg, s_bg]))

    # Hotspots: clustered nodes with elevated stress
    for (xc, yc, sx, sy, peak, n) in hotspot_params:
        x_h = np.random.normal(xc, sx, n)
        y_h = np.random.normal(yc, sy, n)
        # Stress falls off from peak at centre
        dist = np.sqrt(((x_h - xc)/sx)**2 + ((y_h - yc)/sy)**2)
        s_h  = peak * np.exp(-0.5 * dist**2) + np.random.normal(0, 10, n)
        s_h  = np.clip(s_h, 50, peak * 1.05)
        rows.append(np.column_stack([x_h, y_h, s_h]))

    data = np.vstack(rows)
    return data


# ── 2. DBSCAN CLUSTERING ──────────────────────────────────────────────────────

def detect_hotspots(nodes, stress_threshold_percentile=85, eps=0.35, min_samples=5):
    """
    Apply DBSCAN to detect spatially dense, high-stress clusters.

    Strategy mirrors the production tool:
    1. Filter to nodes above a stress threshold (reduces noise, focuses on critical regions)
    2. Scale spatial coordinates (DBSCAN is distance-based)
    3. Run DBSCAN on (x, y) of filtered nodes
    4. Rank resulting clusters by peak and mean stress

    Args:
        nodes                     : (N, 3) array — [x, y, stress]
        stress_threshold_percentile: only cluster nodes above this stress percentile
        eps                       : DBSCAN neighbourhood radius (in scaled units)
        min_samples               : minimum points to form a core cluster

    Returns:
        hotspots (list of dicts), labels array for all filtered nodes
    """
    threshold = np.percentile(nodes[:, 2], stress_threshold_percentile)
    mask      = nodes[:, 2] >= threshold
    high_nodes = nodes[mask]

    print(f"\nTotal nodes:          {len(nodes):,}")
    print(f"Stress threshold:     {threshold:.1f} MPa  (p{stress_threshold_percentile})")
    print(f"Nodes above threshold:{mask.sum():,}  ({mask.mean()*100:.1f}%)")

    # Scale spatial coords so DBSCAN treats x and y equally
    scaler  = StandardScaler()
    coords  = scaler.fit_transform(high_nodes[:, :2])

    labels  = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(coords)

    unique_labels = set(labels) - {-1}   # -1 = noise (unclustered)
    print(f"Clusters detected:    {len(unique_labels)}")
    print(f"Noise nodes:          {(labels == -1).sum()}")

    hotspots = []
    for lbl in sorted(unique_labels):
        cluster_nodes = high_nodes[labels == lbl]
        hotspots.append({
            "id":         lbl,
            "n_nodes":    len(cluster_nodes),
            "peak_MPa":   cluster_nodes[:, 2].max(),
            "mean_MPa":   cluster_nodes[:, 2].mean(),
            "x_centre":   cluster_nodes[:, 0].mean(),
            "y_centre":   cluster_nodes[:, 1].mean(),
            "nodes":      cluster_nodes,
        })

    # Sort by peak stress descending — highest severity first
    hotspots.sort(key=lambda h: h["peak_MPa"], reverse=True)
    for rank, h in enumerate(hotspots, 1):
        h["rank"] = rank

    return hotspots, labels, high_nodes


# ── 3. REPORT ─────────────────────────────────────────────────────────────────

def print_report(hotspots):
    """Console report — in production this writes to a PDF/Excel."""
    print("\n" + "=" * 60)
    print("  STRESS HOTSPOT REPORT")
    print("=" * 60)
    print(f"  {'Rank':<6} {'Nodes':<8} {'Peak (MPa)':<14} {'Mean (MPa)':<14} {'Location (x, y)'}")
    print("-" * 60)
    for h in hotspots:
        print(f"  {h['rank']:<6} {h['n_nodes']:<8} "
              f"{h['peak_MPa']:<14.1f} {h['mean_MPa']:<14.1f} "
              f"({h['x_centre']:.1f}, {h['y_centre']:.1f})")
    print("=" * 60)
    print("\n  -> In production: report exported to Excel/PDF with")
    print("     node IDs, load-case context, and design recommendations.")


# ── 4. VISUALISATION ──────────────────────────────────────────────────────────

def plot_results(all_nodes, hotspots, labels, high_nodes):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("DBSCAN FEA Stress Hotspot Detection — Demo",
                 fontsize=13, fontweight="bold", color="#1F3A5F")

    colours = plt.cm.tab10.colors

    # ── Left: full stress field ──
    ax = axes[0]
    sc = ax.scatter(all_nodes[:, 0], all_nodes[:, 1],
                    c=all_nodes[:, 2], cmap="RdYlGn_r",
                    s=3, alpha=0.6, rasterized=True)
    plt.colorbar(sc, ax=ax, label="Von Mises Stress (MPa)")
    ax.set_title("Synthetic FEA Stress Field", fontweight="bold")
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)")
    ax.set_aspect("equal")

    # ── Right: DBSCAN clusters ──
    ax = axes[1]
    # Background high-stress nodes (noise = grey)
    noise_mask = (labels == -1)
    ax.scatter(high_nodes[noise_mask, 0], high_nodes[noise_mask, 1],
               c="lightgray", s=6, label="High-stress (unclustered)", zorder=1)

    legend_patches = [mpatches.Patch(color="lightgray", label="High-stress (noise)")]
    for h in hotspots:
        col = colours[h["id"] % len(colours)]
        ax.scatter(h["nodes"][:, 0], h["nodes"][:, 1],
                   c=[col], s=12, zorder=2)
        ax.annotate(f"Rank {h['rank']}\n{h['peak_MPa']:.0f} MPa",
                    xy=(h["x_centre"], h["y_centre"]),
                    xytext=(8, 8), textcoords="offset points",
                    fontsize=8, color=col, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
        legend_patches.append(
            mpatches.Patch(color=col,
                           label=f"Cluster {h['rank']}  |  Peak {h['peak_MPa']:.0f} MPa"))

    ax.legend(handles=legend_patches, fontsize=8, loc="lower right")
    ax.set_title("DBSCAN Hotspot Clusters (ranked by severity)",
                 fontweight="bold")
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)")
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig("dbscan_hotspot_result.png", dpi=150, bbox_inches="tight")
    print("\n  Plot saved: dbscan_hotspot_result.png")
    plt.show()


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating synthetic FEA stress field ...")
    nodes = stress_field()

    print("Running DBSCAN hotspot detection ...")
    hotspots, labels, high_nodes = detect_hotspots(nodes)

    print_report(hotspots)
    plot_results(nodes, hotspots, labels, high_nodes)
