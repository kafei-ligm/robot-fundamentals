"""
Example 2 — Jacobian Analysis & Manipulability
===============================================
Demonstrates:
  • Geometric Jacobian computation
  • Velocity mapping  (joint velocity → end-effector velocity)
  • Manipulability ellipse at different configurations
  • Singularity detection

Run:
    python -m examples.02_jacobian
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
from core.kinematics import make_2r
from core.visualization import (
    draw_2r_arm, draw_workspace, draw_velocity_ellipse,
    COLORS, _style_axis,
)

L1, L2 = 1.0, 0.8


def demo_manipulability_map():
    """Plot manipulability index over the workspace."""
    arm = make_2r(L1, L2)
    n = 100
    q1 = np.linspace(-np.pi, np.pi, n)
    q2 = np.linspace(-np.pi, np.pi, n)
    Q1, Q2 = np.meshgrid(q1, q2)
    W = np.zeros_like(Q1)

    for i in range(n):
        for j in range(n):
            q = np.array([Q1[i, j], Q2[i, j]])
            W[i, j] = arm.manipulability(q)

    fig, ax = plt.subplots(figsize=(6, 5))
    c = ax.pcolormesh(np.degrees(Q1), np.degrees(Q2), W,
                      cmap="viridis", shading="auto")
    fig.colorbar(c, ax=ax, label="manipulability index")
    ax.contour(np.degrees(Q1), np.degrees(Q2), W,
               levels=[0.01], colors=["red"], linewidths=1.5)
    _style_axis(ax, "Manipulability Map", "q1 (deg)", "q2 (deg)")
    ax.annotate("singular curves (w ≈ 0)", xy=(0, 0), fontsize=8,
                color="red", ha="center",
                xytext=(0, -140), textcoords="offset points")

    fig.tight_layout()
    fig.savefig("assets/02_manipulability_map.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved  assets/02_manipulability_map.png")


def demo_velocity_ellipses():
    """Draw the arm with velocity ellipses at several configurations."""
    arm = make_2r(L1, L2)
    configs = [
        (np.radians([45, 60]),  "Good dexterity"),
        (np.radians([30, 10]),  "Near singularity"),
        (np.radians([60, -90]), "Folded"),
        (np.radians([0, 180]),  "Singular (stretched)"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    for ax, (q, title) in zip(axes, configs):
        draw_workspace(ax, L1, L2)
        draw_2r_arm(ax, q, L1, L2)

        T = arm.fk(q)
        ee = T[:2, 3]
        J = arm.jacobian(q)
        Jv = J[:2, :]   # planar linear Jacobian
        draw_velocity_ellipse(ax, Jv, ee, scale=0.2)

        w = arm.manipulability(q)
        _style_axis(ax, f"{title}\nw = {w:.3f}")
        ax.set_aspect("equal")
        ax.set_xlim(-2, 2); ax.set_ylim(-1.5, 2)

    fig.suptitle("Velocity Ellipses — Manipulability Analysis",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig("assets/02_ellipses.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved  assets/02_ellipses.png")


def demo_velocity_mapping():
    """Show how joint velocities map to end-effector velocities."""
    arm = make_2r(L1, L2)
    q = np.radians([45, 60])
    J = arm.jacobian(q)
    Jv = J[:2, :]  # 2x2 for planar

    # apply different joint velocity vectors
    dq_samples = [
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
        np.array([1.0, 1.0]),
        np.array([1.0, -1.0]),
    ]

    T = arm.fk(q)
    ee = T[:2, 3]

    fig, ax = plt.subplots(figsize=(6, 6))
    draw_workspace(ax, L1, L2)
    draw_2r_arm(ax, q, L1, L2)
    draw_velocity_ellipse(ax, Jv, ee, scale=0.2, color="#CBD5E1")

    for dq in dq_samples:
        v_ee = Jv @ dq
        scale = 0.15
        ax.annotate("", xy=(ee[0] + scale * v_ee[0], ee[1] + scale * v_ee[1]),
                     xytext=(ee[0], ee[1]),
                     arrowprops=dict(arrowstyle="->", color=COLORS["ee"], lw=1.8))
        ax.annotate(f"dq=({dq[0]:.0f},{dq[1]:.0f})",
                    (ee[0] + scale * v_ee[0], ee[1] + scale * v_ee[1]),
                    fontsize=7, ha="center", va="bottom",
                    xytext=(0, 4), textcoords="offset points")

    ax.set_aspect("equal")
    ax.set_xlim(-2, 2); ax.set_ylim(-1, 2)
    _style_axis(ax, "Joint → End-Effector Velocity Mapping", "x (m)", "y (m)")
    fig.tight_layout()
    fig.savefig("assets/02_velocity_map.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved  assets/02_velocity_map.png")


if __name__ == "__main__":
    os.makedirs("assets", exist_ok=True)
    print("Example 2: Jacobian Analysis")
    demo_manipulability_map()
    demo_velocity_ellipses()
    demo_velocity_mapping()
    print("Done.")
