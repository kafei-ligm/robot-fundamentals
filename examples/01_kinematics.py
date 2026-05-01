"""
Example 1 — Forward & Inverse Kinematics of a 2R Planar Arm
=============================================================
Demonstrates:
  • Forward kinematics with DH convention
  • Analytic & numerical inverse kinematics
  • Workspace boundary visualisation
  • Multi-configuration (elbow-up / elbow-down) solutions

Run:
    python -m examples.01_kinematics
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
from core.kinematics import make_2r, ik_2r_analytic
from core.visualization import draw_2r_arm, draw_workspace, COLORS, _style_axis

L1, L2 = 1.0, 0.8


def demo_forward_kinematics():
    """Show the arm at several joint configurations."""
    arm = make_2r(L1, L2)
    configs = [
        (np.radians([30, 45]),  "q = (30°, 45°)"),
        (np.radians([60, -30]), "q = (60°, -30°)"),
        (np.radians([-20, 90]), "q = (-20°, 90°)"),
        (np.radians([90, 0]),   "q = (90°, 0°)"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    for ax, (q, label) in zip(axes, configs):
        T = arm.fk(q)
        draw_workspace(ax, L1, L2)
        draw_2r_arm(ax, q, L1, L2)
        ax.set_aspect("equal")
        ax.set_xlim(-2, 2); ax.set_ylim(-2, 2)
        _style_axis(ax, label)
        ax.plot(*T[:2, 3], "x", color=COLORS["ee"], markersize=8, markeredgewidth=2)
        pos = T[:3, 3]
        ax.annotate(f"({pos[0]:.2f}, {pos[1]:.2f})",
                    (pos[0], pos[1]), fontsize=7, ha="left",
                    xytext=(5, -12), textcoords="offset points")

    fig.suptitle("Forward Kinematics — 2R Planar Arm", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig("assets/01_fk.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved  assets/01_fk.png")


def demo_inverse_kinematics():
    """Show elbow-up and elbow-down solutions for several targets."""
    targets = [
        np.array([1.2, 0.8]),
        np.array([0.5, 1.4]),
        np.array([-0.8, 0.6]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, tgt in zip(axes, targets):
        draw_workspace(ax, L1, L2)

        for elbow_up, style in [(True, 1.0), (False, 0.35)]:
            sol = ik_2r_analytic(L1, L2, tgt[0], tgt[1], elbow_up=elbow_up)
            if sol is not None:
                q = np.array(sol)
                draw_2r_arm(ax, q, L1, L2, alpha=style,
                            label="elbow-up" if elbow_up else "elbow-down")

        ax.plot(*tgt, "*", color=COLORS["ee"], markersize=12, zorder=10)
        ax.annotate(f"target ({tgt[0]:.1f}, {tgt[1]:.1f})",
                    tgt, fontsize=8, ha="left",
                    xytext=(8, 8), textcoords="offset points")
        ax.set_aspect("equal")
        ax.set_xlim(-2, 2); ax.set_ylim(-1, 2)
        _style_axis(ax)

    fig.suptitle("Inverse Kinematics — Elbow-Up vs Elbow-Down",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig("assets/01_ik.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved  assets/01_ik.png")


def demo_workspace_sweep():
    """Sweep q1, q2 and plot reachable end-effector positions."""
    arm = make_2r(L1, L2)
    q1 = np.linspace(-np.pi, np.pi, 120)
    q2 = np.linspace(-np.pi, np.pi, 120)
    Q1, Q2 = np.meshgrid(q1, q2)
    X, Y = np.zeros_like(Q1), np.zeros_like(Q1)
    for i in range(Q1.shape[0]):
        for j in range(Q1.shape[1]):
            T = arm.fk(np.array([Q1[i, j], Q2[i, j]]))
            X[i, j], Y[i, j] = T[0, 3], T[1, 3]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(X.ravel(), Y.ravel(), s=0.2, alpha=0.3, color=COLORS["traj"])
    draw_workspace(ax, L1, L2)
    ax.set_aspect("equal")
    ax.set_xlim(-2.2, 2.2); ax.set_ylim(-2.2, 2.2)
    _style_axis(ax, "Reachable Workspace", "x (m)", "y (m)")
    fig.tight_layout()
    fig.savefig("assets/01_workspace.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved  assets/01_workspace.png")


if __name__ == "__main__":
    os.makedirs("assets", exist_ok=True)
    print("Example 1: Kinematics")
    demo_forward_kinematics()
    demo_inverse_kinematics()
    demo_workspace_sweep()
    print("Done.")
