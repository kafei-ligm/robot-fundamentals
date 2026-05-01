"""
Example 4 — Trajectory Planning: Trapezoidal vs S-Curve
========================================================
Demonstrates:
  • Trapezoidal velocity profile (bang-coast-bang)
  • S-curve velocity profile (continuous jerk)
  • Side-by-side comparison
  • Cartesian straight-line planning
  • End-effector path visualisation on the arm

Run:
    python -m examples.04_trajectory
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
from core.trajectory import TrapezoidPlanner, SCurvePlanner, plan_cartesian_line
from core.kinematics import make_2r, ik_2r_analytic
from core.visualization import (
    draw_2r_arm, draw_workspace, plot_trajectory_1d,
    COLORS, _style_axis,
)

L1, L2 = 1.0, 0.8


def demo_profile_comparison():
    """Compare trapezoidal and S-curve profiles for the same motion."""
    trap = TrapezoidPlanner(v_max=2.0, a_max=5.0)
    scurve = SCurvePlanner(v_max=2.0, a_max=5.0, j_max=30.0)

    start, goal = 0.0, np.pi / 2

    res_t = trap.plan(start, goal, n_pts=1000)
    res_s = scurve.plan(start, goal, n_pts=2000)

    fig, axes = plt.subplots(3, 1, figsize=(9, 6), sharex=False)

    # position
    axes[0].plot(res_t.t, np.degrees(res_t.pos), color=COLORS["link1"],
                 linewidth=1.5, label="Trapezoidal")
    axes[0].plot(res_s.t, np.degrees(res_s.pos), color=COLORS["link2"],
                 linewidth=1.5, label="S-Curve")
    _style_axis(axes[0], "Position", ylabel="deg")
    axes[0].legend(fontsize=8)

    # velocity
    axes[1].plot(res_t.t, res_t.vel, color=COLORS["link1"], linewidth=1.5)
    axes[1].plot(res_s.t, res_s.vel, color=COLORS["link2"], linewidth=1.5)
    _style_axis(axes[1], "Velocity", ylabel="rad/s")

    # acceleration
    axes[2].plot(res_t.t, res_t.acc, color=COLORS["link1"], linewidth=1.5)
    axes[2].plot(res_s.t, res_s.acc, color=COLORS["link2"], linewidth=1.5)
    _style_axis(axes[2], "Acceleration", xlabel="time (s)", ylabel="rad/s²")

    fig.suptitle("Trapezoidal vs S-Curve Velocity Profile",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig("assets/04_profile_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved  assets/04_profile_comparison.png")


def demo_cartesian_path():
    """Plan a Cartesian straight line and show the arm following it."""
    p_start = np.array([1.2, 0.5, 0])
    p_goal = np.array([0.3, 1.3, 0])

    trap = TrapezoidPlanner(v_max=0.5, a_max=1.5)
    cart = plan_cartesian_line(p_start, p_goal, trap, n_pts=500)

    # convert each Cartesian point to joint angles via analytic IK
    q_traj = []
    for pos in cart["pos"]:
        sol = ik_2r_analytic(L1, L2, pos[0], pos[1], elbow_up=True)
        if sol is not None:
            q_traj.append(sol)
        else:
            q_traj.append(q_traj[-1] if q_traj else (0, 0))
    q_traj = np.array(q_traj)

    # plot arm snapshots along the path
    fig, ax = plt.subplots(figsize=(6, 6))
    draw_workspace(ax, L1, L2)

    n_snap = 10
    indices = np.linspace(0, len(q_traj) - 1, n_snap, dtype=int)
    for k, idx in enumerate(indices):
        alpha = 0.15 + 0.85 * k / (n_snap - 1)
        draw_2r_arm(ax, q_traj[idx], L1, L2, alpha=alpha, lw=2.5)

    # end-effector path
    ax.plot(cart["pos"][:, 0], cart["pos"][:, 1],
            color=COLORS["traj"], linewidth=2, linestyle="-", zorder=4,
            label="Cartesian path")
    ax.plot(*p_start[:2], "s", color=COLORS["vel"], markersize=8, zorder=6, label="start")
    ax.plot(*p_goal[:2], "*", color=COLORS["ee"], markersize=12, zorder=6, label="goal")

    ax.set_aspect("equal")
    ax.set_xlim(-1.5, 2); ax.set_ylim(-0.5, 2)
    _style_axis(ax, "Cartesian Straight-Line Path", "x (m)", "y (m)")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig("assets/04_cartesian_path.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved  assets/04_cartesian_path.png")


def demo_joint_trajectory():
    """Plot full joint-space profiles for a multi-joint motion."""
    trap = TrapezoidPlanner(v_max=1.5, a_max=4.0)

    q_start = np.radians([0, 0])
    q_goal = np.radians([90, -60])

    results = []
    for i in range(2):
        results.append(trap.plan(q_start[i], q_goal[i], n_pts=800))

    fig, axes = plt.subplots(3, 1, figsize=(9, 6), sharex=True)
    labels = ["Joint 1", "Joint 2"]
    colors = [COLORS["tau1"], COLORS["tau2"]]

    for i in range(2):
        axes[0].plot(results[i].t, np.degrees(results[i].pos),
                     color=colors[i], linewidth=1.5, label=labels[i])
        axes[1].plot(results[i].t, results[i].vel,
                     color=colors[i], linewidth=1.5, label=labels[i])
        axes[2].plot(results[i].t, results[i].acc,
                     color=colors[i], linewidth=1.5, label=labels[i])

    _style_axis(axes[0], "Joint Angle", ylabel="deg")
    _style_axis(axes[1], "Joint Velocity", ylabel="rad/s")
    _style_axis(axes[2], "Joint Acceleration", xlabel="time (s)", ylabel="rad/s²")
    for ax in axes:
        ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("Joint-Space Trapezoidal Trajectory",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig("assets/04_joint_traj.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved  assets/04_joint_traj.png")


if __name__ == "__main__":
    os.makedirs("assets", exist_ok=True)
    print("Example 4: Trajectory Planning")
    demo_profile_comparison()
    demo_cartesian_path()
    demo_joint_trajectory()
    print("Done.")
