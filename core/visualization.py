"""
Visualization utilities for the robot-fundamentals package.

Provides consistent, publication-quality matplotlib figures
for kinematics, dynamics, and trajectory analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
from typing import Optional, List

# ── style ────────────────────────────────────────────────────────────
COLORS = {
    "link1": "#2563EB",
    "link2": "#F59E0B",
    "joint": "#1E293B",
    "ee":    "#EF4444",
    "ghost": "#94A3B8",
    "grid":  "#E2E8F0",
    "traj":  "#8B5CF6",
    "vel":   "#10B981",
    "acc":   "#F97316",
    "tau1":  "#2563EB",
    "tau2":  "#F59E0B",
}


def _style_axis(ax, title="", xlabel="", ylabel=""):
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, linewidth=0.3, color=COLORS["grid"])
    ax.tick_params(labelsize=8)


# ======================================================================
# Arm drawing
# ======================================================================
def draw_2r_arm(
    ax,
    q: np.ndarray,
    l1: float = 1.0,
    l2: float = 0.8,
    color_l1: str = None,
    color_l2: str = None,
    alpha: float = 1.0,
    lw: float = 4,
    label: str = "",
):
    """Draw a 2R planar arm on a matplotlib axes."""
    c1 = color_l1 or COLORS["link1"]
    c2 = color_l2 or COLORS["link2"]

    x0, y0 = 0, 0
    x1 = l1 * np.cos(q[0])
    y1 = l1 * np.sin(q[0])
    x2 = x1 + l2 * np.cos(q[0] + q[1])
    y2 = y1 + l2 * np.sin(q[0] + q[1])

    ax.plot([x0, x1], [y0, y1], color=c1, linewidth=lw, solid_capstyle="round", alpha=alpha)
    ax.plot([x1, x2], [y1, y2], color=c2, linewidth=lw, solid_capstyle="round", alpha=alpha)

    # joints
    for (jx, jy) in [(x0, y0), (x1, y1)]:
        ax.plot(jx, jy, "o", color=COLORS["joint"], markersize=7, alpha=alpha, zorder=5)
    # end-effector
    ax.plot(x2, y2, "o", color=COLORS["ee"], markersize=6, alpha=alpha, zorder=5)

    if label:
        ax.annotate(label, (x2, y2), fontsize=7, ha="left", va="bottom",
                    xytext=(5, 5), textcoords="offset points", alpha=alpha)


def draw_workspace(
    ax, l1: float = 1.0, l2: float = 0.8, n_pts: int = 200
):
    """Draw the reachable workspace annulus for a 2R arm."""
    theta = np.linspace(0, 2 * np.pi, n_pts)
    r_outer = l1 + l2
    r_inner = abs(l1 - l2)
    ax.plot(r_outer * np.cos(theta), r_outer * np.sin(theta),
            "--", color=COLORS["ghost"], linewidth=0.8, label="workspace boundary")
    if r_inner > 0.01:
        ax.plot(r_inner * np.cos(theta), r_inner * np.sin(theta),
                "--", color=COLORS["ghost"], linewidth=0.8)


# ======================================================================
# Jacobian / manipulability ellipse
# ======================================================================
def draw_velocity_ellipse(
    ax,
    J: np.ndarray,
    centre: np.ndarray,
    scale: float = 0.15,
    color: str = "#8B5CF6",
):
    """
    Draw the velocity (manipulability) ellipse at a given position.

    J should be the (2, n) linear-velocity Jacobian for a planar arm.
    """
    A = J @ J.T  # 2x2
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.maximum(eigvals, 0)

    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    w = 2 * scale * np.sqrt(eigvals[0])
    h = 2 * scale * np.sqrt(eigvals[1])

    ellipse = patches.Ellipse(
        centre, w, h, angle=angle,
        fill=False, edgecolor=color, linewidth=1.5, linestyle="-",
    )
    ax.add_patch(ellipse)


# ======================================================================
# Trajectory plots
# ======================================================================
def plot_trajectory_1d(traj, title="Trajectory Profile"):
    """Plot position, velocity, acceleration (and jerk) for a 1-D trajectory."""
    has_jerk = traj.jerk is not None
    n_rows = 4 if has_jerk else 3

    fig, axes = plt.subplots(n_rows, 1, figsize=(8, 2.2 * n_rows), sharex=True)

    axes[0].plot(traj.t, traj.pos, color=COLORS["traj"], linewidth=1.5)
    _style_axis(axes[0], "Position", ylabel="pos")

    axes[1].plot(traj.t, traj.vel, color=COLORS["vel"], linewidth=1.5)
    _style_axis(axes[1], "Velocity", ylabel="vel")

    axes[2].plot(traj.t, traj.acc, color=COLORS["acc"], linewidth=1.5)
    _style_axis(axes[2], "Acceleration", ylabel="acc")

    if has_jerk:
        axes[3].plot(traj.t, traj.jerk, color=COLORS["ee"], linewidth=1.5)
        _style_axis(axes[3], "Jerk", xlabel="time (s)", ylabel="jerk")
    else:
        axes[-1].set_xlabel("time (s)", fontsize=9)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


def plot_dynamics_sim(sim: dict, title="Dynamics Simulation"):
    """Plot joint angles, velocities, and torques from a simulation result."""
    t = sim["t"]
    n = sim["q"].shape[1]

    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)

    for j in range(n):
        c = COLORS[f"tau{j+1}"] if f"tau{j+1}" in COLORS else f"C{j}"
        axes[0].plot(t, np.degrees(sim["q"][:, j]), color=c, linewidth=1.3, label=f"q{j+1}")
        axes[1].plot(t, sim["dq"][:, j], color=c, linewidth=1.3, label=f"dq{j+1}")
        axes[2].plot(t, sim["tau"][:, j], color=c, linewidth=1.3, label=f"τ{j+1}")

    _style_axis(axes[0], "Joint Angles", ylabel="deg")
    _style_axis(axes[1], "Joint Velocities", ylabel="rad/s")
    _style_axis(axes[2], "Joint Torques", xlabel="time (s)", ylabel="N·m")

    for ax in axes:
        ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig
