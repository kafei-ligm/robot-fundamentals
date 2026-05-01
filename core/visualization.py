"""
robot-fundamentals 包的可视化工具模块。

为运动学、动力学和轨迹分析提供一致的、出版级别的 matplotlib 图表。
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
from typing import Optional, List

# ── 样式配置 (Style) ────────────────────────────────────────────────────────────
COLORS = {
    "link1": "#2563EB",  # 连杆 1 颜色
    "link2": "#F59E0B",  # 连杆 2 颜色
    "joint": "#1E293B",  # 关节颜色
    "ee":    "#EF4444",  # 末端执行器 (End-effector) 颜色
    "ghost": "#94A3B8",  # 辅助线/边界虚线颜色
    "grid":  "#E2E8F0",  # 网格颜色
    "traj":  "#8B5CF6",  # 轨迹颜色
    "vel":   "#10B981",  # 速度曲线颜色
    "acc":   "#F97316",  # 加速度曲线颜色
    "tau1":  "#2563EB",  # 扭矩 1 颜色
    "tau2":  "#F59E0B",  # 扭矩 2 颜色
}


def _style_axis(ax, title="", xlabel="", ylabel=""):
    """统一的坐标轴样式设置辅助函数。"""
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, linewidth=0.3, color=COLORS["grid"])
    ax.tick_params(labelsize=8)


# ======================================================================
# 机械臂绘制
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
    """在 matplotlib 坐标轴上绘制平面 2R 机械臂。"""
    c1 = color_l1 or COLORS["link1"]
    c2 = color_l2 or COLORS["link2"]

    # 计算关节点位置
    x0, y0 = 0, 0
    x1 = l1 * np.cos(q[0])
    y1 = l1 * np.sin(q[0])
    x2 = x1 + l2 * np.cos(q[0] + q[1])
    y2 = y1 + l2 * np.sin(q[0] + q[1])

    # 绘制连杆
    ax.plot([x0, x1], [y0, y1], color=c1, linewidth=lw, solid_capstyle="round", alpha=alpha)
    ax.plot([x1, x2], [y1, y2], color=c2, linewidth=lw, solid_capstyle="round", alpha=alpha)

    # 绘制关节
    for (jx, jy) in [(x0, y0), (x1, y1)]:
        ax.plot(jx, jy, "o", color=COLORS["joint"], markersize=7, alpha=alpha, zorder=5)
    
    # 绘制末端执行器
    ax.plot(x2, y2, "o", color=COLORS["ee"], markersize=6, alpha=alpha, zorder=5)

    # 添加标签
    if label:
        ax.annotate(label, (x2, y2), fontsize=7, ha="left", va="bottom",
                    xytext=(5, 5), textcoords="offset points", alpha=alpha)


def draw_workspace(
    ax, l1: float = 1.0, l2: float = 0.8, n_pts: int = 200
):
    """绘制 2R 机械臂的可达工作空间圆环。"""
    theta = np.linspace(0, 2 * np.pi, n_pts)
    r_outer = l1 + l2
    r_inner = abs(l1 - l2)
    
    # 绘制外边界
    ax.plot(r_outer * np.cos(theta), r_outer * np.sin(theta),
            "--", color=COLORS["ghost"], linewidth=0.8, label="workspace boundary")
    
    # 绘制内边界（如果存在）
    if r_inner > 0.01:
        ax.plot(r_inner * np.cos(theta), r_inner * np.sin(theta),
                "--", color=COLORS["ghost"], linewidth=0.8)


# ======================================================================
# 雅可比矩阵 / 可操作度椭圆
# ======================================================================
def draw_velocity_ellipse(
    ax,
    J: np.ndarray,
    centre: np.ndarray,
    scale: float = 0.15,
    color: str = "#8B5CF6",
):
    """
    在给定位置绘制速度（可操作度）椭圆。

    J 必须是平面机械臂的 (2, n) 线速度雅可比矩阵。
    """
    A = J @ J.T  # 2x2 矩阵
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.maximum(eigvals, 0)  # 确保特征值为非负数

    # 计算椭圆的角度和轴长
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    w = 2 * scale * np.sqrt(eigvals[0])
    h = 2 * scale * np.sqrt(eigvals[1])

    ellipse = patches.Ellipse(
        centre, w, h, angle=angle,
        fill=False, edgecolor=color, linewidth=1.5, linestyle="-",
    )
    ax.add_patch(ellipse)


# ======================================================================
# 轨迹与动力学曲线绘制
# ======================================================================
def plot_trajectory_1d(traj, title="Trajectory Profile"):
    """绘制一维轨迹的位置、速度、加速度（以及加加速度）曲线。"""
    has_jerk = traj.jerk is not None
    n_rows = 4 if has_jerk else 3

    fig, axes = plt.subplots(n_rows, 1, figsize=(8, 2.2 * n_rows), sharex=True)

    # 位置
    axes[0].plot(traj.t, traj.pos, color=COLORS["traj"], linewidth=1.5)
    _style_axis(axes[0], "Position (位置)", ylabel="pos")

    # 速度
    axes[1].plot(traj.t, traj.vel, color=COLORS["vel"], linewidth=1.5)
    _style_axis(axes[1], "Velocity (速度)", ylabel="vel")

    # 加速度
    axes[2].plot(traj.t, traj.acc, color=COLORS["acc"], linewidth=1.5)
    _style_axis(axes[2], "Acceleration (加速度)", ylabel="acc")

    # 加加速度 (Jerk)
    if has_jerk:
        axes[3].plot(traj.t, traj.jerk, color=COLORS["ee"], linewidth=1.5)
        _style_axis(axes[3], "Jerk (加加速度)", xlabel="time (s)", ylabel="jerk")
    else:
        axes[-1].set_xlabel("time (s)", fontsize=9)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


def plot_dynamics_sim(sim: dict, title="Dynamics Simulation"):
    """根据仿真结果绘制关节角度、速度和扭矩随时间变化的曲线。"""
    t = sim["t"]
    n = sim["q"].shape[1]

    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)

    for j in range(n):
        c = COLORS[f"tau{j+1}"] if f"tau{j+1}" in COLORS else f"C{j}"
        axes[0].plot(t, np.degrees(sim["q"][:, j]), color=c, linewidth=1.3, label=f"q{j+1}")
        axes[1].plot(t, sim["dq"][:, j], color=c, linewidth=1.3, label=f"dq{j+1}")
        axes[2].plot(t, sim["tau"][:, j], color=c, linewidth=1.3, label=f"τ{j+1}")

    _style_axis(axes[0], "Joint Angles (关节角度)", ylabel="deg (度)")
    _style_axis(axes[1], "Joint Velocities (关节速度)", ylabel="rad/s")
    _style_axis(axes[2], "Joint Torques (关节扭矩)", xlabel="time (s)", ylabel="N·m")

    for ax in axes:
        ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig
