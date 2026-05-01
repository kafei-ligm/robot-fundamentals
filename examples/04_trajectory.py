"""
示例 4 — 轨迹规划：梯形速度曲线 vs S型速度曲线 (Trapezoidal vs S-Curve)
========================================================================
演示内容:
  • 梯形速度曲线 (bang-coast-bang 启停控制)
  • S型速度曲线 (连续加加速度/Jerk)
  • 两种曲线的并排对比
  • 笛卡尔空间直线轨迹规划
  • 机械臂末端执行器路径可视化

运行方式:
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
    """对比相同运动下的梯形与S型速度曲线。"""
    trap = TrapezoidPlanner(v_max=2.0, a_max=5.0)
    scurve = SCurvePlanner(v_max=2.0, a_max=5.0, j_max=30.0)

    start, goal = 0.0, np.pi / 2

    res_t = trap.plan(start, goal, n_pts=1000)
    res_s = scurve.plan(start, goal, n_pts=2000)

    fig, axes = plt.subplots(3, 1, figsize=(9, 6), sharex=False)

    # 位置 (Position)
    axes[0].plot(res_t.t, np.degrees(res_t.pos), color=COLORS["link1"],
                 linewidth=1.5, label="梯形曲线 (Trapezoidal)")
    axes[0].plot(res_s.t, np.degrees(res_s.pos), color=COLORS["link2"],
                 linewidth=1.5, label="S型曲线 (S-Curve)")
    _style_axis(axes[0], "位置 (Position)", ylabel="deg (度)")
    axes[0].legend(fontsize=8)

    # 速度 (Velocity)
    axes[1].plot(res_t.t, res_t.vel, color=COLORS["link1"], linewidth=1.5)
    axes[1].plot(res_s.t, res_s.vel, color=COLORS["link2"], linewidth=1.5)
    _style_axis(axes[1], "速度 (Velocity)", ylabel="rad/s")

    # 加速度 (Acceleration)
    axes[2].plot(res_t.t, res_t.acc, color=COLORS["link1"], linewidth=1.5)
    axes[2].plot(res_s.t, res_s.acc, color=COLORS["link2"], linewidth=1.5)
    _style_axis(axes[2], "加速度 (Acceleration)", xlabel="time (s)", ylabel="rad/s²")

    fig.suptitle("梯形 vs S型 速度曲线对比",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig("assets/04_profile_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  已保存  assets/04_profile_comparison.png")


def demo_cartesian_path():
    """规划一条笛卡尔空间直线轨迹，并展示机械臂的跟随过程。"""
    p_start = np.array([1.2, 0.5, 0])
    p_goal = np.array([0.3, 1.3, 0])

    trap = TrapezoidPlanner(v_max=0.5, a_max=1.5)
    cart = plan_cartesian_line(p_start, p_goal, trap, n_pts=500)

    # 通过解析法逆向运动学，将每个笛卡尔坐标点转换为关节角度
    q_traj = []
    for pos in cart["pos"]:
        sol = ik_2r_analytic(L1, L2, pos[0], pos[1], elbow_up=True)
        if sol is not None:
            q_traj.append(sol)
        else:
            q_traj.append(q_traj[-1] if q_traj else (0, 0))
    q_traj = np.array(q_traj)

    # 绘制沿路径的机械臂快照
    fig, ax = plt.subplots(figsize=(6, 6))
    draw_workspace(ax, L1, L2)

    n_snap = 10
    indices = np.linspace(0, len(q_traj) - 1, n_snap, dtype=int)
    for k, idx in enumerate(indices):
        alpha = 0.15 + 0.85 * k / (n_snap - 1)
        draw_2r_arm(ax, q_traj[idx], L1, L2, alpha=alpha, lw=2.5)

    # 末端执行器路径
    ax.plot(cart["pos"][:, 0], cart["pos"][:, 1],
            color=COLORS["traj"], linewidth=2, linestyle="-", zorder=4,
            label="笛卡尔直线路径 (Cartesian path)")
    ax.plot(*p_start[:2], "s", color=COLORS["vel"], markersize=8, zorder=6, label="起点 (start)")
    ax.plot(*p_goal[:2], "*", color=COLORS["ee"], markersize=12, zorder=6, label="目标点 (goal)")

    ax.set_aspect("equal")
    ax.set_xlim(-1.5, 2); ax.set_ylim(-0.5, 2)
    _style_axis(ax, "笛卡尔空间直线轨迹 (Cartesian Straight-Line Path)", "x (m)", "y (m)")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig("assets/04_cartesian_path.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  已保存  assets/04_cartesian_path.png")


def demo_joint_trajectory():
    """绘制多关节运动的完整关节空间轨迹曲线。"""
    trap = TrapezoidPlanner(v_max=1.5, a_max=4.0)

    q_start = np.radians([0, 0])
    q_goal = np.radians([90, -60])

    results = []
    for i in range(2):
        results.append(trap.plan(q_start[i], q_goal[i], n_pts=800))

    fig, axes = plt.subplots(3, 1, figsize=(9, 6), sharex=True)
    labels = ["关节 1 (Joint 1)", "关节 2 (Joint 2)"]
    colors = [COLORS["tau1"], COLORS["tau2"]]

    for i in range(2):
        axes[0].plot(results[i].t, np.degrees(results[i].pos),
                     color=colors[i], linewidth=1.5, label=labels[i])
        axes[1].plot(results[i].t, results[i].vel,
                     color=colors[i], linewidth=1.5, label=labels[i])
        axes[2].plot(results[i].t, results[i].acc,
                     color=colors[i], linewidth=1.5, label=labels[i])

    _style_axis(axes[0], "关节角度 (Joint Angle)", ylabel="deg (度)")
    _style_axis(axes[1], "关节速度 (Joint Velocity)", ylabel="rad/s")
    _style_axis(axes[2], "关节加速度 (Joint Acceleration)", xlabel="time (s)", ylabel="rad/s²")
    for ax in axes:
        ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("关节空间梯形轨迹 (Joint-Space Trapezoidal Trajectory)",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig("assets/04_joint_traj.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  已保存  assets/04_joint_traj.png")


if __name__ == "__main__":
    os.makedirs("assets", exist_ok=True)
    print("示例 4: 轨迹规划 (Trajectory Planning)")
    demo_profile_comparison()
    demo_cartesian_path()
    demo_joint_trajectory()
    print("运行完成。")
