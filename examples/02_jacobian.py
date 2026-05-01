"""
示例 2 — 雅可比矩阵分析与可操作度 (Jacobian Analysis & Manipulability)
====================================================================
演示内容:
  • 几何雅可比矩阵 (Geometric Jacobian) 的计算
  • 速度映射 (关节空间速度 → 末端执行器笛卡尔空间速度)
  • 不同姿态下的可操作度椭圆 (Manipulability ellipse)
  • 奇异点 (Singularity) 检测

运行方式:
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
    """在整个关节空间内绘制可操作度指数的分布热力图。"""
    arm = make_2r(L1, L2)
    n = 100
    q1 = np.linspace(-np.pi, np.pi, n)
    q2 = np.linspace(-np.pi, np.pi, n)
    Q1, Q2 = np.meshgrid(q1, q2)
    W = np.zeros_like(Q1)

    # 计算整个网格的可操作度指数 (w = sqrt(det(J*J^T)))
    for i in range(n):
        for j in range(n):
            q = np.array([Q1[i, j], Q2[i, j]])
            W[i, j] = arm.manipulability(q)

    fig, ax = plt.subplots(figsize=(6, 5))
    c = ax.pcolormesh(np.degrees(Q1), np.degrees(Q2), W,
                      cmap="viridis", shading="auto")
    fig.colorbar(c, ax=ax, label="可操作度指数 (Manipulability Index)")
    
    # 绘制奇异点等高线 (w接近于0的地方)
    ax.contour(np.degrees(Q1), np.degrees(Q2), W,
               levels=[0.01], colors=["red"], linewidths=1.5)
    _style_axis(ax, "可操作度分布图 (Manipulability Map)", "q1 (deg)", "q2 (deg)")
    ax.annotate("奇异边界 / Singular curves (w ≈ 0)", xy=(0, 0), fontsize=8,
                color="red", ha="center",
                xytext=(0, -140), textcoords="offset points")

    fig.tight_layout()
    fig.savefig("assets/02_manipulability_map.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  已保存  assets/02_manipulability_map.png")


def demo_velocity_ellipses():
    """在几种代表性姿态下绘制机械臂及其速度椭圆。"""
    arm = make_2r(L1, L2)
    configs = [
        (np.radians([45, 60]),  "灵活性良好\n(Good dexterity)"),
        (np.radians([30, 10]),  "接近奇异点\n(Near singularity)"),
        (np.radians([60, -90]), "折叠状态\n(Folded)"),
        (np.radians([0, 180]),  "奇异点伸展状态\n(Singular)"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    for ax, (q, title) in zip(axes, configs):
        draw_workspace(ax, L1, L2)
        draw_2r_arm(ax, q, L1, L2)

        T = arm.fk(q)
        ee = T[:2, 3]
        J = arm.jacobian(q)
        Jv = J[:2, :]   # 取前两行，即平面的线速度雅可比矩阵
        draw_velocity_ellipse(ax, Jv, ee, scale=0.2)

        w = arm.manipulability(q)
        _style_axis(ax, f"{title}\nw = {w:.3f}")
        ax.set_aspect("equal")
        ax.set_xlim(-2, 2); ax.set_ylim(-1.5, 2)

    fig.suptitle("速度椭圆 — 可操作度分析 (Velocity Ellipses)",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig("assets/02_ellipses.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  已保存  assets/02_ellipses.png")


def demo_velocity_mapping():
    """展示关节速度矢量如何通过雅可比矩阵映射为末端执行器的速度矢量。"""
    arm = make_2r(L1, L2)
    q = np.radians([45, 60])
    J = arm.jacobian(q)
    Jv = J[:2, :]  # 平面 2x2

    # 施加不同的关节速度输入样本 [dq1, dq2]
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
    # 绘制背景速度椭圆以作参考
    draw_velocity_ellipse(ax, Jv, ee, scale=0.2, color="#CBD5E1")

    # 映射并绘制末端速度向量
    for dq in dq_samples:
        v_ee = Jv @ dq
        scale = 0.15
        # 绘制箭头
        ax.annotate("", xy=(ee[0] + scale * v_ee[0], ee[1] + scale * v_ee[1]),
                     xytext=(ee[0], ee[1]),
                     arrowprops=dict(arrowstyle="->", color=COLORS["ee"], lw=1.8))
        ax.annotate(f"dq=({dq[0]:.0f},{dq[1]:.0f})",
                    (ee[0] + scale * v_ee[0], ee[1] + scale * v_ee[1]),
                    fontsize=7, ha="center", va="bottom",
                    xytext=(0, 4), textcoords="offset points")

    ax.set_aspect("equal")
    ax.set_xlim(-2, 2); ax.set_ylim(-1, 2)
    _style_axis(ax, "关节 → 末端速度映射 (Velocity Mapping)", "x (m)", "y (m)")
    fig.tight_layout()
    fig.savefig("assets/02_velocity_map.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  已保存  assets/02_velocity_map.png")


if __name__ == "__main__":
    os.makedirs("assets", exist_ok=True)
    print("示例 2: 雅可比矩阵分析 (Jacobian Analysis)")
    demo_manipulability_map()
    demo_velocity_ellipses()
    demo_velocity_mapping()
    print("运行完成。")
