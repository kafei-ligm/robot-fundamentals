"""
示例 1 — 二自由度 (2R) 平面机械臂的正向与逆向运动学
=============================================================
演示内容:
  • 使用 DH 参数法的正向运动学
  • 解析法与数值法逆向运动学
  • 工作空间边界可视化
  • 多姿态 (elbow-up / elbow-down，即手肘向上/手肘向下) 求解

运行方式:
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
    """展示机械臂在不同关节配置下的状态。"""
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
        
        # 标记末端执行器位置
        ax.plot(*T[:2, 3], "x", color=COLORS["ee"], markersize=8, markeredgewidth=2)
        pos = T[:3, 3]
        ax.annotate(f"({pos[0]:.2f}, {pos[1]:.2f})",
                    (pos[0], pos[1]), fontsize=7, ha="left",
                    xytext=(5, -12), textcoords="offset points")

    fig.suptitle("正向运动学 (Forward Kinematics) — 2R 平面机械臂", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig("assets/01_fk.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  已保存  assets/01_fk.png")


def demo_inverse_kinematics():
    """展示多个目标点的“手肘向上”(elbow-up) 和“手肘向下”(elbow-down) 的逆解。"""
    targets = [
        np.array([1.2, 0.8]),
        np.array([0.5, 1.4]),
        np.array([-0.8, 0.6]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, tgt in zip(axes, targets):
        draw_workspace(ax, L1, L2)

        # 分别计算并绘制手肘向上和向下的解
        for elbow_up, style in [(True, 1.0), (False, 0.35)]:
            sol = ik_2r_analytic(L1, L2, tgt[0], tgt[1], elbow_up=elbow_up)
            if sol is not None:
                q = np.array(sol)
                draw_2r_arm(ax, q, L1, L2, alpha=style,
                            label="手肘向上" if elbow_up else "手肘向下")

        # 绘制目标点
        ax.plot(*tgt, "*", color=COLORS["ee"], markersize=12, zorder=10)
        ax.annotate(f"目标点 ({tgt[0]:.1f}, {tgt[1]:.1f})",
                    tgt, fontsize=8, ha="left",
                    xytext=(8, 8), textcoords="offset points")
        ax.set_aspect("equal")
        ax.set_xlim(-2, 2); ax.set_ylim(-1, 2)
        _style_axis(ax)

    fig.suptitle("逆向运动学 (Inverse Kinematics) — 手肘向上 vs 手肘向下",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig("assets/01_ik.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  已保存  assets/01_ik.png")


def demo_workspace_sweep():
    """遍历 q1, q2 并绘制末端执行器的可达位置。"""
    arm = make_2r(L1, L2)
    q1 = np.linspace(-np.pi, np.pi, 120)
    q2 = np.linspace(-np.pi, np.pi, 120)
    Q1, Q2 = np.meshgrid(q1, q2)
    X, Y = np.zeros_like(Q1), np.zeros_like(Q1)
    
    # 穷举计算正运动学求末端位置
    for i in range(Q1.shape[0]):
        for j in range(Q1.shape[1]):
            T = arm.fk(np.array([Q1[i, j], Q2[i, j]]))
            X[i, j], Y[i, j] = T[0, 3], T[1, 3]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(X.ravel(), Y.ravel(), s=0.2, alpha=0.3, color=COLORS["traj"])
    draw_workspace(ax, L1, L2)
    ax.set_aspect("equal")
    ax.set_xlim(-2.2, 2.2); ax.set_ylim(-2.2, 2.2)
    _style_axis(ax, "可达工作空间 (Reachable Workspace)", "x (m)", "y (m)")
    fig.tight_layout()
    fig.savefig("assets/01_workspace.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  已保存  assets/01_workspace.png")


if __name__ == "__main__":
    os.makedirs("assets", exist_ok=True)
    print("示例 1: 运动学 (Kinematics)")
    demo_forward_kinematics()
    demo_inverse_kinematics()
    demo_workspace_sweep()
    print("运行完成。")
