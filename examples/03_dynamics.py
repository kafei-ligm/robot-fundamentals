"""
示例 3 — 拉格朗日动力学与仿真 (Lagrangian Dynamics & Simulation)
=================================================================
演示内容:
  • 质量矩阵 (Mass matrix)、科里奥利力 (Coriolis) 和重力 (Gravity) 计算
  • 逆向动力学 (运动 → 扭矩)
  • 正向动力学 (扭矩 → 运动)
  • 重力作用下的自由落体仿真
  • 基于 PD 控制的关节轨迹跟踪

运行方式:
    python -m examples.03_dynamics
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
from core.dynamics import make_2r_dynamics
from core.visualization import (
    draw_2r_arm, COLORS, _style_axis, plot_dynamics_sim,
)

L1, L2 = 1.0, 0.8


def demo_inverse_dynamics():
    """在示例姿态下计算并显示 M, C, g 矩阵/向量。"""
    dyn = make_2r_dynamics(L1, L2)

    q = np.radians([45, 30])
    dq = np.array([0.5, -0.3])
    ddq = np.array([0.1, 0.2])

    tau, terms = dyn.inverse_dynamics(q, dq, ddq)

    print("\n  姿态配置:  q = (45°, 30°),  dq = (0.5, -0.3) rad/s")
    print(f"  质量矩阵 (Mass matrix M):\n{terms['M']}")
    print(f"  科里奥利力 (Coriolis C):\n{terms['C']}")
    print(f"  重力向量 (Gravity g): {terms['g']}")
    print(f"  所需扭矩: τ = {tau}")

    # 使用正向动力学进行验证
    ddq_check, _ = dyn.forward_dynamics(q, dq, tau)
    print(f"  正向动力学验证: ddq = {ddq_check}  (应该近似于 {ddq})")
    print(f"  误差范数 (Error norm): {np.linalg.norm(ddq_check - ddq):.2e}")


def demo_free_fall():
    """模拟机械臂从水平姿态在重力作用下自由落体的过程。"""
    dyn = make_2r_dynamics(L1, L2)
    q0 = np.array([0.0, 0.0])   # 水平姿态
    dq0 = np.zeros(2)

    # 仿真：不施加任何扭矩 (tau = 0)
    sim = dyn.simulate(q0, dq0, tau_fn=lambda t, q, dq: np.zeros(2),
                       dt=5e-4, duration=2.0)

    # 轨迹图
    fig = plot_dynamics_sim(sim, title="重力下的自由落体 (Free Fall Under Gravity - 无扭矩)")
    fig.savefig("assets/03_freefall.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  已保存  assets/03_freefall.png")

    # 过程快照
    fig, ax = plt.subplots(figsize=(7, 5))
    n_frames = 8
    indices = np.linspace(0, len(sim["t"]) - 1, n_frames, dtype=int)
    for k, idx in enumerate(indices):
        alpha = 0.2 + 0.8 * k / (n_frames - 1)
        draw_2r_arm(ax, sim["q"][idx], L1, L2, alpha=alpha,
                    label=f"t={sim['t'][idx]:.2f}s")
    ax.set_aspect("equal")
    ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.5, 1.5)
    _style_axis(ax, "自由落体过程中的机械臂快照 (Arm Snapshots)", "x (m)", "y (m)")
    fig.tight_layout()
    fig.savefig("assets/03_freefall_snapshots.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  已保存  assets/03_freefall_snapshots.png")


def demo_pd_tracking():
    """使用 PD 控制器跟踪正弦参考轨迹。"""
    dyn = make_2r_dynamics(L1, L2)

    # 期望轨迹 (Desired trajectory)
    def q_des(t):
        return np.array([
            np.radians(45) * np.sin(2 * np.pi * 0.5 * t),
            np.radians(30) * np.sin(2 * np.pi * 0.3 * t + np.pi / 4),
        ])

    def dq_des(t):
        return np.array([
            np.radians(45) * 2 * np.pi * 0.5 * np.cos(2 * np.pi * 0.5 * t),
            np.radians(30) * 2 * np.pi * 0.3 * np.cos(2 * np.pi * 0.3 * t + np.pi / 4),
        ])

    Kp = np.diag([50.0, 30.0])
    Kd = np.diag([10.0, 6.0])

    def pd_controller(t, q, dq):
        e = q_des(t) - q
        de = dq_des(t) - dq
        # 类似计算扭矩法 (Computed-Torque): PD 控制 + 重力补偿
        g = dyn.gravity_vector(q)
        return Kp @ e + Kd @ de + g

    q0 = np.zeros(2)
    dq0 = np.zeros(2)

    sim = dyn.simulate(q0, dq0, tau_fn=pd_controller,
                       dt=5e-4, duration=4.0)

    # 计算期望轨迹以便对比
    t_arr = sim["t"]
    q_ref = np.array([q_des(t) for t in t_arr])

    fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    for j in range(2):
        c = COLORS[f"tau{j+1}"]
        axes[0].plot(t_arr, np.degrees(sim["q"][:, j]),
                     color=c, linewidth=1.3, label=f"q{j+1} 实际值")
        axes[0].plot(t_arr, np.degrees(q_ref[:, j]),
                     "--", color=c, linewidth=1, alpha=0.6, label=f"q{j+1} 期望值")
        axes[1].plot(t_arr, sim["tau"][:, j],
                     color=c, linewidth=1.3, label=f"τ{j+1}")

    _style_axis(axes[0], "PD 跟踪 — 关节角度", ylabel="deg (度)")
    _style_axis(axes[1], "控制扭矩 (Control Torques)", xlabel="time (s)", ylabel="N·m")
    axes[0].legend(fontsize=7, ncol=2, loc="upper right")
    axes[1].legend(fontsize=8, loc="upper right")

    fig.suptitle("PD 控制 + 重力补偿跟踪",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig("assets/03_pd_tracking.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  已保存  assets/03_pd_tracking.png")


if __name__ == "__main__":
    os.makedirs("assets", exist_ok=True)
    print("示例 3: 动力学 (Dynamics)")
    demo_inverse_dynamics()
    demo_free_fall()
    demo_pd_tracking()
    print("运行完成。")
