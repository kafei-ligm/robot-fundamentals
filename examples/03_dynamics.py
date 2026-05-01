"""
Example 3 — Lagrangian Dynamics & Simulation
=============================================
Demonstrates:
  • Mass matrix, Coriolis, and gravity computation
  • Inverse dynamics  (motion → torques)
  • Forward dynamics  (torques → motion)
  • Free-fall simulation under gravity
  • PD-controlled joint tracking

Run:
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
    """Compute and display M, C, g at a sample configuration."""
    dyn = make_2r_dynamics(L1, L2)

    q = np.radians([45, 30])
    dq = np.array([0.5, -0.3])
    ddq = np.array([0.1, 0.2])

    tau, terms = dyn.inverse_dynamics(q, dq, ddq)

    print("\n  Configuration:  q = (45°, 30°),  dq = (0.5, -0.3) rad/s")
    print(f"  Mass matrix M:\n{terms['M']}")
    print(f"  Coriolis C:\n{terms['C']}")
    print(f"  Gravity g: {terms['g']}")
    print(f"  Required torque: τ = {tau}")

    # verify with forward dynamics
    ddq_check, _ = dyn.forward_dynamics(q, dq, tau)
    print(f"  Forward-dynamics check: ddq = {ddq_check}  (should ≈ {ddq})")
    print(f"  Error norm: {np.linalg.norm(ddq_check - ddq):.2e}")


def demo_free_fall():
    """Simulate the arm falling under gravity from a horizontal pose."""
    dyn = make_2r_dynamics(L1, L2)
    q0 = np.array([0.0, 0.0])   # horizontal
    dq0 = np.zeros(2)

    sim = dyn.simulate(q0, dq0, tau_fn=lambda t, q, dq: np.zeros(2),
                       dt=5e-4, duration=2.0)

    # trajectory plot
    fig = plot_dynamics_sim(sim, title="Free Fall Under Gravity (no torque)")
    fig.savefig("assets/03_freefall.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved  assets/03_freefall.png")

    # snapshots
    fig, ax = plt.subplots(figsize=(7, 5))
    n_frames = 8
    indices = np.linspace(0, len(sim["t"]) - 1, n_frames, dtype=int)
    for k, idx in enumerate(indices):
        alpha = 0.2 + 0.8 * k / (n_frames - 1)
        draw_2r_arm(ax, sim["q"][idx], L1, L2, alpha=alpha,
                    label=f"t={sim['t'][idx]:.2f}s")
    ax.set_aspect("equal")
    ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.5, 1.5)
    _style_axis(ax, "Arm Snapshots During Free Fall", "x (m)", "y (m)")
    fig.tight_layout()
    fig.savefig("assets/03_freefall_snapshots.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved  assets/03_freefall_snapshots.png")


def demo_pd_tracking():
    """PD controller tracking a sinusoidal reference trajectory."""
    dyn = make_2r_dynamics(L1, L2)

    # desired trajectory
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
        # computed-torque-like: PD + gravity compensation
        g = dyn.gravity_vector(q)
        return Kp @ e + Kd @ de + g

    q0 = np.zeros(2)
    dq0 = np.zeros(2)

    sim = dyn.simulate(q0, dq0, tau_fn=pd_controller,
                       dt=5e-4, duration=4.0)

    # compute desired for comparison
    t_arr = sim["t"]
    q_ref = np.array([q_des(t) for t in t_arr])

    fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    for j in range(2):
        c = COLORS[f"tau{j+1}"]
        axes[0].plot(t_arr, np.degrees(sim["q"][:, j]),
                     color=c, linewidth=1.3, label=f"q{j+1} actual")
        axes[0].plot(t_arr, np.degrees(q_ref[:, j]),
                     "--", color=c, linewidth=1, alpha=0.6, label=f"q{j+1} desired")
        axes[1].plot(t_arr, sim["tau"][:, j],
                     color=c, linewidth=1.3, label=f"τ{j+1}")

    _style_axis(axes[0], "PD Tracking — Joint Angles", ylabel="deg")
    _style_axis(axes[1], "Control Torques", xlabel="time (s)", ylabel="N·m")
    axes[0].legend(fontsize=7, ncol=2, loc="upper right")
    axes[1].legend(fontsize=8, loc="upper right")

    fig.suptitle("PD + Gravity Compensation Tracking",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig("assets/03_pd_tracking.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved  assets/03_pd_tracking.png")


if __name__ == "__main__":
    os.makedirs("assets", exist_ok=True)
    print("Example 3: Dynamics")
    demo_inverse_dynamics()
    demo_free_fall()
    demo_pd_tracking()
    print("Done.")
