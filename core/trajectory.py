"""
Trajectory planning module.

Provides trapezoidal and S-curve velocity profiles for both
joint-space and Cartesian-space motion planning.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict


@dataclass
class TrajectoryResult:
    """Container for a planned trajectory."""
    t: np.ndarray       # (N,) time stamps
    pos: np.ndarray     # (N,) or (N,3) position
    vel: np.ndarray     # (N,) or (N,3) velocity
    acc: np.ndarray     # (N,) or (N,3) acceleration
    jerk: np.ndarray = None  # (N,) or (N,3) jerk (S-curve only)


# ======================================================================
# Trapezoidal velocity profile
# ======================================================================
class TrapezoidPlanner:
    """
    Trapezoidal (bang-coast-bang) velocity profile.

    Guarantees continuous position and velocity; acceleration is
    piecewise constant with three phases: accelerate, cruise, decelerate.
    """

    def __init__(self, v_max: float, a_max: float):
        self.v_max = abs(v_max)
        self.a_max = abs(a_max)

    def plan(self, start: float, goal: float, n_pts: int = 1000) -> TrajectoryResult:
        """
        Plan a 1-D trapezoidal profile from *start* to *goal*.
        """
        delta = goal - start
        sign = np.sign(delta)
        d = abs(delta)

        if d < 1e-12:
            t = np.zeros(n_pts)
            z = np.full(n_pts, start)
            return TrajectoryResult(t, z, np.zeros(n_pts), np.zeros(n_pts))

        # timing
        t_acc = self.v_max / self.a_max
        d_acc = 0.5 * self.a_max * t_acc ** 2

        if 2 * d_acc >= d:
            # triangle profile (never reaches v_max)
            t_acc = np.sqrt(d / self.a_max)
            t_cruise = 0.0
            v_peak = self.a_max * t_acc
        else:
            t_cruise = (d - 2 * d_acc) / self.v_max
            v_peak = self.v_max

        T = 2 * t_acc + t_cruise
        t = np.linspace(0, T, n_pts)

        pos = np.empty(n_pts)
        vel = np.empty(n_pts)
        acc = np.empty(n_pts)

        for i, ti in enumerate(t):
            if ti <= t_acc:
                # accelerate
                acc[i] = sign * self.a_max
                vel[i] = sign * self.a_max * ti
                pos[i] = start + sign * 0.5 * self.a_max * ti ** 2
            elif ti <= t_acc + t_cruise:
                # cruise
                dt = ti - t_acc
                acc[i] = 0.0
                vel[i] = sign * v_peak
                pos[i] = start + sign * (d_acc + v_peak * dt) if 2 * d_acc < d \
                    else start + sign * (0.5 * self.a_max * t_acc**2 + v_peak * dt)
            else:
                # decelerate
                dt = ti - t_acc - t_cruise
                acc[i] = -sign * self.a_max
                vel[i] = sign * (v_peak - self.a_max * dt)
                pos[i] = goal - sign * 0.5 * self.a_max * (T - ti) ** 2

        return TrajectoryResult(t, pos, vel, acc)


# ======================================================================
# S-curve velocity profile
# ======================================================================
class SCurvePlanner:
    """
    Seven-segment S-curve velocity profile.

    Provides continuous jerk, yielding smooth acceleration transitions
    that reduce mechanical vibration.
    """

    def __init__(self, v_max: float, a_max: float, j_max: float):
        self.v_max = abs(v_max)
        self.a_max = abs(a_max)
        self.j_max = abs(j_max)

    def plan(self, start: float, goal: float, n_pts: int = 2000) -> TrajectoryResult:
        """
        Plan a 1-D S-curve profile from *start* to *goal*.
        """
        delta = goal - start
        sign = np.sign(delta)
        d = abs(delta)

        if d < 1e-12:
            t = np.zeros(n_pts)
            z = np.full(n_pts, start)
            return TrajectoryResult(t, z, np.zeros(n_pts), np.zeros(n_pts), np.zeros(n_pts))

        # Phase durations --------------------------------------------------
        # t_j = jerk phase, t_a = constant-accel phase, t_v = cruise phase
        t_j = self.a_max / self.j_max  # jerk ramp time

        # check if we can reach v_max
        v_acc = self.a_max * t_j  # speed gained during full accel phase (j+const+j)
        # full accel phase = 2*t_j for jerk ramps + t_a for const accel
        # first assume t_a exists
        t_a = (self.v_max - self.a_max * t_j) / self.a_max
        if t_a < 0:
            # can't reach v_max even with full accel
            t_a = 0
            t_j = np.sqrt(self.v_max / self.j_max)

        # distance covered in accel & decel (symmetric)
        d_accel = self.j_max * t_j**2 * t_j / 6 + \
                  0.5 * self.a_max * t_a * (t_a + 2 * t_j) + \
                  self.j_max * t_j**2 / 2 * t_a  # simplified
        # use numerical integration for reliability
        d_accel = self._accel_distance(t_j, t_a)
        d_decel = d_accel  # symmetric

        if 2 * d_accel >= d:
            # no cruise phase; scale down
            ratio = np.sqrt(d / (2 * d_accel)) if d_accel > 0 else 0
            t_j *= ratio
            t_a *= ratio
            d_accel = d / 2
            t_v = 0.0
        else:
            v_peak = self._peak_vel(t_j, t_a)
            t_v = (d - 2 * d_accel) / v_peak if v_peak > 0 else 0.0

        # build 7 segment durations
        segs = [t_j, t_a, t_j, t_v, t_j, t_a, t_j]
        T = sum(segs)
        t_bounds = np.cumsum([0] + segs)

        t = np.linspace(0, T, n_pts)
        pos = np.empty(n_pts)
        vel = np.empty(n_pts)
        acc = np.empty(n_pts)
        jerk = np.empty(n_pts)

        # state integration
        s = 0.0; v = 0.0; a = 0.0
        jerk_signs = [1, 0, -1, 0, -1, 0, 1]  # jerk direction per segment

        prev_idx = 0
        for seg_i in range(7):
            j_val = sign * jerk_signs[seg_i] * self.j_max
            seg_mask = (t >= t_bounds[seg_i]) & (t < t_bounds[seg_i + 1])
            if seg_i == 6:
                seg_mask = (t >= t_bounds[seg_i]) & (t <= t_bounds[seg_i + 1])
            indices = np.where(seg_mask)[0]

            for idx in indices:
                dt = t[idx] - (t[indices[0]] if len(indices) > 0 else t_bounds[seg_i])
                if idx == indices[0]:
                    # store state at segment start
                    s0, v0, a0 = s, v, a

                dt = t[idx] - t_bounds[seg_i]
                jerk[idx] = j_val
                acc[idx] = a0 + j_val * dt
                vel[idx] = v0 + a0 * dt + 0.5 * j_val * dt ** 2
                pos[idx] = start + sign * abs(
                    s0 + v0 * dt + 0.5 * a0 * dt**2 + (1 / 6) * j_val * dt**3
                ) * sign  # keep sign consistent

            # advance state to end of segment
            dt_seg = segs[seg_i]
            a = a0 + j_val * dt_seg
            v = v0 + a0 * dt_seg + 0.5 * j_val * dt_seg ** 2
            s = s0 + v0 * dt_seg + 0.5 * a0 * dt_seg**2 + (1 / 6) * j_val * dt_seg**3

        # fix last point
        pos[-1] = goal
        vel[-1] = 0
        acc[-1] = 0
        jerk[-1] = 0

        return TrajectoryResult(t, pos, vel, acc, jerk)

    def _accel_distance(self, t_j, t_a):
        """Distance covered during the 3-segment acceleration phase."""
        j = self.j_max
        a = self.a_max if t_a > 0 else j * t_j

        # segment 1: jerk ramp up
        d1 = (1 / 6) * j * t_j ** 3
        v1 = 0.5 * j * t_j ** 2
        a1 = j * t_j

        # segment 2: constant accel
        d2 = v1 * t_a + 0.5 * a1 * t_a ** 2
        v2 = v1 + a1 * t_a

        # segment 3: jerk ramp down
        d3 = v2 * t_j + 0.5 * a1 * t_j ** 2 - (1 / 6) * j * t_j ** 3

        return d1 + d2 + d3

    def _peak_vel(self, t_j, t_a):
        """Velocity reached at the end of the acceleration phase."""
        j = self.j_max
        v1 = 0.5 * j * t_j ** 2
        a1 = j * t_j
        v2 = v1 + a1 * t_a
        v3 = v2 + a1 * t_j - 0.5 * j * t_j ** 2
        return v3


# ======================================================================
# Cartesian straight-line planner
# ======================================================================
def plan_cartesian_line(
    start: np.ndarray,
    goal: np.ndarray,
    planner,
    n_pts: int = 1000,
) -> Dict[str, np.ndarray]:
    """
    Plan a straight-line Cartesian trajectory using a 1-D profile.

    Parameters
    ----------
    start, goal : (3,) arrays.
    planner     : TrapezoidPlanner or SCurvePlanner instance.
    n_pts       : number of trajectory points.

    Returns
    -------
    dict with 't', 'pos' (N,3), 'vel' (N,3), 'acc' (N,3).
    """
    delta = goal - start
    dist = np.linalg.norm(delta)
    if dist < 1e-12:
        t = np.zeros(n_pts)
        return {
            "t": t,
            "pos": np.tile(start, (n_pts, 1)),
            "vel": np.zeros((n_pts, 3)),
            "acc": np.zeros((n_pts, 3)),
        }

    direction = delta / dist
    res = planner.plan(0.0, dist, n_pts)

    return {
        "t": res.t,
        "pos": start[None, :] + np.outer(res.pos, direction),
        "vel": np.outer(res.vel, direction),
        "acc": np.outer(res.acc, direction),
    }
