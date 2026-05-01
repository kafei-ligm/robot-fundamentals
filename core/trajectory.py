"""
轨迹规划模块。

为关节空间和笛卡尔空间的运动规划提供梯形和 S 型 (S-curve) 速度曲线。
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict


@dataclass
class TrajectoryResult:
    """规划轨迹的容器。"""
    t: np.ndarray            # (N,) 时间戳
    pos: np.ndarray          # (N,) 或 (N,3) 位置
    vel: np.ndarray          # (N,) 或 (N,3) 速度
    acc: np.ndarray          # (N,) 或 (N,3) 加速度
    jerk: np.ndarray = None  # (N,) 或 (N,3) 加加速度/急动度 (仅限S型曲线)


# ======================================================================
# 梯形速度曲线
# ======================================================================
class TrapezoidPlanner:
    """
    梯形（加速-匀速-减速，bang-coast-bang）速度曲线。

    保证位置和速度的连续性；加速度是分段常数，分为三个阶段：加速、匀速（巡航）和减速。
    """

    def __init__(self, v_max: float, a_max: float):
        self.v_max = abs(v_max)
        self.a_max = abs(a_max)

    def plan(self, start: float, goal: float, n_pts: int = 1000) -> TrajectoryResult:
        """
        规划从 *start* 到 *goal* 的一维梯形速度曲线。
        """
        delta = goal - start
        sign = np.sign(delta)
        d = abs(delta)

        if d < 1e-12:
            t = np.zeros(n_pts)
            z = np.full(n_pts, start)
            return TrajectoryResult(t, z, np.zeros(n_pts), np.zeros(n_pts))

        # 时间计算
        t_acc = self.v_max / self.a_max
        d_acc = 0.5 * self.a_max * t_acc ** 2

        if 2 * d_acc >= d:
            # 三角形曲线 (无法达到最大速度 v_max)
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
                # 加速段
                acc[i] = sign * self.a_max
                vel[i] = sign * self.a_max * ti
                pos[i] = start + sign * 0.5 * self.a_max * ti ** 2
            elif ti <= t_acc + t_cruise:
                # 匀速段
                dt = ti - t_acc
                acc[i] = 0.0
                vel[i] = sign * v_peak
                pos[i] = start + sign * (d_acc + v_peak * dt) if 2 * d_acc < d \
                    else start + sign * (0.5 * self.a_max * t_acc**2 + v_peak * dt)
            else:
                # 减速段
                dt = ti - t_acc - t_cruise
                acc[i] = -sign * self.a_max
                vel[i] = sign * (v_peak - self.a_max * dt)
                pos[i] = goal - sign * 0.5 * self.a_max * (T - ti) ** 2

        return TrajectoryResult(t, pos, vel, acc)


# ======================================================================
# S 型速度曲线
# ======================================================================
class SCurvePlanner:
    """
    七段式 S 型速度曲线。

    提供连续的加加速度（jerk），产生平滑的加速度过渡，从而减少机械振动。
    """

    def __init__(self, v_max: float, a_max: float, j_max: float):
        self.v_max = abs(v_max)
        self.a_max = abs(a_max)
        self.j_max = abs(j_max)

    def plan(self, start: float, goal: float, n_pts: int = 2000) -> TrajectoryResult:
        """
        规划从 *start* 到 *goal* 的一维 S 型速度曲线。
        """
        delta = goal - start
        sign = np.sign(delta)
        d = abs(delta)

        if d < 1e-12:
            t = np.zeros(n_pts)
            z = np.full(n_pts, start)
            return TrajectoryResult(t, z, np.zeros(n_pts), np.zeros(n_pts), np.zeros(n_pts))

        # 阶段持续时间 --------------------------------------------------
        # t_j = 加加速度段, t_a = 匀加速段, t_v = 匀速段
        t_j = self.a_max / self.j_max  # 加加速度斜坡时间

        # 检查是否能达到 v_max
        v_acc = self.a_max * t_j  # 在完整加速阶段 (j+匀加+j) 获得的速度
        # 完整加速阶段 = 2*t_j (加加速度斜坡) + t_a (匀加速)
        # 首先假设存在 t_a 段
        t_a = (self.v_max - self.a_max * t_j) / self.a_max
        if t_a < 0:
            # 即使全力加速也无法达到 v_max
            t_a = 0
            t_j = np.sqrt(self.v_max / self.j_max)

        # 加速和减速阶段走过的距离 (对称)
        d_accel = self.j_max * t_j**2 * t_j / 6 + \
                  0.5 * self.a_max * t_a * (t_a + 2 * t_j) + \
                  self.j_max * t_j**2 / 2 * t_a  # 简化公式
        # 使用数值积分以确保可靠性
        d_accel = self._accel_distance(t_j, t_a)
        d_decel = d_accel  # 对称

        if 2 * d_accel >= d:
            # 没有匀速段；等比例缩放
            ratio = np.sqrt(d / (2 * d_accel)) if d_accel > 0 else 0
            t_j *= ratio
            t_a *= ratio
            d_accel = d / 2
            t_v = 0.0
        else:
            v_peak = self._peak_vel(t_j, t_a)
            t_v = (d - 2 * d_accel) / v_peak if v_peak > 0 else 0.0

        # 构建 7 个时间段的持续时间
        segs = [t_j, t_a, t_j, t_v, t_j, t_a, t_j]
        T = sum(segs)
        t_bounds = np.cumsum([0] + segs)

        t = np.linspace(0, T, n_pts)
        pos = np.empty(n_pts)
        vel = np.empty(n_pts)
        acc = np.empty(n_pts)
        jerk = np.empty(n_pts)

        # 状态积分
        s = 0.0; v = 0.0; a = 0.0
        jerk_signs = [1, 0, -1, 0, -1, 0, 1]  # 每个时间段的加加速度方向

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
                    # 存储时间段开始时的状态
                    s0, v0, a0 = s, v, a

                dt = t[idx] - t_bounds[seg_i]
                jerk[idx] = j_val
                acc[idx] = a0 + j_val * dt
                vel[idx] = v0 + a0 * dt + 0.5 * j_val * dt ** 2
                pos[idx] = start + sign * abs(
                    s0 + v0 * dt + 0.5 * a0 * dt**2 + (1 / 6) * j_val * dt**3
                ) * sign  # 保持符号一致

            # 将状态推进到时间段结束
            dt_seg = segs[seg_i]
            a = a0 + j_val * dt_seg
            v = v0 + a0 * dt_seg + 0.5 * j_val * dt_seg ** 2
            s = s0 + v0 * dt_seg + 0.5 * a0 * dt_seg**2 + (1 / 6) * j_val * dt_seg**3

        # 修正最后一个点
        pos[-1] = goal
        vel[-1] = 0
        acc[-1] = 0
        jerk[-1] = 0

        return TrajectoryResult(t, pos, vel, acc, jerk)

    def _accel_distance(self, t_j, t_a):
        """在由3段组成的加速阶段中走过的距离。"""
        j = self.j_max
        a = self.a_max if t_a > 0 else j * t_j

        # 第 1 段: 加加速度上升
        d1 = (1 / 6) * j * t_j ** 3
        v1 = 0.5 * j * t_j ** 2
        a1 = j * t_j

        # 第 2 段: 匀加速
        d2 = v1 * t_a + 0.5 * a1 * t_a ** 2
        v2 = v1 + a1 * t_a

        # 第 3 段: 加加速度下降
        d3 = v2 * t_j + 0.5 * a1 * t_j ** 2 - (1 / 6) * j * t_j ** 3

        return d1 + d2 + d3

    def _peak_vel(self, t_j, t_a):
        """在加速阶段结束时达到的速度。"""
        j = self.j_max
        v1 = 0.5 * j * t_j ** 2
        a1 = j * t_j
        v2 = v1 + a1 * t_a
        v3 = v2 + a1 * t_j - 0.5 * j * t_j ** 2
        return v3


# ======================================================================
# 笛卡尔空间直线规划器
# ======================================================================
def plan_cartesian_line(
    start: np.ndarray,
    goal: np.ndarray,
    planner,
    n_pts: int = 1000,
) -> Dict[str, np.ndarray]:
    """
    使用一维速度曲线规划笛卡尔空间中的直线轨迹。

    参数
    ----------
    start, goal : (3,) 数组。
    planner     : TrapezoidPlanner 或 SCurvePlanner 实例。
    n_pts       : 轨迹点的数量。

    返回
    -------
    包含 't', 'pos' (N,3), 'vel' (N,3), 'acc' (N,3) 的字典。
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
