"""
用于串联机械臂的运动学模块。

提供使用 DH (Denavit-Hartenberg) 参数法的平面和空间机械臂的
正向/逆向运动学及雅可比矩阵分析。
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class DHParams:
    """单个关节的标准 Denavit-Hartenberg (DH) 参数。"""
    a: float       # 连杆长度 (link length)
    alpha: float   # 连杆扭转角 (link twist)
    d: float       # 连杆偏移量 (link offset)
    theta0: float  # 关节角偏移量 (joint angle offset)

    def transform(self, q: float) -> np.ndarray:
        """
        计算该关节在角度 q 下的 4x4 齐次变换矩阵。

        T_i = Rot_z(theta) * Trans_z(d) * Trans_x(a) * Rot_x(alpha)
        """
        theta = q + self.theta0
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(self.alpha), np.sin(self.alpha)
        return np.array([
            [ct, -st * ca,  st * sa, self.a * ct],
            [st,  ct * ca, -ct * sa, self.a * st],
            [0,        sa,       ca,      self.d],
            [0,         0,        0,           1],
        ])


class SerialKinematics:
    """
    n自由度串联机械臂运动学求解器。

    参数
    ----------
    dh_table : 包含 DHParams 的列表
        从基座到末端执行器的每个关节的 DH 参数。
    """

    def __init__(self, dh_table: List[DHParams]):
        self.dh = dh_table
        self.n_joints = len(dh_table)

    # ------------------------------------------------------------------
    # 正向运动学
    # ------------------------------------------------------------------
    def fk(self, q: np.ndarray) -> np.ndarray:
        """
        正向运动学：关节角度 -> 末端执行器位姿。

        返回
        -------
        T : (4, 4) 从基座到末端执行器的齐次变换矩阵。
        """
        T = np.eye(4)
        for i, dh in enumerate(self.dh):
            T = T @ dh.transform(q[i])
        return T

    def fk_all(self, q: np.ndarray) -> List[np.ndarray]:
        """
        计算每个坐标系的正向运动学（包含基座标系）。

        返回
        -------
        frames : 包含 (4, 4) 变换矩阵的列表，长度 = n_joints + 1。
                 frames[0] = 基座，frames[-1] = 末端执行器。
        """
        frames = [np.eye(4)]
        T = np.eye(4)
        for i, dh in enumerate(self.dh):
            T = T @ dh.transform(q[i])
            frames.append(T.copy())
        return frames

    # ------------------------------------------------------------------
    # 几何雅可比矩阵
    # ------------------------------------------------------------------
    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """
        基坐标系下的几何雅可比矩阵（假设均为旋转关节）。

        返回
        -------
        J : (6, n) 矩阵。 J[:3] = 线速度， J[3:] = 角速度。
        """
        frames = self.fk_all(q)
        p_e = frames[-1][:3, 3]  # 末端执行器位置

        J = np.zeros((6, self.n_joints))
        for i in range(self.n_joints):
            z_i = frames[i][:3, 2]          # 基坐标系下的关节轴
            p_i = frames[i][:3, 3]          # 基坐标系下的关节原点
            J[:3, i] = np.cross(z_i, p_e - p_i)  # 线性部分 (线速度)
            J[3:, i] = z_i                       # 角度部分 (角速度)
        return J

    def manipulability(self, q: np.ndarray) -> float:
        """吉川 (Yoshikawa) 可操作度指数 w = sqrt(det(J * J^T))。"""
        J = self.jacobian(q)
        Jv = J[:3, :]   # 为了平面情况下的逻辑清晰，仅使用线性部分
        return float(np.sqrt(max(np.linalg.det(Jv @ Jv.T), 0)))

    # ------------------------------------------------------------------
    # 逆向运动学 (基于雅可比伪逆矩阵的迭代法)
    # ------------------------------------------------------------------
    def ik(
        self,
        target_pos: np.ndarray,
        q0: Optional[np.ndarray] = None,
        tol: float = 1e-4,
        max_iter: int = 200,
        alpha: float = 0.5,
    ) -> Tuple[np.ndarray, float]:
        """
        使用阻尼最小二乘法的数值逆向运动学（仅限位置控制）。

        参数
        ----------
        target_pos : (3,) 期望的末端执行器位置。
        q0         : 初始猜测值（如果为 None 则全为零）。
        tol        : 收敛容差 (m)。
        max_iter   : 最大迭代次数。
        alpha      : 步长。

        返回
        -------
        q     : 关节角度。
        error : 最终位置误差范数。
        """
        q = q0.copy() if q0 is not None else np.zeros(self.n_joints)
        damping = 1e-3

        for _ in range(max_iter):
            T = self.fk(q)
            e = target_pos - T[:3, 3]
            err = np.linalg.norm(e)
            if err < tol:
                break
            Jv = self.jacobian(q)[:3, :]
            # 阻尼最小二乘法 (Damped least-squares)
            dq = Jv.T @ np.linalg.solve(Jv @ Jv.T + damping * np.eye(3), e)
            q = q + alpha * dq

        return q, float(np.linalg.norm(target_pos - self.fk(q)[:3, 3]))


# ======================================================================
# 快捷功能：二自由度 (2-R) 平面机械臂 (经典教科书模型)
# ======================================================================
def make_2r(l1: float = 1.0, l2: float = 0.8) -> SerialKinematics:
    """创建具有给定连杆长度的平面 2R 机械臂。"""
    dh_table = [
        DHParams(a=l1, alpha=0, d=0, theta0=0),
        DHParams(a=l2, alpha=0, d=0, theta0=0),
    ]
    return SerialKinematics(dh_table)


def ik_2r_analytic(
    l1: float, l2: float, x: float, y: float, elbow_up: bool = True
) -> Optional[Tuple[float, float]]:
    """
    平面 2R 机械臂的解析解（闭式解）逆向运动学。

    返回 (theta1, theta2)，如果目标位置不可达则返回 None。
    """
    r2 = x * x + y * y
    cos_q2 = (r2 - l1**2 - l2**2) / (2 * l1 * l2)
    if abs(cos_q2) > 1:
        return None
    q2 = np.arccos(cos_q2) * (1 if elbow_up else -1)
    q1 = np.arctan2(y, x) - np.arctan2(l2 * np.sin(q2), l1 + l2 * np.cos(q2))
    return float(q1), float(q2)
