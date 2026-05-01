"""
用于串联机械臂的动力学模块。

实现了拉格朗日（基于能量）和牛顿-欧拉（递归）公式，用于计算标准的机械臂方程：

    M(q) * q_ddot + C(q, q_dot) * q_dot + g(q) = tau

所有矩阵均通过几何雅可比矩阵进行符号构建，因此该代码可推广到由DH参数定义的任意串联链。
"""

import numpy as np
from typing import Tuple, Dict
from .kinematics import SerialKinematics, DHParams


class LinkInertia:
    """单个刚性连杆的惯性参数。"""

    def __init__(
        self,
        mass: float,
        com: np.ndarray,
        inertia: np.ndarray,
    ):
        """
        参数
        ----------
        mass    : 连杆质量 (kg)。
        com     : (3,) 连杆自身坐标系下的质心位置。
        inertia : (3, 3) 绕质心的旋转惯量张量，在连杆坐标系下表示。
        """
        self.mass = mass
        self.com = np.asarray(com, dtype=float)
        self.inertia = np.asarray(inertia, dtype=float)


class ManipulatorDynamics:
    """
    使用复合刚体算法（拉格朗日公式）计算串联机械臂的M、C、g矩阵。

    参数
    ----------
    kinematics : SerialKinematics 实例 (提供DH参数和正向运动学)。
    link_props : 包含 LinkInertia 的列表，每个关节/连杆对应一个。
    gravity    : (3,) 基坐标系下的重力向量，默认为 [0, -9.81, 0]。
    """

    def __init__(
        self,
        kinematics: SerialKinematics,
        link_props: list,
        gravity: np.ndarray = np.array([0, -9.81, 0]),
    ):
        self.kin = kinematics
        self.links = link_props
        self.n = kinematics.n_joints
        self.gravity = np.asarray(gravity, dtype=float)
        assert len(link_props) == self.n

    # ------------------------------------------------------------------
    # 辅助函数：每个连杆的雅可比矩阵（用于质心的线速度和角速度）
    # ------------------------------------------------------------------
    def _com_jacobians(
        self, q: np.ndarray
    ) -> list:
        """
        返回每个连杆质心的 (Jv_i, Jw_i) 列表。
        每个 Jv_i 维度为 (3, n)，Jw_i 维度为 (3, n)。
        """
        frames = self.kin.fk_all(q)
        result = []

        for i in range(self.n):
            # 基坐标系下的质心位置
            R_i = frames[i + 1][:3, :3]
            p_i = frames[i + 1][:3, 3]
            p_com = p_i + R_i @ self.links[i].com  # 近似值（转换到连杆坐标系）

            Jv = np.zeros((3, self.n))
            Jw = np.zeros((3, self.n))

            for j in range(i + 1):
                z_j = frames[j][:3, 2]
                p_j = frames[j][:3, 3]
                Jv[:, j] = np.cross(z_j, p_com - p_j)
                Jw[:, j] = z_j

            result.append((Jv, Jw))
        return result

    # ------------------------------------------------------------------
    # 质量（惯性）矩阵   M(q)
    # ------------------------------------------------------------------
    def mass_matrix(self, q: np.ndarray) -> np.ndarray:
        """
        计算 n x n 的关节空间惯性矩阵 M(q)。

        M = sum_i  m_i * Jv_i^T Jv_i  +  Jw_i^T  R_i I_i R_i^T  Jw_i
        """
        frames = self.kin.fk_all(q)
        jacs = self._com_jacobians(q)

        M = np.zeros((self.n, self.n))
        for i, (Jv, Jw) in enumerate(jacs):
            m = self.links[i].mass
            R_i = frames[i + 1][:3, :3]
            I_base = R_i @ self.links[i].inertia @ R_i.T  # 将惯量张量转换到基坐标系

            M += m * (Jv.T @ Jv) + Jw.T @ I_base @ Jw
        return M

    # ------------------------------------------------------------------
    # 科里奥利/离心力矩阵   C(q, qdot)   （通过克里斯托费尔符号）
    # ------------------------------------------------------------------
    def coriolis_matrix(
        self, q: np.ndarray, dq: np.ndarray, delta: float = 1e-7
    ) -> np.ndarray:
        """
        科里奥利矩阵 C(q, qdot)，使得与速度相关的力为 C(q, qdot) * qdot。

        通过第一类克里斯托费尔符号计算：
            c_{ijk} = 0.5 * (dM_{ij}/dq_k + dM_{ik}/dq_j - dM_{jk}/dq_i)
            C_{ij}  = sum_k  c_{ijk} * dq_k
        """
        n = self.n
        M0 = self.mass_matrix(q)

        # 通过中心差分法求偏导数  dM/dq_k
        dM = np.zeros((n, n, n))  # dM[i,j,k] = dM_{ij}/dq_k
        for k in range(n):
            q_plus = q.copy();  q_plus[k] += delta
            q_minus = q.copy(); q_minus[k] -= delta
            dM[:, :, k] = (self.mass_matrix(q_plus) - self.mass_matrix(q_minus)) / (2 * delta)

        C = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    c_ijk = 0.5 * (dM[i, j, k] + dM[i, k, j] - dM[j, k, i])
                    C[i, j] += c_ijk * dq[k]
        return C

    # ------------------------------------------------------------------
    # 重力向量   g(q)
    # ------------------------------------------------------------------
    def gravity_vector(self, q: np.ndarray) -> np.ndarray:
        """
        重力矩向量 g(q)。

        g_j = - sum_i  m_i * g^T * Jv_i[:, j]
        """
        jacs = self._com_jacobians(q)
        g = np.zeros(self.n)
        for i, (Jv, _) in enumerate(jacs):
            m = self.links[i].mass
            g -= m * (self.gravity @ Jv)
        return g

    # ------------------------------------------------------------------
    # 逆向动力学:  tau = M q_ddot + C qdot + g
    # ------------------------------------------------------------------
    def inverse_dynamics(
        self, q: np.ndarray, dq: np.ndarray, ddq: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        给定运动状态 (q, dq, ddq)，计算所需的关节扭矩。

        返回
        -------
        tau   : (n,) 关节扭矩。
        terms : 包含键 'M', 'C', 'g' 的字典，便于检查中间项。
        """
        M = self.mass_matrix(q)
        C = self.coriolis_matrix(q, dq)
        g = self.gravity_vector(q)
        tau = M @ ddq + C @ dq + g
        return tau, {"M": M, "C": C, "g": g}

    # ------------------------------------------------------------------
    # 正向动力学:  q_ddot = M^{-1} (tau - C qdot - g)
    # ------------------------------------------------------------------
    def forward_dynamics(
        self, q: np.ndarray, dq: np.ndarray, tau: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        给定扭矩和当前状态，计算关节加速度。

        返回
        -------
        ddq   : (n,) 关节加速度。
        terms : 包含键 'M', 'C', 'g' 的字典。
        """
        M = self.mass_matrix(q)
        C = self.coriolis_matrix(q, dq)
        g = self.gravity_vector(q)
        ddq = np.linalg.solve(M, tau - C @ dq - g)
        return ddq, {"M": M, "C": C, "g": g}

    # ------------------------------------------------------------------
    # 仿真 (欧拉积分)
    # ------------------------------------------------------------------
    def simulate(
        self,
        q0: np.ndarray,
        dq0: np.ndarray,
        tau_fn,
        dt: float = 1e-3,
        duration: float = 2.0,
    ) -> Dict[str, np.ndarray]:
        """
        使用给定的扭矩策略模拟正向动力学。

        参数
        ----------
        q0      : 初始关节角度。
        dq0     : 初始关节速度。
        tau_fn  : 可调用对象(t, q, dq) -> 返回 tau 数组。
        dt      : 时间步长 (s)。
        duration: 总时间 (s)。

        返回
        -------
        包含键 't', 'q', 'dq', 'ddq', 'tau' 的字典。
        """
        steps = int(duration / dt)
        t_arr = np.zeros(steps)
        q_arr = np.zeros((steps, self.n))
        dq_arr = np.zeros((steps, self.n))
        ddq_arr = np.zeros((steps, self.n))
        tau_arr = np.zeros((steps, self.n))

        q = q0.copy()
        dq = dq0.copy()

        for k in range(steps):
            t = k * dt
            tau = tau_fn(t, q, dq)
            ddq, _ = self.forward_dynamics(q, dq, tau)

            t_arr[k] = t
            q_arr[k] = q
            dq_arr[k] = dq
            ddq_arr[k] = ddq
            tau_arr[k] = tau

            # 半隐式欧拉法
            dq = dq + ddq * dt
            q = q + dq * dt

        return {"t": t_arr, "q": q_arr, "dq": dq_arr, "ddq": ddq_arr, "tau": tau_arr}


# ======================================================================
# 快捷功能：二自由度 (2-R) 平面机械臂动力学
# ======================================================================
def make_2r_dynamics(
    l1: float = 1.0,
    l2: float = 0.8,
    m1: float = 1.0,
    m2: float = 0.8,
) -> ManipulatorDynamics:
    """
    创建平面2R机械臂的动力学模型。

    假设连杆为均匀细长杆，且质心位于连杆中点。
    """
    from .kinematics import make_2r

    kin = make_2r(l1, l2)
    links = [
        LinkInertia(
            mass=m1,
            com=np.array([l1 / 2, 0, 0]),
            inertia=np.diag([0, 0, m1 * l1**2 / 12]),
        ),
        LinkInertia(
            mass=m2,
            com=np.array([l2 / 2, 0, 0]),
            inertia=np.diag([0, 0, m2 * l2**2 / 12]),
        ),
    ]
    return ManipulatorDynamics(kin, links, gravity=np.array([0, -9.81, 0]))
