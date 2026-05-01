"""
Kinematics module for serial-link manipulators.

Provides forward/inverse kinematics and Jacobian analysis
for planar and spatial manipulators using DH convention.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class DHParams:
    """Standard Denavit-Hartenberg parameters for one joint."""
    a: float       # link length
    alpha: float   # link twist
    d: float       # link offset
    theta0: float  # joint angle offset

    def transform(self, q: float) -> np.ndarray:
        """
        Compute the 4x4 homogeneous transformation matrix
        for this joint at angle q.

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
    Kinematics solver for an n-DOF serial-link manipulator.

    Parameters
    ----------
    dh_table : list of DHParams
        DH parameters for each joint, from base to end-effector.
    """

    def __init__(self, dh_table: List[DHParams]):
        self.dh = dh_table
        self.n_joints = len(dh_table)

    # ------------------------------------------------------------------
    # Forward kinematics
    # ------------------------------------------------------------------
    def fk(self, q: np.ndarray) -> np.ndarray:
        """
        Forward kinematics: joint angles -> end-effector pose.

        Returns
        -------
        T : (4, 4) homogeneous transformation from base to end-effector.
        """
        T = np.eye(4)
        for i, dh in enumerate(self.dh):
            T = T @ dh.transform(q[i])
        return T

    def fk_all(self, q: np.ndarray) -> List[np.ndarray]:
        """
        Forward kinematics for every frame (base included).

        Returns
        -------
        frames : list of (4, 4) transforms, length = n_joints + 1.
                 frames[0] = base, frames[-1] = end-effector.
        """
        frames = [np.eye(4)]
        T = np.eye(4)
        for i, dh in enumerate(self.dh):
            T = T @ dh.transform(q[i])
            frames.append(T.copy())
        return frames

    # ------------------------------------------------------------------
    # Geometric Jacobian
    # ------------------------------------------------------------------
    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """
        Geometric Jacobian in base frame (revolute joints assumed).

        Returns
        -------
        J : (6, n) matrix.  J[:3] = linear velocity,  J[3:] = angular velocity.
        """
        frames = self.fk_all(q)
        p_e = frames[-1][:3, 3]  # end-effector position

        J = np.zeros((6, self.n_joints))
        for i in range(self.n_joints):
            z_i = frames[i][:3, 2]          # joint axis in base frame
            p_i = frames[i][:3, 3]          # joint origin in base frame
            J[:3, i] = np.cross(z_i, p_e - p_i)  # linear part
            J[3:, i] = z_i                        # angular part
        return J

    def manipulability(self, q: np.ndarray) -> float:
        """Yoshikawa manipulability index  w = sqrt(det(J * J^T))."""
        J = self.jacobian(q)
        Jv = J[:3, :]   # use only linear part for planar clarity
        return float(np.sqrt(max(np.linalg.det(Jv @ Jv.T), 0)))

    # ------------------------------------------------------------------
    # Inverse kinematics  (Jacobian pseudo-inverse, iterative)
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
        Numerical IK using damped least-squares (position only).

        Parameters
        ----------
        target_pos : (3,) desired end-effector position.
        q0         : initial guess (zeros if None).
        tol        : convergence tolerance (m).
        max_iter   : iteration cap.
        alpha      : step size.

        Returns
        -------
        q     : joint angles.
        error : final position error norm.
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
            # damped least-squares
            dq = Jv.T @ np.linalg.solve(Jv @ Jv.T + damping * np.eye(3), e)
            q = q + alpha * dq

        return q, float(np.linalg.norm(target_pos - self.fk(q)[:3, 3]))


# ======================================================================
# Convenience: 2-R planar arm (classic textbook model)
# ======================================================================
def make_2r(l1: float = 1.0, l2: float = 0.8) -> SerialKinematics:
    """Create a planar 2R manipulator with given link lengths."""
    dh_table = [
        DHParams(a=l1, alpha=0, d=0, theta0=0),
        DHParams(a=l2, alpha=0, d=0, theta0=0),
    ]
    return SerialKinematics(dh_table)


def ik_2r_analytic(
    l1: float, l2: float, x: float, y: float, elbow_up: bool = True
) -> Optional[Tuple[float, float]]:
    """
    Closed-form IK for a planar 2R arm.

    Returns (theta1, theta2) or None if unreachable.
    """
    r2 = x * x + y * y
    cos_q2 = (r2 - l1**2 - l2**2) / (2 * l1 * l2)
    if abs(cos_q2) > 1:
        return None
    q2 = np.arccos(cos_q2) * (1 if elbow_up else -1)
    q1 = np.arctan2(y, x) - np.arctan2(l2 * np.sin(q2), l1 + l2 * np.cos(q2))
    return float(q1), float(q2)
