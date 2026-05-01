"""
Dynamics module for serial-link manipulators.

Implements both Lagrangian (energy-based) and Newton-Euler (recursive)
formulations for computing the standard manipulator equation:

    M(q) * q_ddot + C(q, q_dot) * q_dot + g(q) = tau

All matrices are built symbolically via the geometric Jacobian so the
code generalises to arbitrary serial chains defined by DH parameters.
"""

import numpy as np
from typing import Tuple, Dict
from .kinematics import SerialKinematics, DHParams


class LinkInertia:
    """Inertial parameters for a single rigid link."""

    def __init__(
        self,
        mass: float,
        com: np.ndarray,
        inertia: np.ndarray,
    ):
        """
        Parameters
        ----------
        mass    : link mass (kg).
        com     : (3,) centre-of-mass in the link's own frame.
        inertia : (3, 3) rotational inertia tensor about the CoM,
                  expressed in the link frame.
        """
        self.mass = mass
        self.com = np.asarray(com, dtype=float)
        self.inertia = np.asarray(inertia, dtype=float)


class ManipulatorDynamics:
    """
    Compute M, C, g for a serial manipulator using the
    composite-rigid-body algorithm (Lagrangian formulation).

    Parameters
    ----------
    kinematics : SerialKinematics instance (provides DH and FK).
    link_props : list of LinkInertia, one per joint/link.
    gravity    : (3,) gravity vector in base frame, default [0, -9.81, 0].
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
    # helpers: per-link Jacobians (linear + angular for CoM)
    # ------------------------------------------------------------------
    def _com_jacobians(
        self, q: np.ndarray
    ) -> list:
        """
        Return list of (Jv_i, Jw_i) for each link's centre of mass.
        Each Jv_i is (3, n), Jw_i is (3, n).
        """
        frames = self.kin.fk_all(q)
        result = []

        for i in range(self.n):
            # CoM position in base frame
            R_i = frames[i + 1][:3, :3]
            p_i = frames[i + 1][:3, 3]
            p_com = p_i + R_i @ self.links[i].com  # approximate (link frame)

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
    # Mass (inertia) matrix   M(q)
    # ------------------------------------------------------------------
    def mass_matrix(self, q: np.ndarray) -> np.ndarray:
        """
        Compute the n x n joint-space inertia matrix M(q).

        M = sum_i  m_i * Jv_i^T Jv_i  +  Jw_i^T  R_i I_i R_i^T  Jw_i
        """
        frames = self.kin.fk_all(q)
        jacs = self._com_jacobians(q)

        M = np.zeros((self.n, self.n))
        for i, (Jv, Jw) in enumerate(jacs):
            m = self.links[i].mass
            R_i = frames[i + 1][:3, :3]
            I_base = R_i @ self.links[i].inertia @ R_i.T  # inertia in base

            M += m * (Jv.T @ Jv) + Jw.T @ I_base @ Jw
        return M

    # ------------------------------------------------------------------
    # Coriolis / centrifugal   C(q, qdot)   via Christoffel symbols
    # ------------------------------------------------------------------
    def coriolis_matrix(
        self, q: np.ndarray, dq: np.ndarray, delta: float = 1e-7
    ) -> np.ndarray:
        """
        Coriolis matrix C(q, qdot) such that the velocity-dependent
        forces are  C(q, qdot) * qdot.

        Computed from Christoffel symbols of the first kind:
            c_{ijk} = 0.5 * (dM_{ij}/dq_k + dM_{ik}/dq_j - dM_{jk}/dq_i)
            C_{ij}  = sum_k  c_{ijk} * dq_k
        """
        n = self.n
        M0 = self.mass_matrix(q)

        # partial derivatives  dM/dq_k  via central differences
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
    # Gravity vector   g(q)
    # ------------------------------------------------------------------
    def gravity_vector(self, q: np.ndarray) -> np.ndarray:
        """
        Gravity torque vector g(q).

        g_j = - sum_i  m_i * g^T * Jv_i[:, j]
        """
        jacs = self._com_jacobians(q)
        g = np.zeros(self.n)
        for i, (Jv, _) in enumerate(jacs):
            m = self.links[i].mass
            g -= m * (self.gravity @ Jv)
        return g

    # ------------------------------------------------------------------
    # Inverse dynamics:  tau = M q_ddot + C qdot + g
    # ------------------------------------------------------------------
    def inverse_dynamics(
        self, q: np.ndarray, dq: np.ndarray, ddq: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Given motion (q, dq, ddq), compute required joint torques.

        Returns
        -------
        tau   : (n,) joint torques.
        terms : dict with keys 'M', 'C', 'g' for inspection.
        """
        M = self.mass_matrix(q)
        C = self.coriolis_matrix(q, dq)
        g = self.gravity_vector(q)
        tau = M @ ddq + C @ dq + g
        return tau, {"M": M, "C": C, "g": g}

    # ------------------------------------------------------------------
    # Forward dynamics:  q_ddot = M^{-1} (tau - C qdot - g)
    # ------------------------------------------------------------------
    def forward_dynamics(
        self, q: np.ndarray, dq: np.ndarray, tau: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Given torques and current state, compute joint accelerations.

        Returns
        -------
        ddq   : (n,) joint accelerations.
        terms : dict with keys 'M', 'C', 'g'.
        """
        M = self.mass_matrix(q)
        C = self.coriolis_matrix(q, dq)
        g = self.gravity_vector(q)
        ddq = np.linalg.solve(M, tau - C @ dq - g)
        return ddq, {"M": M, "C": C, "g": g}

    # ------------------------------------------------------------------
    # Simulation (Euler integration)
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
        Simulate forward dynamics with a given torque policy.

        Parameters
        ----------
        q0      : initial joint angles.
        dq0     : initial joint velocities.
        tau_fn  : callable(t, q, dq) -> tau array.
        dt      : time step (s).
        duration: total time (s).

        Returns
        -------
        dict with keys 't', 'q', 'dq', 'ddq', 'tau'.
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

            # semi-implicit Euler
            dq = dq + ddq * dt
            q = q + dq * dt

        return {"t": t_arr, "q": q_arr, "dq": dq_arr, "ddq": ddq_arr, "tau": tau_arr}


# ======================================================================
# Convenience: 2-R planar arm dynamics
# ======================================================================
def make_2r_dynamics(
    l1: float = 1.0,
    l2: float = 0.8,
    m1: float = 1.0,
    m2: float = 0.8,
) -> ManipulatorDynamics:
    """
    Create dynamics model for a planar 2R arm.

    Assumes uniform slender rods with CoM at mid-link.
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
