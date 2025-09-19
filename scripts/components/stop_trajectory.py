import numpy as np
from typing import Tuple

class StopTrajPlanner:
    """
    Quintic stop trajectory.
    Matches p(0)=x0, v(0)=v0, a(0)=a0; p(T)=xf, v(T)=0, a(T)=0.
    v0,a0 default to 0 for backward compatibility.
    """
    def __init__(self,
                 x0: np.ndarray,
                 xf: np.ndarray,
                 T_b: float,
                 v0: np.ndarray | None = None,
                 a0: np.ndarray | None = None):
        self.dim = x0.size
        self.T   = float(T_b)
        t        = self.T
        self.xf  = np.array(xf, dtype=float).copy()

        # default initial vel/acc = 0 (old behavior)
        if v0 is None: v0 = np.zeros_like(x0)
        if a0 is None: a0 = np.zeros_like(x0)
        v0 = np.asarray(v0, dtype=float).reshape(self.dim)
        a0 = np.asarray(a0, dtype=float).reshape(self.dim)

        # 6 boundary conditions -> 6x6
        A = np.array([
            [1,    0,     0,      0,       0,        0],   # p(0)
            [0,    1,     0,      0,       0,        0],   # v(0)
            [0,    0,     2,      0,       0,        0],   # a(0)
            [1,    t,   t**2,   t**3,    t**4,     t**5],  # p(T)
            [0,    1,   2*t,   3*t**2,  4*t**3,   5*t**4], # v(T)=0
            [0,    0,     2,    6*t,   12*t**2,  20*t**3], # a(T)=0
        ], dtype=float)

        self.coeff = np.empty((self.dim, 6), dtype=float)
        for k in range(self.dim):
            rhs = np.array([x0[k], v0[k], a0[k], xf[k], 0.0, 0.0], dtype=float)
            self.coeff[k] = np.linalg.solve(A, rhs)

    def polyval(self, tau: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (x, x_dot, x_ddot) for tau ∈ [0,T]."""
        t = np.clip(tau, 0.0, self.T)  # clamp
        P   = np.array([1, t, t**2, t**3, t**4, t**5], dtype=float)
        dP  = np.array([0, 1, 2*t, 3*t**2, 4*t**3, 5*t**4], dtype=float)
        d2P = np.array([0, 0, 2,   6*t,  12*t**2, 20*t**3], dtype=float)
        x     = self.coeff @ P
        x_dot = self.coeff @ dP
        x_dd  = self.coeff @ d2P
        return x, x_dot, x_dd