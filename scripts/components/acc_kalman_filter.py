import numpy as np

class AccKalmanFilter:
    """
    Constant-acceleration Kalman filter:
      State x = [p (3), v (3), a (3)].
      Measure position only.
    """
    def __init__(self,
                 dt_init: float,        # 1/CTRL_HZ
                 q_var: float = 1e-5,   # process noise (accel)
                 r_var: float = 2e-5    # measurement noise (pos)
                 ):
        # initial settings
        self.dt   = dt_init
        self.q    = q_var
        I3        = np.eye(3)

        # build measurement model
        self.H = np.hstack((I3, np.zeros((3, 6))))  # measure p only
        self.R = r_var * I3

        # allocate matrices
        self.x = np.zeros(9)             # [p, v, a]
        self.P = np.eye(9) * 1e3         # large initial uncertainty
        self.F = np.eye(9)
        self.Q = np.zeros((9, 9))
        self._init = False

    def _build_matrices(self, dt: float):
        I3 = np.eye(3)
        dt2 = dt * dt / 2.0

        # state-transition
        F = np.eye(9)
        F[0:3, 3:6]   = dt * I3
        F[0:3, 6:9]   = dt2 * I3
        F[3:6, 6:9]   = dt * I3
        self.F = F

        # process noise Q
        q = self.q
        Q = np.zeros((9,9))
        Q[0:3, 0:3]   = (dt**5/20) * q * I3
        Q[0:3, 3:6]   = (dt**4/8)  * q * I3
        Q[0:3, 6:9]   = (dt**3/6)  * q * I3
        Q[3:6, 0:3]   = Q[0:3,3:6].T
        Q[3:6, 3:6]   = (dt**3/3)  * q * I3
        Q[3:6, 6:9]   = (dt**2/2)  * q * I3
        Q[6:9, 0:3]   = Q[0:3,6:9].T
        Q[6:9, 3:6]   = Q[3:6,6:9].T
        Q[6:9, 6:9]   = dt * q * I3
        self.Q = Q

    def predict(self, dt: float):
        """Kalman predict step."""
        self.dt = dt
        self._build_matrices(dt)
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z: np.ndarray):
        """Kalman update with position measurement z."""
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x += K @ y
        I9 = np.eye(9)
        self.P = (I9 - K @ self.H) @ self.P

    def step(self, z: np.ndarray, dt: float):
        """
        One full predict+update cycle.
        First call seeds position, velocity and accel to zero.
        """
        if not self._init:
            # seed p, leave v,a at zero
            self.x[0:3] = z
            self._init   = True
            return
        self.predict(dt)
        self.update(z)

    @property
    def pos(self) -> np.ndarray:
        return self.x[0:3]

    @property
    def vel(self) -> np.ndarray:
        return self.x[3:6]

    @property
    def acc(self) -> np.ndarray:
        return self.x[6:9]