import math
import numpy as np

class SecondOrderButterworth:
    def __init__(self, cutoff_hz: float):
        self.fc      = cutoff_hz          
        self.x_hist  = [None, None]       
        self.y_hist  = [None, None]       
        self.b, self.a = None, None      

    def _compute_coeffs(self, dt):
        w0 = 2.0 * math.pi * self.fc
        k  = w0 * dt
        k2 = k * k

        # Butterworth ζ = √2/2
        a0 = k2 + math.sqrt(2) * k + 1
        self.b = np.array([k2, 2*k2, k2]) / a0
        self.a = np.array([1.0,
                           2.0*(k2 - 1)/a0,
                           (k2 - math.sqrt(2)*k + 1)/a0])

    def reset(self, x0: np.ndarray):
        self.x_hist = [np.array(x0, copy=True), np.array(x0, copy=True)]
        self.y_hist = [np.array(x0, copy=True), np.array(x0, copy=True)]

    def __call__(self, x: np.ndarray, dt: float) -> np.ndarray:
        if self.x_hist[0] is None:
            self.reset(x)
            return x
        if self.b is None:
            self._compute_coeffs(dt)
        y = ( self.b[0] * x
            + self.b[1] * self.x_hist[0]
            + self.b[2] * self.x_hist[1]
            - self.a[1] * self.y_hist[0]
            - self.a[2] * self.y_hist[1] )
        self.x_hist[1] = self.x_hist[0]
        self.x_hist[0] = x
        self.y_hist[1] = self.y_hist[0]
        self.y_hist[0] = y
        return y