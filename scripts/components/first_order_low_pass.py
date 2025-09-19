import numpy as np
import math

class FirstOrderLowPass:
    def __init__(self, cutoff_hz: float):
        self.tau = 1.0 / (2.0 * math.pi * cutoff_hz)   
        self.y   = None                                

    def reset(self, x0: np.ndarray):
        self.y = np.array(x0, copy=True)

    def __call__(self, x: np.ndarray, dt: float) -> np.ndarray:
        if self.y is None:
            self.reset(x)              
            return self.y
        alpha = dt / (self.tau + dt)   
        self.y = alpha * x + (1.0 - alpha) * self.y
        return self.y