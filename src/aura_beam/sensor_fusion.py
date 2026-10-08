"""A fixed-noise constant-velocity Kalman filter for image coordinates."""
from __future__ import annotations

import numpy as np


class KalmanFilter2D:
    """Track image position with state [x, y, vx, vy] and fixed noise."""

    def __init__(
        self,
        dt: float = 1.0,
        *,
        process_variance: float = 1.0,
        measurement_variance: float = 1.0,
        initial_variance: float = 1.0,
    ) -> None:
        for name, value in (
            ("dt", dt), ("process_variance", process_variance),
            ("measurement_variance", measurement_variance),
            ("initial_variance", initial_variance),
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        self.dt = float(dt)
        self.x = np.zeros((4, 1), dtype=np.float64)
        self.F = np.array([
            [1, 0, self.dt, 0], [0, 1, 0, self.dt],
            [0, 0, 1, 0], [0, 0, 0, 1],
        ], dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64) * initial_variance
        self.Q = np.eye(4, dtype=np.float64) * process_variance
        self.R = np.eye(2, dtype=np.float64) * measurement_variance
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
        self.identity = np.eye(4, dtype=np.float64)

    def initialize(self, x: float, y: float) -> None:
        """Set the initial image position; retain the current velocity."""
        if not np.isfinite(x) or not np.isfinite(y):
            raise ValueError("position must be finite")
        self.x[0, 0] = float(x)
        self.x[1, 0] = float(y)

    def predict(self) -> tuple[float, float]:
        """Advance the constant-velocity state by one time step."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.get_state()

    def update(
        self,
        meas_x: float | None,
        meas_y: float | None,
        predict_first: bool = True,
    ) -> tuple[float, float]:
        """Apply a complete image-position observation, or predict when missing."""
        if meas_x is not None and meas_y is not None:
            if not np.isfinite(meas_x) or not np.isfinite(meas_y):
                raise ValueError("measurement must be finite")
        if predict_first:
            self.predict()
        if meas_x is None or meas_y is None:
            return self.get_state()
        measurement = np.array([[meas_x], [meas_y]], dtype=np.float64)
        innovation = measurement - self.H @ self.x
        innovation_covariance = self.H @ self.P @ self.H.T + self.R
        kalman_gain = self.P @ self.H.T @ np.linalg.inv(innovation_covariance)
        self.x = self.x + kalman_gain @ innovation
        self.P = (self.identity - kalman_gain @ self.H) @ self.P
        return self.get_state()

    def get_state(self) -> tuple[float, float]:
        """Return the current image position."""
        return float(self.x[0, 0]), float(self.x[1, 0])
