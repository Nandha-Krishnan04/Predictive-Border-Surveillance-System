"""
Constant Velocity Kalman Filter (CV-KF)
========================================
State: [x, y, vx, vy]^T
Measurement: [x, y]^T

Used to track and project 2D position under an assumed constant velocity model.
"""

from typing import List, Tuple
import numpy as np


class KalmanFilterCV:
    """
    2D Constant Velocity Kalman Filter.
    """

    def __init__(
        self,
        initial_pos: Tuple[float, float],
        dt: float = 1.0,
        process_noise_std: float = 1.5,
        measurement_noise_std: float = 2.0,
    ):
        """
        Initialize filter state and covariance matrices.

        Args:
            initial_pos: Initial (x, y) coordinates.
            dt: Time step (typically 1 frame).
            process_noise_std: Standard deviation of acceleration process noise.
            measurement_noise_std: Standard deviation of position measurement noise.
        """
        self.dt = dt

        # State vector: [x, y, vx, vy]
        self.x = np.array([initial_pos[0], initial_pos[1], 0.0, 0.0], dtype=np.float32)

        # State transition matrix F
        self.F = np.array([
            [1.0, 0.0, dt,  0.0],
            [0.0, 1.0, 0.0, dt ],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float32)

        # Measurement matrix H (we measure x and y)
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ], dtype=np.float32)

        # Measurement noise covariance R
        r_val = measurement_noise_std ** 2
        self.R = np.array([
            [r_val, 0.0  ],
            [0.0,   r_val],
        ], dtype=np.float32)

        # Process noise covariance Q (discrete white noise model)
        q = process_noise_std ** 2
        dt2 = (dt ** 2) / 2.0
        dt3 = (dt ** 3) / 3.0
        self.Q = np.array([
            [dt3 * q, 0.0,     dt2 * q, 0.0    ],
            [0.0,     dt3 * q, 0.0,     dt2 * q],
            [dt2 * q, 0.0,     dt * q,  0.0    ],
            [0.0,     dt2 * q, 0.0,     dt * q ],
        ], dtype=np.float32)

        # State covariance P
        self.P = np.eye(4, dtype=np.float32) * 50.0

    def predict(self) -> Tuple[float, float]:
        """
        Execute the Kalman time-update (prediction step).

        Returns:
            Predicted (x, y) coordinates for current step.
        """
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return float(self.x[0]), float(self.x[1])

    def update(self, pos: Tuple[float, float]):
        """
        Execute the Kalman measurement-update step with observed (x, y).

        Args:
            pos: Measured (x, y) coordinates.
        """
        z = np.array([pos[0], pos[1]], dtype=np.float32)
        y = z - (self.H @ self.x)  # Innovation
        S = self.H @ self.P @ self.H.T + self.R  # Innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain

        self.x = self.x + K @ y
        I = np.eye(4, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P

    def predict_future(self, steps: int = 10) -> List[Tuple[float, float]]:
        """
        Extrapolate future positions without altering filter state.

        Args:
            steps: Number of future time steps to project.

        Returns:
            List of (x, y) coordinate tuples.
        """
        future_points = []
        x_proj = self.x.copy()
        for _ in range(steps):
            x_proj = self.F @ x_proj
            future_points.append((float(x_proj[0]), float(x_proj[1])))
        return future_points
