"""
Constant Acceleration Kalman Filter (CA-KF)
============================================
State: [x, y, vx, vy, ax, ay]^T
Measurement: [x, y]^T

Used to track and project 2D position under an assumed constant acceleration model.
"""

from typing import List, Tuple
import numpy as np


class KalmanFilterCA:
    """
    2D Constant Acceleration Kalman Filter.
    """

    def __init__(
        self,
        initial_pos: Tuple[float, float],
        dt: float = 1.0,
        process_noise_std: float = 0.8,
        measurement_noise_std: float = 2.0,
    ):
        """
        Initialize filter state and covariance matrices.

        Args:
            initial_pos: Initial (x, y) coordinates.
            dt: Time step (typically 1 frame).
            process_noise_std: Standard deviation of jerk process noise.
            measurement_noise_std: Standard deviation of position measurement noise.
        """
        self.dt = dt

        # State vector: [x, y, vx, vy, ax, ay]
        self.x = np.array([
            initial_pos[0], initial_pos[1], 0.0, 0.0, 0.0, 0.0
        ], dtype=np.float32)

        dt2 = 0.5 * (dt ** 2)

        # State transition matrix F
        self.F = np.array([
            [1.0, 0.0, dt,  0.0, dt2, 0.0],
            [0.0, 1.0, 0.0, dt,  0.0, dt2],
            [0.0, 0.0, 1.0, 0.0, dt,  0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, dt ],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float32)

        # Measurement matrix H
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        ], dtype=np.float32)

        # Measurement noise covariance R
        r_val = measurement_noise_std ** 2
        self.R = np.array([
            [r_val, 0.0  ],
            [0.0,   r_val],
        ], dtype=np.float32)

        # Process noise covariance Q (continuous jerk perturbation model)
        q = process_noise_std ** 2
        dt3 = (dt ** 3) / 6.0
        dt4 = (dt ** 4) / 4.0
        dt5 = (dt ** 5) / 20.0

        # Sub-blocks for 1D: [dt5, dt4, dt3; dt4, dt3*2, dt2; dt3, dt2, dt]
        q_1d = np.array([
            [dt5, (dt ** 4) / 8.0, dt3],
            [(dt ** 4) / 8.0, (dt ** 3) / 3.0, dt2],
            [dt3, dt2, dt],
        ], dtype=np.float32) * q

        # Combine for 2D (x and y independent)
        self.Q = np.zeros((6, 6), dtype=np.float32)
        idx_x = [0, 2, 4]
        idx_y = [1, 3, 5]
        for i in range(3):
            for j in range(3):
                self.Q[idx_x[i], idx_x[j]] = q_1d[i, j]
                self.Q[idx_y[i], idx_y[j]] = q_1d[i, j]

        # Initial covariance P
        self.P = np.eye(6, dtype=np.float32) * 50.0

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
        I = np.eye(6, dtype=np.float32)
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
