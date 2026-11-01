"""
Motion Predictor Module
========================
Coordinates Constant Velocity (CV) and Constant Acceleration (CA) Kalman Filters
for tracked people.

Key evaluation requirement:
At frame t, each model predicts position for frame t+1.
When actual tracked position at frame t+1 is observed, calculates Euclidean error.
Maintains a rolling error window to dynamically select the better model for each person.
Extrapolates future trajectory for restricted zone intrusion anticipation.
"""

import math
import logging
from collections import deque
from typing import Dict, List, Tuple, Any
import cv2
import numpy as np

import config
from prediction.kalman_cv import KalmanFilterCV
from prediction.kalman_ca import KalmanFilterCA

logger = logging.getLogger(__name__)


class MotionPredictor:
    """
    Manages dual Kalman filters (CV and CA) per tracking ID.
    Performs one-step-ahead error evaluation and multi-step future trajectory projection.
    """

    def __init__(
        self,
        horizon: int = config.PREDICTION_HORIZON,
        error_window: int = config.PREDICTION_ERROR_WINDOW,
        min_track_length: int = config.MIN_TRACK_LENGTH_FOR_PREDICTION,
    ):
        self.horizon = horizon
        self.error_window = error_window
        self.min_track_length = min_track_length

        # Per-ID filter instances
        self.cv_filters: Dict[int, KalmanFilterCV] = {}
        self.ca_filters: Dict[int, KalmanFilterCA] = {}

        # 1-step ahead prediction storage: tid -> (pred_x, pred_y) generated at frame t for frame t+1
        self.pending_cv_preds: Dict[int, Tuple[float, float]] = {}
        self.pending_ca_preds: Dict[int, Tuple[float, float]] = {}

        # Rolling error histories: tid -> deque of float errors
        self.cv_errors: Dict[int, deque] = {}
        self.ca_errors: Dict[int, deque] = {}

        # Track observation count: tid -> int
        self.track_counts: Dict[int, int] = {}

        # Cumulative system-wide prediction stats
        self.total_evaluations = 0
        self.total_cv_error_sum = 0.0
        self.total_ca_error_sum = 0.0

    def update_and_predict(self, tracking_result: dict) -> dict:
        """
        Process tracking results:
        1. Evaluate previous 1-step predictions against current measurements.
        2. Update Kalman filters with current measurements.
        3. Formulate next 1-step predictions (for evaluation on next frame).
        4. Select best model per ID and project multi-step trajectory.

        Args:
            tracking_result: Output from ObjectTracker.update().

        Returns:
            Dict containing:
                'predictions': Dict mapping tid -> {
                    'selected_model': 'CV' or 'CA',
                    'future_points': List of (x, y) tuples,
                    'cv_mae': float,
                    'ca_mae': float
                }
                'overall_cv_mae': float,
                'overall_ca_mae': float,
        """
        tracker_ids = tracking_result.get("tracker_ids", [])
        centers = tracking_result.get("centers", [])

        current_active_ids = set()
        predictions_output = {}

        for tid_raw, center in zip(tracker_ids, centers):
            tid = int(tid_raw)
            current_active_ids.add(tid)
            meas_x, meas_y = float(center[0]), float(center[1])

            # ── 1. Evaluate Pending Predictions from Frame t-1 ────────────────
            if tid in self.pending_cv_preds and tid in self.pending_ca_preds:
                cv_pred = self.pending_cv_preds[tid]
                ca_pred = self.pending_ca_preds[tid]

                cv_err = math.hypot(cv_pred[0] - meas_x, cv_pred[1] - meas_y)
                ca_err = math.hypot(ca_pred[0] - meas_x, ca_pred[1] - meas_y)

                if tid not in self.cv_errors:
                    self.cv_errors[tid] = deque(maxlen=self.error_window)
                    self.ca_errors[tid] = deque(maxlen=self.error_window)

                self.cv_errors[tid].append(cv_err)
                self.ca_errors[tid].append(ca_err)

                self.total_evaluations += 1
                self.total_cv_error_sum += cv_err
                self.total_ca_error_sum += ca_err

            # ── 2. Initialize or Update Kalman Filters ────────────────────────
            if tid not in self.cv_filters:
                self.cv_filters[tid] = KalmanFilterCV(initial_pos=(meas_x, meas_y))
                self.ca_filters[tid] = KalmanFilterCA(initial_pos=(meas_x, meas_y))
                self.track_counts[tid] = 1
            else:
                self.track_counts[tid] += 1
                # Kalman Predict step
                self.cv_filters[tid].predict()
                self.ca_filters[tid].predict()
                # Kalman Measurement Update step
                self.cv_filters[tid].update((meas_x, meas_y))
                self.ca_filters[tid].update((meas_x, meas_y))

            # ── 3. Generate Next 1-Step Prediction (for frame t+1 evaluation) ──
            # Lookahead of 1 step from updated state
            cv_next_1 = self.cv_filters[tid].predict_future(steps=1)[0]
            ca_next_1 = self.ca_filters[tid].predict_future(steps=1)[0]
            self.pending_cv_preds[tid] = cv_next_1
            self.pending_ca_preds[tid] = ca_next_1

            # ── 4. Dynamic Motion Model Selection (CV vs CA) ──────────────────
            cv_mae = (
                float(np.mean(self.cv_errors[tid]))
                if (tid in self.cv_errors and len(self.cv_errors[tid]) > 0)
                else 0.0
            )
            ca_mae = (
                float(np.mean(self.ca_errors[tid]))
                if (tid in self.ca_errors and len(self.ca_errors[tid]) > 0)
                else 0.0
            )

            # Require a minimum track history before selecting CA
            # CV is typically more stable on early short tracks
            if self.track_counts[tid] >= self.min_track_length and len(self.ca_errors.get(tid, [])) >= 3:
                selected_model = "CA" if ca_mae < cv_mae else "CV"
            else:
                selected_model = "CV"

            # ── 5. Extrapolate Future Horizon Trajectory ──────────────────────
            if selected_model == "CA":
                future_pts = self.ca_filters[tid].predict_future(steps=self.horizon)
            else:
                future_pts = self.cv_filters[tid].predict_future(steps=self.horizon)

            predictions_output[tid] = {
                "selected_model": selected_model,
                "future_points": future_pts,
                "cv_mae": cv_mae,
                "ca_mae": ca_mae,
            }

        # ── 6. Clean up dead tracks ───────────────────────────────────────────
        # Clean up IDs that disappeared
        known_ids = list(self.cv_filters.keys())
        for old_id in known_ids:
            if old_id not in current_active_ids:
                # Remove if absent
                self.cv_filters.pop(old_id, None)
                self.ca_filters.pop(old_id, None)
                self.pending_cv_preds.pop(old_id, None)
                self.pending_ca_preds.pop(old_id, None)
                self.cv_errors.pop(old_id, None)
                self.ca_errors.pop(old_id, None)
                self.track_counts.pop(old_id, None)

        overall_cv_mae = (
            (self.total_cv_error_sum / self.total_evaluations)
            if self.total_evaluations > 0
            else 0.0
        )
        overall_ca_mae = (
            (self.total_ca_error_sum / self.total_evaluations)
            if self.total_evaluations > 0
            else 0.0
        )

        return {
            "predictions": predictions_output,
            "overall_cv_mae": overall_cv_mae,
            "overall_ca_mae": overall_ca_mae,
            "total_evaluations": self.total_evaluations,
        }

    def draw_predictions(
        self,
        frame: np.ndarray,
        predictions_result: dict,
        color: Tuple[int, int, int] = config.PREDICTION_COLOR,
    ) -> np.ndarray:
        """
        Draw predicted future trajectories on the frame.

        Args:
            frame: Input BGR image.
            predictions_result: Output from update_and_predict().
            color: Color for predicted trajectory lines and points (BGR).

        Returns:
            Annotated frame copy.
        """
        annotated = frame.copy()
        predictions = predictions_result.get("predictions", {})

        for tid, pred_data in predictions.items():
            pts = pred_data.get("future_points", [])
            model_type = pred_data.get("selected_model", "CV")

            if len(pts) < 2:
                continue

            # Draw trajectory path
            for i in range(len(pts) - 1):
                p1 = (int(pts[i][0]), int(pts[i][1]))
                p2 = (int(pts[i + 1][0]), int(pts[i + 1][1]))
                cv2.line(annotated, p1, p2, color, 2, cv2.LINE_AA)
                # Draw waypoint dot
                cv2.circle(annotated, p2, 3, (0, 165, 255), -1)

            # Label terminal predicted point
            end_pt = (int(pts[-1][0]), int(pts[-1][1]))
            cv2.putText(
                annotated,
                f"Pred:{model_type}",
                (end_pt[0] + 4, end_pt[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
                cv2.LINE_AA,
            )

        return annotated

    def reset(self):
        """Reset internal filter states and histories."""
        self.cv_filters.clear()
        self.ca_filters.clear()
        self.pending_cv_preds.clear()
        self.pending_ca_preds.clear()
        self.cv_errors.clear()
        self.ca_errors.clear()
        self.track_counts.clear()
        self.total_evaluations = 0
        self.total_cv_error_sum = 0.0
        self.total_ca_error_sum = 0.0
        logger.info("MotionPredictor state reset.")
