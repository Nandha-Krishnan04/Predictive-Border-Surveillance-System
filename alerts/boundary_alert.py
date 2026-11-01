"""
Boundary Alert Module
=====================
Implements a configurable virtual restricted zone and intrusion detection.

Key features:
- Supports rectangular and arbitrary polygon virtual perimeters
- Point-in-polygon checks for tracked person ground positions
- Cooldown logic to prevent repeated alert flooding for the same person
- Maintains detailed historical alert log
- Visual rendering of the restricted zone with semi-transparent overlay
"""

import time
import logging
from typing import List, Tuple, Dict, Any, Optional
import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)


class BoundaryAlertSystem:
    """
    Monitors tracked personnel against a virtual security perimeter.
    Generates alerts when an unauthorized person crosses into the restricted zone.
    """

    def __init__(
        self,
        zone_points: Optional[List[Tuple[int, int]]] = None,
        alert_cooldown_frames: int = config.ALERT_COOLDOWN_FRAMES,
        zone_color: Tuple[int, int, int] = config.RESTRICTED_ZONE_COLOR,
        zone_alpha: float = config.RESTRICTED_ZONE_ALPHA,
    ):
        """
        Initialize the boundary alert system.

        Args:
            zone_points: List of (x, y) vertices defining the zone polygon.
                         If None, defaults to config.DEFAULT_RESTRICTED_ZONE.
            alert_cooldown_frames: Minimum frame interval between repeat alerts for same ID.
            zone_color: BGR color for boundary visualization.
            zone_alpha: Overlay transparency.
        """
        self.alert_cooldown_frames = alert_cooldown_frames
        self.zone_color = zone_color
        self.zone_alpha = zone_alpha

        # Set perimeter points
        if zone_points is not None:
            self.set_zone(zone_points)
        else:
            # Default rectangle: [x1, y1, x2, y2]
            r = config.DEFAULT_RESTRICTED_ZONE
            self.set_zone([(r[0], r[1]), (r[2], r[1]), (r[2], r[3]), (r[0], r[3])])

        # Per-ID last alert frame tracking
        self.last_alert_frame: Dict[int, int] = {}

        # Alert history: list of event dicts
        self.alert_history: List[Dict[str, Any]] = []

    def set_zone(self, points: List[Tuple[int, int]]):
        """Set or update the polygon vertices for the restricted zone."""
        self.zone_points = np.array(points, dtype=np.int32)
        logger.info(f"Restricted zone configured with {len(points)} vertices.")

    def check_zone_entry(
        self,
        tracking_result: dict,
        frame_num: int,
        timestamp_str: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Check whether any tracked people are inside the restricted boundary.

        Args:
            tracking_result: Output from ObjectTracker.update().
            frame_num: Current video frame index.
            timestamp_str: Optional human-readable timestamp.

        Returns:
            List of newly triggered alert dicts for this frame.
        """
        if timestamp_str is None:
            timestamp_str = time.strftime("%H:%M:%S")

        boxes = tracking_result.get("boxes", [])
        tracker_ids = tracking_result.get("tracker_ids", [])
        new_alerts = []

        for i in range(len(boxes)):
            box = boxes[i]
            tid = int(tracker_ids[i])

            # Use the bottom-center of bounding box (approximate ground/foot contact point)
            foot_x = (box[0] + box[2]) / 2.0
            foot_y = box[3]
            point = (float(foot_x), float(foot_y))

            # Point-in-polygon test (positive distance or 0 means inside/on border)
            # measureDist=False returns +1 (inside), -1 (outside), 0 (on edge)
            dist = cv2.pointPolygonTest(self.zone_points, point, False)
            is_inside = dist >= 0

            if is_inside:
                # Check cooldown to prevent duplicate alert spam on consecutive frames
                last_frame = self.last_alert_frame.get(tid, -self.alert_cooldown_frames - 1)
                if (frame_num - last_frame) > self.alert_cooldown_frames:
                    self.last_alert_frame[tid] = frame_num

                    alert_event = {
                        "alert_id": len(self.alert_history) + 1,
                        "tracker_id": tid,
                        "frame_number": frame_num,
                        "timestamp": timestamp_str,
                        "position": (int(foot_x), int(foot_y)),
                        "event_type": "PROTOTYPE_RESTRICTED_ZONE_ENTRY",
                        "description": f"Person (ID {tid}) entered restricted virtual perimeter at frame {frame_num}",
                    }

                    self.alert_history.append(alert_event)
                    new_alerts.append(alert_event)
                    logger.warning(f"ALERT: Person ID {tid} entered restricted zone at frame {frame_num}!")

        return new_alerts

    def draw_zone(self, frame: np.ndarray, active_alert: bool = False) -> np.ndarray:
        """
        Draw the restricted zone onto the frame with transparent fill and border.

        Args:
            frame: Input BGR image.
            active_alert: If True, draws with brighter flashing style.

        Returns:
            Frame with zone overlay.
        """
        overlay = frame.copy()
        color = (0, 0, 255) if not active_alert else (0, 0, 255)

        # Draw semi-transparent filled polygon
        cv2.fillPoly(overlay, [self.zone_points], color)
        annotated = cv2.addWeighted(overlay, self.zone_alpha, frame, 1.0 - self.zone_alpha, 0)

        # Draw solid contour border
        cv2.polylines(annotated, [self.zone_points], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)

        # Label the zone
        x_min = int(np.min(self.zone_points[:, 0]))
        y_min = int(np.min(self.zone_points[:, 1]))
        label = "RESTRICTED ZONE (PROTOTYPE)"
        cv2.putText(
            annotated,
            label,
            (x_min + 5, y_min - 8 if y_min > 20 else y_min + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

        return annotated

    def reset(self):
        """Reset alert state and history."""
        self.last_alert_frame.clear()
        self.alert_history.clear()
        logger.info("BoundaryAlertSystem reset.")
