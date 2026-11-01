"""
ByteTrack Object Tracker Module
================================
Tracks detected people across consecutive video frames using ByteTrack.

Key features:
- Standalone ByteTrack tracker independent of YOLO model switching
- Persists tracking IDs across runtime YOLO Nano <-> Small switches
- Maintains unique tracking IDs across frames
- Tracks and draws movement trails (breadcrumbs) for each tracked person
- Cleans up inactive tracks
"""

import logging
from collections import deque
from types import SimpleNamespace
from typing import Dict, List, Tuple
import cv2
import numpy as np
from ultralytics.trackers.byte_tracker import BYTETracker

import config

logger = logging.getLogger(__name__)


class _DetectionContainer:
    """
    Lightweight container wrapping detections for ByteTrack.
    ByteTrack expects an object with .xywh, .conf, .cls supporting boolean indexing and len().
    """

    def __init__(self, xywh: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        self.xywh = np.asarray(xywh, dtype=np.float32)
        self.conf = np.asarray(conf, dtype=np.float32)
        self.cls = np.asarray(cls, dtype=np.float32)

    def __getitem__(self, idx):
        return _DetectionContainer(self.xywh[idx], self.conf[idx], self.cls[idx])

    def __len__(self):
        return len(self.conf)


class ObjectTracker:
    """
    ByteTrack-based multi-person tracker.
    Maintains persistent IDs and movement trail history independently of detection model.
    """

    def __init__(
        self,
        track_activation_threshold: float = config.TRACK_HIGH_THRESH,
        track_low_threshold: float = config.TRACK_LOW_THRESH,
        lost_track_buffer: int = config.TRACK_BUFFER,
        minimum_matching_threshold: float = config.MATCH_THRESH,
        max_trail_length: int = config.MAX_TRAIL_LENGTH,
    ):
        """
        Initialize the ByteTrack tracker and history records.
        """
        self.args = SimpleNamespace(
            track_high_thresh=track_activation_threshold,
            track_low_thresh=track_low_threshold,
            new_track_thresh=track_activation_threshold,
            track_buffer=lost_track_buffer,
            match_thresh=minimum_matching_threshold,
            fuse_score=True,
        )
        self.tracker = BYTETracker(self.args)
        self.max_trail_length = max_trail_length

        # Dictionary mapping tracker_id -> deque of (x_center, y_center)
        self.trail_history: Dict[int, deque] = {}

        # List of currently active tracker IDs
        self.active_ids: List[int] = []

        logger.info("ByteTrack ObjectTracker initialized.")

    def update(self, detection_result: dict) -> dict:
        """
        Update tracking state using detector outputs from the current frame.

        Args:
            detection_result: Dict containing 'boxes' [x1, y1, x2, y2], 'confidences', 'class_ids'.

        Returns:
            Dict containing:
                'boxes': (N, 4) tracked bounding boxes [x1, y1, x2, y2]
                'tracker_ids': (N,) unique integer tracker IDs
                'confidences': (N,) detection confidence scores
                'class_ids': (N,) class IDs (0 for person)
                'centers': (N, 2) center coordinates (cx, cy)
                'trail_history': Dict of tracker_id -> deque of (cx, cy) points
                'num_tracked': int, number of tracked people in this frame
        """
        boxes = detection_result.get("boxes", np.empty((0, 4), dtype=np.float32))
        confidences = detection_result.get("confidences", np.empty((0,), dtype=np.float32))
        class_ids = detection_result.get("class_ids", np.empty((0,), dtype=np.int32))

        if len(boxes) == 0:
            xywh = np.empty((0, 4), dtype=np.float32)
            confs = np.empty((0,), dtype=np.float32)
            clss = np.empty((0,), dtype=np.float32)
        else:
            boxes = np.asarray(boxes, dtype=np.float32)
            # Convert xyxy to xywh (center_x, center_y, width, height)
            cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
            cy = (boxes[:, 1] + boxes[:, 3]) / 2.0
            w = boxes[:, 2] - boxes[:, 0]
            h = boxes[:, 3] - boxes[:, 1]
            xywh = np.column_stack([cx, cy, w, h])
            confs = np.asarray(confidences, dtype=np.float32)
            clss = np.asarray(class_ids, dtype=np.float32)

        det_wrapper = _DetectionContainer(xywh, confs, clss)
        raw_tracks = self.tracker.update(det_wrapper)

        if len(raw_tracks) > 0:
            # raw_tracks is shape (N, 8): [x1, y1, x2, y2, track_id, score, cls, idx]
            tracked_boxes = raw_tracks[:, :4]
            tracker_ids = raw_tracks[:, 4].astype(int)
            tracked_confidences = raw_tracks[:, 5]
            tracked_class_ids = raw_tracks[:, 6].astype(int)

            # Center coordinates
            centers = np.column_stack([
                (tracked_boxes[:, 0] + tracked_boxes[:, 2]) / 2.0,
                (tracked_boxes[:, 1] + tracked_boxes[:, 3]) / 2.0,
            ])

            self.active_ids = tracker_ids.tolist()

            # Update movement trails
            for tid, center in zip(tracker_ids, centers):
                tid = int(tid)
                pt = (int(center[0]), int(center[1]))
                if tid not in self.trail_history:
                    self.trail_history[tid] = deque(maxlen=self.max_trail_length)
                self.trail_history[tid].append(pt)

        else:
            tracked_boxes = np.empty((0, 4), dtype=np.float32)
            tracker_ids = np.empty((0,), dtype=np.int32)
            tracked_confidences = np.empty((0,), dtype=np.float32)
            tracked_class_ids = np.empty((0,), dtype=np.int32)
            centers = np.empty((0, 2), dtype=np.float32)
            self.active_ids = []

        return {
            "boxes": tracked_boxes,
            "tracker_ids": tracker_ids,
            "confidences": tracked_confidences,
            "class_ids": tracked_class_ids,
            "centers": centers,
            "trail_history": self.trail_history,
            "num_tracked": len(tracked_boxes),
        }

    def draw_tracks_and_trails(
        self,
        frame: np.ndarray,
        tracking_result: dict,
        color: Tuple[int, int, int] = config.BBOX_COLOR,
        trail_color: Tuple[int, int, int] = config.TRAIL_COLOR,
    ) -> np.ndarray:
        """
        Draw bounding boxes with tracker IDs and movement trails on the frame.
        """
        annotated = frame.copy()
        boxes = tracking_result["boxes"]
        tracker_ids = tracking_result["tracker_ids"]
        confidences = tracking_result["confidences"]

        # 1. Draw movement trails
        for tid, points in self.trail_history.items():
            pts_list = list(points)
            if len(pts_list) < 2:
                continue
            for i in range(1, len(pts_list)):
                thickness = int(np.sqrt(float(i) / len(pts_list)) * 2) + 1
                cv2.line(annotated, pts_list[i - 1], pts_list[i], trail_color, thickness)

        # 2. Draw bounding boxes & labels
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i].astype(int)
            tid = int(tracker_ids[i])
            conf = confidences[i]

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, config.BBOX_THICKNESS)

            label = f"ID:{tid} ({conf:.2f})"
            label_size, _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, config.TEXT_FONT_SCALE, config.TEXT_THICKNESS
            )
            cv2.rectangle(
                annotated,
                (x1, y1 - label_size[1] - 6),
                (x1 + label_size[0] + 4, y1),
                color,
                -1,
            )
            cv2.putText(
                annotated,
                label,
                (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                config.TEXT_FONT_SCALE,
                (0, 0, 0),
                config.TEXT_THICKNESS,
            )

        return annotated

    def reset(self):
        """Reset tracking state and histories."""
        self.tracker = BYTETracker(self.args)
        self.trail_history.clear()
        self.active_ids.clear()
        logger.info("ObjectTracker state reset.")
