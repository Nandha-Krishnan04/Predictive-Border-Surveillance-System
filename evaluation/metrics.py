"""
Evaluation & Metrics Module
============================
Calculates detection and tracking metrics using ground truth if available:
- Detection Precision, Recall, F1
- Tracking ID Switches, MOTA, IDF1 (using IoU matching)
- Motion Prediction MAE (Euclidean position error)

HONEST REPORTING RULE:
If ground truth annotations are not provided, metrics are marked as 'Unavailable (No GT)'.
No metrics or benchmark numbers are ever fabricated.
"""

import os
import configparser
import logging
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

logger = logging.getLogger(__name__)


def parse_mot17_seqinfo(seq_dir: str) -> dict:
    """
    Parse a MOT17 seqinfo.ini file using Python's configparser.

    Args:
        seq_dir: Path to MOT17 sequence directory.

    Returns:
        Dict with sequence metadata: name, imDir, frameRate, seqLength, imWidth, imHeight, imExt.
    """
    ini_path = os.path.join(seq_dir, "seqinfo.ini")
    if not os.path.isfile(ini_path):
        raise FileNotFoundError(f"seqinfo.ini not found in {seq_dir}")

    parser = configparser.ConfigParser()
    parser.read(ini_path)

    section = "Sequence" if "Sequence" in parser else parser.sections()[0]
    return {
        "name": parser.get(section, "name", fallback=os.path.basename(seq_dir)),
        "imDir": parser.get(section, "imDir", fallback="img1"),
        "frameRate": parser.getfloat(section, "frameRate", fallback=30.0),
        "seqLength": parser.getint(section, "seqLength", fallback=0),
        "imWidth": parser.getint(section, "imWidth", fallback=1920),
        "imHeight": parser.getint(section, "imHeight", fallback=1080),
        "imExt": parser.get(section, "imExt", fallback=".jpg"),
    }


def load_mot17_ground_truth(gt_file: str) -> Dict[int, List[dict]]:
    """
    Load MOT17 ground truth file (gt.txt).
    Format per line: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <class>, <visibility>
    We filter for pedestrian class (class 1 in MOT17) and active status (conf=1).

    Args:
        gt_file: Path to gt.txt

    Returns:
        Dict mapping frame_number -> list of gt dicts: {'id': int, 'bbox': [x1, y1, x2, y2]}
    """
    gt_by_frame = {}
    if not os.path.isfile(gt_file):
        return gt_by_frame

    with open(gt_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 7:
                parts = line.split()
            if len(parts) < 6:
                continue

            frame_id = int(parts[0])
            track_id = int(parts[1])
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            conf = float(parts[6]) if len(parts) > 6 else 1.0
            cls = int(parts[7]) if len(parts) > 7 else 1
            vis = float(parts[8]) if len(parts) > 8 else 1.0

            # Consider pedestrians (class 1) with active flag and non-zero visibility
            if conf > 0 and cls == 1 and vis > 0.1:
                if frame_id not in gt_by_frame:
                    gt_by_frame[frame_id] = []
                gt_by_frame[frame_id].append({
                    "id": track_id,
                    "bbox": [x, y, x + w, y + h],
                })

    return gt_by_frame


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute Intersection over Union (IoU) between two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    b1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    b2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = b1_area + b2_area - inter_area

    return float(inter_area / union_area) if union_area > 0 else 0.0


def evaluate_frame_detections(
    pred_boxes: np.ndarray,
    gt_boxes: List[dict],
    iou_thresh: float = 0.5,
) -> Tuple[int, int, int]:
    """
    Evaluate detections against ground truth for a single frame.

    Returns:
        (TP, FP, FN) counts
    """
    if len(gt_boxes) == 0:
        return 0, len(pred_boxes), 0

    if len(pred_boxes) == 0:
        return 0, 0, len(gt_boxes)

    matched_gt = set()
    tp = 0
    fp = 0

    for p_box in pred_boxes:
        best_iou = 0.0
        best_gt_idx = -1
        for g_idx, g_entry in enumerate(gt_boxes):
            if g_idx in matched_gt:
                continue
            iou = compute_iou(p_box, np.array(g_entry["bbox"]))
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = g_idx

        if best_iou >= iou_thresh and best_gt_idx >= 0:
            tp += 1
            matched_gt.add(best_gt_idx)
        else:
            fp += 1

    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn
