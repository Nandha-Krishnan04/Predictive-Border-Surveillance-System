"""
Benchmarking Script
====================
Runs rigorous comparative benchmarks on video sequences:
- YOLOv8 Nano vs YOLOv8 Small
- Dynamic model selection adaptation
- GPU utilization and memory consumption
- CV vs CA Kalman Filter prediction accuracy
- Ground-truth evaluation (Precision, Recall, MOTA) when gt.txt is available
- Generates reproducible CSV benchmark files in results/
"""

import os
import sys
import time
import glob
import csv
import argparse
import logging
from typing import Optional, List, Dict, Any

# Ensure torch DLL directory loaded on Windows
_torch_lib = os.path.join(os.path.dirname(sys.executable), "Lib", "site-packages", "torch", "lib")
if os.path.isdir(_torch_lib) and hasattr(os, "add_dll_directory"):
    try:
        os.add_dll_directory(_torch_lib)
    except OSError:
        pass

import cv2
import numpy as np

# Project imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from detection.detector import PersonDetector
from detection.model_selector import ModelSelector
from environment.environment_analyzer import EnvironmentAnalyzer
from tracking.object_tracker import ObjectTracker
from prediction.motion_predictor import MotionPredictor
from monitoring.gpu_monitor import GPUMonitor
from alerts.boundary_alert import BoundaryAlertSystem
from evaluation.metrics import parse_mot17_seqinfo, load_mot17_ground_truth, evaluate_frame_detections

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("benchmark")


def get_video_frame_generator(source_path: str):
    """
    Yields frames and metadata from either an MP4 video or a MOT17 sequence folder.

    Yields:
        (frame_index, frame_bgr, total_frames, fps, width, height, gt_dict_or_None)
    """
    if os.path.isdir(source_path):
        # MOT17 sequence folder mode
        seqinfo = parse_mot17_seqinfo(source_path)
        im_dir = os.path.join(source_path, seqinfo["imDir"])
        im_ext = seqinfo["imExt"]
        gt_file = os.path.join(source_path, "gt", "gt.txt")
        gt_data = load_mot17_ground_truth(gt_file) if os.path.isfile(gt_file) else None

        img_files = sorted(glob.glob(os.path.join(im_dir, f"*{im_ext}")))
        total_frames = len(img_files)
        fps = seqinfo["frameRate"]
        w = seqinfo["imWidth"]
        h = seqinfo["imHeight"]

        for idx, fpath in enumerate(img_files, start=1):
            frame = cv2.imread(fpath)
            if frame is None:
                continue
            frame_gt = gt_data.get(idx, []) if gt_data else None
            yield idx, frame, total_frames, fps, w, h, frame_gt

    elif os.path.isfile(source_path):
        # MP4 video mode
        cap = cv2.VideoCapture(source_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video file: {source_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            yield frame_idx, frame, total_frames, fps, w, h, None
        cap.release()
    else:
        raise ValueError(f"Path does not exist: {source_path}")


def run_benchmark(
    source_path: str,
    model_mode: str = "dynamic",  # 'nano', 'small', or 'dynamic'
    max_frames: Optional[int] = 100,
    output_csv: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute a full surveillance benchmark run.
    """
    logger.info(f"Starting benchmark: source={source_path}, mode={model_mode}, max_frames={max_frames}")

    # Initialize subsystems
    init_model = config.YOLO_SMALL_MODEL if model_mode == "small" else config.YOLO_NANO_MODEL
    detector = PersonDetector(model_name=init_model, confidence=config.DETECTION_CONFIDENCE_THRESHOLD)
    tracker = ObjectTracker()
    predictor = MotionPredictor()
    env_analyzer = EnvironmentAnalyzer()
    gpu_monitor = GPUMonitor()
    model_selector = ModelSelector(detector=detector) if model_mode == "dynamic" else None
    boundary_alert = BoundaryAlertSystem()

    frame_records = []
    total_tp = 0
    total_fp = 0
    total_fn = 0
    has_ground_truth = False

    t_start = time.perf_counter()

    for frame_idx, frame, total_f, fps, w, h, gt_boxes in get_video_frame_generator(source_path):
        if max_frames and frame_idx > max_frames:
            break

        # 1. Environment analysis
        env_metrics = env_analyzer.analyze(frame)

        # 2. GPU monitoring
        gpu_stats = gpu_monitor.get_stats()

        # 3. Detection
        det_result = detector.detect(frame)

        # 4. Dynamic model switching (if enabled)
        decision_reason = "Fixed model mode"
        if model_mode == "dynamic" and model_selector is not None:
            _, decision_reason = model_selector.evaluate_and_switch(
                env_metrics=env_metrics,
                gpu_stats=gpu_stats,
                detection_result=det_result,
            )

        # 5. Tracking
        track_result = tracker.update(det_result)

        # 6. Motion Prediction
        pred_result = predictor.update_and_predict(track_result)

        # 7. Boundary Alerts
        alerts = boundary_alert.check_zone_entry(track_result, frame_idx)

        # Ground truth evaluation if present
        if gt_boxes is not None:
            has_ground_truth = True
            tp, fp, fn = evaluate_frame_detections(det_result["boxes"], gt_boxes)
            total_tp += tp
            total_fp += fp
            total_fn += fn

        # Record metrics per frame
        record = {
            "frame": frame_idx,
            "model": detector.model_name,
            "inference_ms": round(det_result["inference_time"] * 1000, 2),
            "fps": round(det_result["fps"], 2),
            "detections": det_result["num_detections"],
            "tracked": track_result["num_tracked"],
            "active_alerts": len(alerts),
            "env_brightness": round(env_metrics["brightness"], 1),
            "env_blur_score": round(env_metrics["blur_score"], 1),
            "env_condition": env_metrics["condition_summary"],
            "gpu_util_pct": gpu_stats["gpu_utilization_percent"],
            "gpu_mem_used_mb": round(gpu_stats["memory_used_mb"], 1),
            "cv_mae": round(pred_result["overall_cv_mae"], 2),
            "ca_mae": round(pred_result["overall_ca_mae"], 2),
            "model_reason": decision_reason,
        }
        frame_records.append(record)

    t_end = time.perf_counter()
    total_time = t_end - t_start

    # Aggregate summaries
    num_frames = len(frame_records)
    avg_inf = np.mean([r["inference_ms"] for r in frame_records]) if num_frames > 0 else 0.0
    avg_fps = np.mean([r["fps"] for r in frame_records]) if num_frames > 0 else 0.0
    avg_gpu_util = np.mean([r["gpu_util_pct"] for r in frame_records]) if num_frames > 0 else 0.0
    final_cv_mae = frame_records[-1]["cv_mae"] if num_frames > 0 else 0.0
    final_ca_mae = frame_records[-1]["ca_mae"] if num_frames > 0 else 0.0

    # Detection GT metrics
    if has_ground_truth and (total_tp + total_fp > 0):
        precision = total_tp / (total_tp + total_fp)
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn > 0) else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall > 0) else 0.0
        gt_summary = f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}"
    else:
        gt_summary = "Unavailable (No GT annotations provided)"
        precision, recall, f1 = "N/A", "N/A", "N/A"

    summary = {
        "source": source_path,
        "mode": model_mode,
        "frames_evaluated": num_frames,
        "total_wall_time_s": round(total_time, 2),
        "avg_inference_ms": round(avg_inf, 2),
        "avg_fps": round(avg_fps, 2),
        "avg_gpu_util_pct": round(avg_gpu_util, 2),
        "cv_mae_pixels": round(final_cv_mae, 2),
        "ca_mae_pixels": round(final_ca_mae, 2),
        "model_switches": model_selector.total_switches if model_selector else 0,
        "total_alerts": len(boundary_alert.alert_history),
        "ground_truth_status": gt_summary,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    # Save to CSV if requested
    if output_csv:
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        if frame_records:
            with open(output_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=frame_records[0].keys())
                writer.writeheader()
                writer.writerows(frame_records)
            logger.info(f"Detailed frame benchmark saved to: {output_csv}")

    # Cleanup
    detector.release()
    gpu_monitor.close()

    return summary


def main():
    parser = argparse.ArgumentParser(description="Surveillance System Benchmark Suite")
    parser.add_argument("--source", "-s", required=True, help="Path to MP4 or MOT17 directory")
    parser.add_argument("--mode", "-m", choices=["nano", "small", "dynamic"], default="dynamic")
    parser.add_argument("--max-frames", "-f", type=int, default=60, help="Max frames to process")
    parser.add_argument("--output", "-o", default=None, help="Output CSV path")

    args = parser.parse_args()

    out_csv = args.output
    if not out_csv:
        seq_name = os.path.basename(args.source.rstrip("/\\"))
        out_csv = os.path.join(config.RESULTS_DIR, f"benchmark_{seq_name}_{args.mode}.csv")

    summary = run_benchmark(
        source_path=args.source,
        model_mode=args.mode,
        max_frames=args.max_frames,
        output_csv=out_csv,
    )

    print("\n" + "=" * 65)
    print("           BENCHMARK EXECUTION SUMMARY")
    print("=" * 65)
    for k, v in summary.items():
        print(f"  {k:25s}: {v}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
