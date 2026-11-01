"""
Border Surveillance System — Complete Prototype Pipeline
==========================================================
Integrates all surveillance stages into an end-to-end execution pipeline:
- Mode A: MP4 video input (local file or external USB)
- Mode B: MOT17 sequence folder input (seqinfo.ini via ConfigParser)
- YOLOv8 Person Detection (Nano & Small)
- Supervision ByteTrack Multi-Person Tracking
- Environmental Quality Analysis (Brightness, Blur, Contrast)
- GPU Hardware Monitoring (NVML / CUDA fallback)
- Environment-Aware Dynamic Model Selector
- Dual Kalman Filter Motion Prediction (CV vs CA)
- Virtual Restricted Boundary Intrusion Alerts
- Real-time Video Annotation & Output MP4 Export
"""

import argparse
import os
import sys
import time
import glob
import logging
import warnings
from typing import Optional, Tuple, Generator

# Fix DLL loading for PyTorch CUDA on Windows
_torch_lib = os.path.join(os.path.dirname(sys.executable), "Lib", "site-packages", "torch", "lib")
if os.path.isdir(_torch_lib) and hasattr(os, "add_dll_directory"):
    try:
        os.add_dll_directory(_torch_lib)
    except OSError:
        pass

warnings.filterwarnings("ignore", message=".*pynvml.*deprecated.*")

import cv2
import numpy as np

# Project root imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from detection.detector import PersonDetector
from detection.model_selector import ModelSelector
from environment.environment_analyzer import EnvironmentAnalyzer
from tracking.object_tracker import ObjectTracker
from prediction.motion_predictor import MotionPredictor
from monitoring.gpu_monitor import GPUMonitor
from alerts.boundary_alert import BoundaryAlertSystem
from evaluation.metrics import parse_mot17_seqinfo

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("main_pipeline")


def create_frame_generator(source_path: str) -> Tuple[Generator, float, int, int, int]:
    """
    Creates an iterator returning video frames from either:
      Mode A: MP4 file
      Mode B: MOT17 directory with seqinfo.ini

    Returns:
        (generator, fps, total_frames, width, height)
    """
    if os.path.isdir(source_path):
        # ── Mode B: MOT17 sequence directory ──
        logger.info(f"Input Mode B: MOT17 Sequence Directory -> {source_path}")
        seqinfo = parse_mot17_seqinfo(source_path)
        im_dir = os.path.join(source_path, seqinfo["imDir"])
        im_ext = seqinfo["imExt"]

        if not os.path.isdir(im_dir):
            raise FileNotFoundError(f"Image directory not found: {im_dir}")

        img_files = sorted(glob.glob(os.path.join(im_dir, f"*{im_ext}")))
        total_frames = len(img_files)
        if total_frames == 0:
            raise ValueError(f"No image frames found in {im_dir} with extension {im_ext}")

        fps = seqinfo["frameRate"]
        w = seqinfo["imWidth"]
        h = seqinfo["imHeight"]

        def gen():
            for fpath in img_files:
                frame = cv2.imread(fpath)
                if frame is not None:
                    yield frame

        return gen(), fps, total_frames, w, h

    elif os.path.isfile(source_path):
        # ── Mode A: MP4 video file ──
        logger.info(f"Input Mode A: Video File -> {source_path}")
        cap = cv2.VideoCapture(source_path)
        if not cap.isOpened():
            raise IOError(f"Failed to open video file: {source_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        def gen():
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                yield frame
            cap.release()

        return gen(), fps, total_frames, w, h
    else:
        raise ValueError(f"Input path not found: {source_path}")


def run_pipeline(
    input_path: str,
    output_path: Optional[str] = None,
    dynamic_selection: bool = True,
    model_name: Optional[str] = None,
    show_display: bool = True,
    max_frames: Optional[int] = None,
):
    """
    Execute the end-to-end Predictive Border Surveillance pipeline.
    """
    frame_gen, fps, total_frames, width, height = create_frame_generator(input_path)

    # 1. Initialize detector
    active_model = model_name or config.DEFAULT_MODEL
    detector = PersonDetector(
        model_name=active_model,
        confidence=config.DETECTION_CONFIDENCE_THRESHOLD,
        device=config.DEVICE,
    )

    # 2. Subsystems
    tracker = ObjectTracker()
    predictor = MotionPredictor()
    env_analyzer = EnvironmentAnalyzer()
    gpu_monitor = GPUMonitor()
    boundary_alert = BoundaryAlertSystem()
    model_selector = ModelSelector(detector=detector) if dynamic_selection else None

    # Video Writer
    writer = None
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*config.OUTPUT_VIDEO_CODEC)
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        logger.info(f"Annotated output will be saved to: {output_path}")

    frame_num = 0
    t_start = time.perf_counter()
    last_pred_result = {"overall_cv_mae": 0.0, "overall_ca_mae": 0.0}

    try:
        for frame in frame_gen:
            frame_num += 1
            if max_frames and frame_num > max_frames:
                break

            # ── Step 1: Environment Analysis ──
            env_metrics = env_analyzer.analyze(frame)

            # ── Step 2: GPU Monitoring ──
            gpu_stats = gpu_monitor.get_stats()

            # ── Step 3: YOLO Person Detection ──
            det_result = detector.detect(frame)

            # ── Step 4: Dynamic Model Selection ──
            selection_reason = "Fixed model mode"
            if model_selector is not None:
                _, selection_reason = model_selector.evaluate_and_switch(
                    env_metrics=env_metrics,
                    gpu_stats=gpu_stats,
                    detection_result=det_result,
                )

            # ── Step 5: ByteTrack Object Tracking ──
            track_result = tracker.update(det_result)

            # ── Step 6: Kalman Motion Prediction (CV vs CA) ──
            pred_result = predictor.update_and_predict(track_result)
            last_pred_result = pred_result

            # ── Step 7: Restricted Zone Entry Alerts ──
            new_alerts = boundary_alert.check_zone_entry(track_result, frame_num)

            # ── Step 8: Comprehensive Frame Visualization ──
            # 8a. Draw restricted perimeter
            annotated = boundary_alert.draw_zone(frame, active_alert=len(new_alerts) > 0)

            # 8b. Draw tracks & movement trails
            annotated = tracker.draw_tracks_and_trails(annotated, track_result)

            # 8c. Draw predicted future trajectories
            annotated = predictor.draw_predictions(annotated, pred_result)

            # 8d. Render HUD overlay
            hud_lines = [
                f"Model: {detector.model_name} ({detector.device}) | FPS: {det_result['fps']:.1f}",
                f"Env: {env_metrics['condition_summary']} (Bri:{env_metrics['brightness']:.0f}, Blur:{env_metrics['blur_score']:.0f})",
                f"GPU: {gpu_stats['gpu_utilization_percent']:.0f}% | VRAM Used: {gpu_stats['memory_used_mb']:.0f} MB",
                f"Tracked IDs: {track_result['num_tracked']} | CV MAE: {pred_result['overall_cv_mae']:.1f}px | CA MAE: {pred_result['overall_ca_mae']:.1f}px",
                f"Total Alerts: {len(boundary_alert.alert_history)} | Frame: {frame_num}/{total_frames}",
            ]
            for i, line in enumerate(hud_lines):
                cv2.putText(
                    annotated, line, (12, 22 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA
                )

            # Flash alert banner if recent intrusion occurred
            if len(new_alerts) > 0:
                cv2.rectangle(annotated, (10, height - 50), (width - 10, height - 10), (0, 0, 255), -1)
                alert_text = f"*** INTRUSION ALERT: Person ID {new_alerts[0]['tracker_id']} Entered Zone! ***"
                cv2.putText(
                    annotated, alert_text, (25, height - 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA
                )

            # Write to output video
            if writer is not None:
                writer.write(annotated)

            # Display window
            if show_display:
                disp = annotated
                if width > 1280:
                    scale = 1280.0 / width
                    disp = cv2.resize(annotated, (0, 0), fx=scale, fy=scale)
                cv2.imshow("Border Surveillance - Live Prototype", disp)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("User interrupted with 'q'.")
                    break

            if frame_num % 50 == 0:
                logger.info(
                    f"Frame {frame_num}/{total_frames} | Model: {detector.model_name} | "
                    f"Tracked: {track_result['num_tracked']} | Alerts: {len(boundary_alert.alert_history)}"
                )

    finally:
        total_time = time.perf_counter() - t_start
        if writer is not None:
            writer.release()
        if show_display:
            cv2.destroyAllWindows()
        detector.release()
        gpu_monitor.close()

    avg_fps = frame_num / total_time if total_time > 0 else 0.0
    print("\n" + "=" * 65)
    print("      BORDER SURVEILLANCE PIPELINE EXECUTION COMPLETED")
    print("=" * 65)
    print(f"  Frames Processed:      {frame_num}")
    print(f"  Total Run Time:        {total_time:.2f} s")
    print(f"  Overall Average FPS:   {avg_fps:.2f}")
    print(f"  Model Switches:        {model_selector.total_switches if model_selector else 0}")
    print(f"  Total Alerts Logged:   {len(boundary_alert.alert_history)}")
    print(f"  Final CV MAE:          {last_pred_result['overall_cv_mae']:.2f} pixels")
    print(f"  Final CA MAE:          {last_pred_result['overall_ca_mae']:.2f} pixels")
    if output_path:
        print(f"  Saved Output Video:    {output_path}")
    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Predictive Border Surveillance System")
    parser.add_argument("--input", "-i", required=True, help="Path to MP4 video or MOT17 sequence folder")
    parser.add_argument("--output", "-o", default=None, help="Output annotated MP4 file path")
    parser.add_argument("--fixed-model", default=None, help="Force fixed model (e.g. yolov8n.pt or yolov8s.pt)")
    parser.add_argument("--no-display", action="store_true", help="Run without graphical display window")
    parser.add_argument("--max-frames", "-f", type=int, default=None, help="Max frames to process")

    args = parser.parse_args()

    run_pipeline(
        input_path=args.input,
        output_path=args.output,
        dynamic_selection=(args.fixed_model is None),
        model_name=args.fixed_model,
        show_display=not args.no_display,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
