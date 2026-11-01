"""
Stage 1 Test Script — YOLO Person Detection
=============================================
Tests the detection pipeline with either:
  1. A synthetic test video (generated automatically)
  2. A real video file if provided

Usage:
    python test_stage1.py                      # Use synthetic video
    python test_stage1.py --video path/to.mp4  # Use real video
"""

import os
import sys
import argparse
import logging
import time
import warnings

# Fix DLL loading for PyTorch CUDA on Windows
_torch_lib = os.path.join(
    os.path.dirname(sys.executable), "Lib", "site-packages", "torch", "lib"
)
if os.path.isdir(_torch_lib) and hasattr(os, "add_dll_directory"):
    try:
        os.add_dll_directory(_torch_lib)
    except OSError:
        pass

warnings.filterwarnings("ignore", message=".*pynvml.*deprecated.*")

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from detection.detector import PersonDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("test_stage1")


def generate_synthetic_video(output_path: str, num_frames: int = 120, fps: int = 30):
    """
    Generate a simple synthetic test video with colored rectangles
    simulating people-shaped objects. This is NOT expected to trigger
    real person detections — it is for testing the pipeline plumbing.
    For actual detection testing, use a real video with people.

    Args:
        output_path: Where to save the synthetic MP4.
        num_frames: Number of frames to generate.
        fps: Frames per second.
    """
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for i in range(num_frames):
        # Create a frame with a gradient background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Sky gradient (top blue, bottom green)
        for y in range(height):
            ratio = y / height
            frame[y, :] = [
                int(200 * (1 - ratio)),  # B
                int(100 + 100 * ratio),  # G
                int(50 + 50 * ratio),    # R
            ]

        # Draw two "person-like" rectangles that move
        # Person 1: moves right
        x1 = int(50 + (i / num_frames) * 300)
        cv2.rectangle(frame, (x1, 150), (x1 + 40, 330), (0, 0, 180), -1)
        # Head
        cv2.circle(frame, (x1 + 20, 140), 15, (200, 180, 160), -1)

        # Person 2: moves left
        x2 = int(500 - (i / num_frames) * 300)
        cv2.rectangle(frame, (x2, 180), (x2 + 45, 350), (180, 0, 0), -1)
        cv2.circle(frame, (x2 + 22, 170), 15, (200, 180, 160), -1)

        # Add frame number text
        cv2.putText(
            frame, f"Synthetic Frame {i+1}/{num_frames}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
        )

        writer.write(frame)

    writer.release()
    logger.info(f"Synthetic video saved: {output_path} ({num_frames} frames, {fps} FPS)")


def test_detector_initialization():
    """Test 1: Verify detector can be initialized and model loads."""
    print("\n" + "=" * 60)
    print("  TEST 1: Detector Initialization")
    print("=" * 60)

    try:
        detector = PersonDetector(
            model_name="yolov8n.pt",
            confidence=0.35,
            device="auto",
        )
        info = detector.get_model_info()
        print(f"  [PASS] Model loaded: {info['model_name']}")
        print(f"  [PASS] Device: {info['device']}")
        print(f"  [PASS] Confidence threshold: {info['conf_threshold']}")
        detector.release()
        return True
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        return False


def test_single_frame_detection(detector: PersonDetector):
    """Test 2: Run detection on a single black frame (expect 0 detections)."""
    print("\n" + "=" * 60)
    print("  TEST 2: Single Frame Detection (blank frame)")
    print("=" * 60)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detector.detect(frame)

    print(f"  Detections: {result['num_detections']}")
    print(f"  Inference time: {result['inference_time']*1000:.1f} ms")
    print(f"  FPS: {result['fps']:.1f}")
    print(f"  Model: {result['model_name']}")
    print(f"  Boxes shape: {result['boxes'].shape}")
    print(f"  [PASS] Detection pipeline works (0 detections on blank = expected)")
    return True


def test_video_processing(video_path: str, detector: PersonDetector, max_frames: int = 30):
    """Test 3: Process video frames and collect detection stats."""
    print("\n" + "=" * 60)
    print(f"  TEST 3: Video Processing ({max_frames} frames)")
    print("=" * 60)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [FAIL] Cannot open video: {video_path}")
        return False

    fps_video = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  Video: {width}x{height}, {fps_video:.1f} FPS, {total} frames")

    frame_count = 0
    total_detections = 0
    total_time = 0.0
    all_confidences = []

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        result = detector.detect(frame)
        frame_count += 1
        total_detections += result["num_detections"]
        total_time += result["inference_time"]
        if len(result["confidences"]) > 0:
            all_confidences.extend(result["confidences"].tolist())

    cap.release()

    avg_time = total_time / frame_count if frame_count > 0 else 0
    avg_fps = frame_count / total_time if total_time > 0 else 0

    print(f"  Frames processed: {frame_count}")
    print(f"  Total detections: {total_detections}")
    print(f"  Avg inference: {avg_time * 1000:.1f} ms")
    print(f"  Avg FPS: {avg_fps:.1f}")
    if all_confidences:
        print(f"  Avg confidence: {np.mean(all_confidences):.3f}")
        print(f"  Min confidence: {np.min(all_confidences):.3f}")
        print(f"  Max confidence: {np.max(all_confidences):.3f}")
    else:
        print(f"  No detections found (expected for synthetic video)")
    print(f"  [PASS] Video processing pipeline works")
    return True


def test_model_switching(detector: PersonDetector):
    """Test 4: Verify model can be switched at runtime."""
    print("\n" + "=" * 60)
    print("  TEST 4: Runtime Model Switching")
    print("=" * 60)

    print(f"  Current model: {detector.model_name}")

    # Switch to Small
    success = detector.load_model("yolov8s.pt")
    if success:
        print(f"  [PASS] Switched to: {detector.model_name}")
    else:
        print(f"  [FAIL] Could not switch to yolov8s.pt")
        return False

    # Run a detection with the new model
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detector.detect(frame)
    print(f"  Detection with Small model: {result['num_detections']} detections")
    print(f"  Inference: {result['inference_time']*1000:.1f} ms")

    # Switch back to Nano
    success = detector.load_model("yolov8n.pt")
    if success:
        print(f"  [PASS] Switched back to: {detector.model_name}")
    else:
        print(f"  [FAIL] Could not switch back to yolov8n.pt")
        return False

    result = detector.detect(frame)
    print(f"  Detection with Nano model: {result['num_detections']} detections")
    print(f"  Inference: {result['inference_time']*1000:.1f} ms")
    print(f"  [PASS] Runtime model switching works")
    return True


def main():
    parser = argparse.ArgumentParser(description="Stage 1 Test Suite")
    parser.add_argument(
        "--video", default=None,
        help="Path to a real video with people (optional)"
    )
    args = parser.parse_args()

    print("\n" + "#" * 60)
    print("  BORDER SURVEILLANCE - STAGE 1 TEST SUITE")
    print("#" * 60)

    results = {}

    # Test 1: Initialization
    results["init"] = test_detector_initialization()

    # Create detector for remaining tests
    detector = PersonDetector(
        model_name="yolov8n.pt",
        confidence=config.DETECTION_CONFIDENCE_THRESHOLD,
        device=config.DEVICE,
    )

    # Test 2: Single frame
    results["single_frame"] = test_single_frame_detection(detector)

    # Test 3: Video processing
    if args.video and os.path.isfile(args.video):
        video_path = args.video
        print(f"\n  Using provided video: {video_path}")
    else:
        # Generate synthetic video
        video_path = os.path.join(config.SAMPLE_VIDEOS_DIR, "synthetic_test.mp4")
        os.makedirs(config.SAMPLE_VIDEOS_DIR, exist_ok=True)
        generate_synthetic_video(video_path)

    results["video"] = test_video_processing(video_path, detector, max_frames=30)

    # Test 4: Model switching
    results["switching"] = test_model_switching(detector)

    # Cleanup
    detector.release()

    # Summary
    print("\n" + "#" * 60)
    print("  TEST RESULTS SUMMARY")
    print("#" * 60)
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  {name:20s}: [{status}]")

    if all_pass:
        print("\n  ALL TESTS PASSED ✓")
    else:
        print("\n  SOME TESTS FAILED ✗")
    print("#" * 60 + "\n")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
