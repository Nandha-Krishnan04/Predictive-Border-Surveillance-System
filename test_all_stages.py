"""
Comprehensive Test Suite — Stages 1 through 6
==============================================
Validates:
- Stage 1: Detector (YOLOv8 Nano & Small)
- Stage 2: ByteTrack Tracker (ID persistence, trails)
- Stage 3: Environment Analyzer, GPU Monitor, Dynamic Model Selector
- Stage 4: Kalman Motion Prediction (CV and CA filters, MAE evaluation)
- Stage 5: Boundary Alert System (Perimeter intrusion, cooldown, history)
- Stage 6: MOT17 and MP4 integration
"""

import os
import sys
import unittest
import time
import cv2
import numpy as np

# Windows torch DLL fix
_torch_lib = os.path.join(os.path.dirname(sys.executable), "Lib", "site-packages", "torch", "lib")
if os.path.isdir(_torch_lib) and hasattr(os, "add_dll_directory"):
    try:
        os.add_dll_directory(_torch_lib)
    except OSError:
        pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from detection.detector import PersonDetector
from detection.model_selector import ModelSelector
from environment.environment_analyzer import EnvironmentAnalyzer
from tracking.object_tracker import ObjectTracker
from prediction.kalman_cv import KalmanFilterCV
from prediction.kalman_ca import KalmanFilterCA
from prediction.motion_predictor import MotionPredictor
from monitoring.gpu_monitor import GPUMonitor
from alerts.boundary_alert import BoundaryAlertSystem
from evaluation.metrics import compute_iou, evaluate_frame_detections, parse_mot17_seqinfo


class TestSurveillanceStages(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("\n" + "=" * 60)
        print(" RUNNING INTEGRATED TEST SUITE: STAGES 1 TO 6")
        print("=" * 60)

    # ── Stage 1: Person Detection ──
    def test_01_person_detector(self):
        print("\n[TEST] Stage 1: PersonDetector Initialization & Inference")
        detector = PersonDetector(model_name=config.YOLO_NANO_MODEL, device=config.DEVICE)
        self.assertIsNotNone(detector.model)
        
        # Test inference on dummy frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        res = detector.detect(frame)
        self.assertIn("boxes", res)
        self.assertIn("confidences", res)
        self.assertIn("fps", res)
        self.assertGreater(res["fps"], 0.0)
        detector.release()
        print("  -> Stage 1 PersonDetector verified.")

    # ── Stage 2: ByteTrack Tracking ──
    def test_02_object_tracker(self):
        print("\n[TEST] Stage 2: ObjectTracker (ByteTrack)")
        tracker = ObjectTracker()
        
        # Simulate detection of two people moving across frames
        boxes1 = np.array([[100, 100, 150, 200], [300, 300, 350, 400]], dtype=np.float32)
        confs1 = np.array([0.9, 0.85], dtype=np.float32)
        classes1 = np.array([0, 0], dtype=np.int32)
        
        track_res1 = tracker.update({"boxes": boxes1, "confidences": confs1, "class_ids": classes1})
        self.assertEqual(track_res1["num_tracked"], 2)
        
        # Next frame: slightly shifted positions
        boxes2 = np.array([[105, 102, 155, 202], [305, 302, 355, 402]], dtype=np.float32)
        track_res2 = tracker.update({"boxes": boxes2, "confidences": confs1, "class_ids": classes1})
        self.assertEqual(track_res2["num_tracked"], 2)
        
        # Verify IDs persisted
        self.assertEqual(list(track_res1["tracker_ids"]), list(track_res2["tracker_ids"]))
        # Verify trail history populated
        for tid in track_res2["tracker_ids"]:
            self.assertGreaterEqual(len(track_res2["trail_history"][int(tid)]), 2)
            
        print("  -> Stage 2 ObjectTracker verified.")

    # ── Stage 3: Environment, GPU Monitor & Dynamic Selector ──
    def test_03_environment_and_model_selector(self):
        print("\n[TEST] Stage 3: Environment Analysis & Dynamic Model Selector")
        env_analyzer = EnvironmentAnalyzer()
        
        # Clear frame
        clear_frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        env_clear = env_analyzer.analyze(clear_frame)
        self.assertFalse(env_clear["is_low_light"])
        
        # Dark frame
        dark_frame = np.full((480, 640, 3), 20, dtype=np.uint8)
        env_dark = env_analyzer.analyze(dark_frame)
        self.assertTrue(env_dark["is_low_light"])
        self.assertTrue(env_dark["is_difficult"])
        
        # GPU Monitor
        gpu_monitor = GPUMonitor()
        stats = gpu_monitor.get_stats()
        self.assertIn("available", stats)
        
        # Dynamic Model Selector
        detector = PersonDetector(model_name=config.YOLO_NANO_MODEL, device=config.DEVICE)
        selector = ModelSelector(detector=detector, cooldown_frames=2)
        
        # Under dark conditions, model selector should trigger upgrade to Small
        model_name, reason = selector.evaluate_and_switch(
            env_metrics=env_dark,
            gpu_stats={"available": True, "gpu_utilization_percent": 20.0, "memory_free_mb": 2000.0},
            detection_result={"confidences": [0.3]},
        )
        self.assertEqual(model_name, config.YOLO_SMALL_MODEL)
        self.assertIn("Small", reason)
        
        detector.release()
        gpu_monitor.close()
        print("  -> Stage 3 Environment Analyzer & Model Selector verified.")

    # ── Stage 4: Kalman Motion Prediction ──
    def test_04_kalman_prediction(self):
        print("\n[TEST] Stage 4: Constant Velocity and Constant Acceleration Kalman Filters")
        # 1. Test CV filter
        cv_filter = KalmanFilterCV(initial_pos=(100.0, 100.0))
        cv_filter.predict()
        cv_filter.update((110.0, 110.0))
        future_cv = cv_filter.predict_future(steps=5)
        self.assertEqual(len(future_cv), 5)
        self.assertGreater(future_cv[-1][0], 110.0)
        
        # 2. Test CA filter
        ca_filter = KalmanFilterCA(initial_pos=(100.0, 100.0))
        ca_filter.predict()
        ca_filter.update((110.0, 110.0))
        future_ca = ca_filter.predict_future(steps=5)
        self.assertEqual(len(future_ca), 5)
        
        # 3. Test MotionPredictor with 1-step ahead evaluation rule
        predictor = MotionPredictor()
        # Frame 1: person at (100, 100)
        tr1 = {"tracker_ids": np.array([1]), "centers": np.array([[100.0, 100.0]])}
        pred_res1 = predictor.update_and_predict(tr1)
        self.assertIn(1, pred_res1["predictions"])
        
        # Frame 2: person moved to (105, 105) -> evaluates previous prediction error
        tr2 = {"tracker_ids": np.array([1]), "centers": np.array([[105.0, 105.0]])}
        pred_res2 = predictor.update_and_predict(tr2)
        self.assertGreater(pred_res2["total_evaluations"], 0)
        self.assertGreaterEqual(pred_res2["overall_cv_mae"], 0.0)
        self.assertGreaterEqual(pred_res2["overall_ca_mae"], 0.0)
        
        print("  -> Stage 4 Kalman Motion Predictor verified.")

    # ── Stage 5: Restricted Zone Intrusion Alerts ──
    def test_05_boundary_alerts(self):
        print("\n[TEST] Stage 5: Virtual Restricted Boundary Intrusion Alerts")
        # Define zone from (100, 100) to (300, 300)
        zone_pts = [(100, 100), (300, 100), (300, 300), (100, 300)]
        alert_system = BoundaryAlertSystem(zone_points=zone_pts, alert_cooldown_frames=5)
        
        # Frame 1: Person outside zone at (50, 50)
        tr_out = {
            "tracker_ids": np.array([1]),
            "boxes": np.array([[30, 30, 70, 70]]),
        }
        alerts1 = alert_system.check_zone_entry(tr_out, frame_num=1)
        self.assertEqual(len(alerts1), 0)
        
        # Frame 2: Person enters zone at (150, 150) -> foot at (150, 200)
        tr_in = {
            "tracker_ids": np.array([1]),
            "boxes": np.array([[130, 120, 170, 200]]),
        }
        alerts2 = alert_system.check_zone_entry(tr_in, frame_num=2)
        self.assertEqual(len(alerts2), 1)
        self.assertEqual(alerts2[0]["tracker_id"], 1)
        
        # Frame 3: Same person still inside -> should be suppressed by cooldown!
        alerts3 = alert_system.check_zone_entry(tr_in, frame_num=3)
        self.assertEqual(len(alerts3), 0)
        
        # Frame 10: After cooldown expired -> triggers new alert
        alerts10 = alert_system.check_zone_entry(tr_in, frame_num=10)
        self.assertEqual(len(alerts10), 1)
        
        print("  -> Stage 5 Boundary Alerts verified.")

    # ── Stage 6: Evaluation Metrics Helpers ──
    def test_06_evaluation_metrics(self):
        print("\n[TEST] Stage 6: Evaluation Metrics & IoU")
        b1 = np.array([10, 10, 50, 50])
        b2 = np.array([10, 10, 50, 50])
        self.assertAlmostEqual(compute_iou(b1, b2), 1.0)
        
        b3 = np.array([60, 60, 100, 100])
        self.assertAlmostEqual(compute_iou(b1, b3), 0.0)
        
        # Test detection evaluation
        gt = [{"id": 1, "bbox": [10, 10, 50, 50]}]
        tp, fp, fn = evaluate_frame_detections(np.array([b1]), gt)
        self.assertEqual((tp, fp, fn), (1, 0, 0))
        print("  -> Stage 6 Evaluation Metrics verified.")


if __name__ == "__main__":
    unittest.main()
