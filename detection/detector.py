"""
YOLO Object Detector Module
============================
Handles loading YOLOv8 models and running person detection on video frames.

Key features:
- Loads YOLOv8 Nano (default) or Small models
- Detects only people (COCO class 0)
- Measures inference latency and FPS
- Supports GPU with automatic CPU fallback
- Allows runtime model switching
"""

import os
import sys
import time
import logging
import numpy as np

# Fix DLL loading issue on Windows for PyTorch CUDA builds
# The torch\lib directory must be registered before importing torch
_torch_lib = os.path.join(
    os.path.dirname(sys.executable), "Lib", "site-packages", "torch", "lib"
)
if os.path.isdir(_torch_lib) and hasattr(os, "add_dll_directory"):
    try:
        os.add_dll_directory(_torch_lib)
    except OSError:
        pass

import torch
from ultralytics import YOLO

# Set up module logger
logger = logging.getLogger(__name__)


class PersonDetector:
    """
    Wraps a YOLOv8 model for person-only detection.
    
    Attributes:
        model: The loaded YOLO model instance.
        model_name: Name/path of the currently loaded model.
        device: Inference device ('cuda:0' or 'cpu').
        conf_threshold: Minimum detection confidence.
        iou_threshold: NMS IoU threshold.
        last_inference_time: Time taken for the most recent inference (seconds).
        last_fps: Instantaneous FPS from the most recent inference.
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence: float = 0.35,
        iou_threshold: float = 0.45,
        input_size: int = 640,
        device: str = "auto",
    ):
        """
        Initialize the detector.

        Args:
            model_name: YOLOv8 model file name (e.g., 'yolov8n.pt').
            confidence: Minimum confidence threshold for detections.
            iou_threshold: IoU threshold for Non-Maximum Suppression.
            input_size: Input image size for YOLO (square, in pixels).
            device: 'auto', 'cpu', 'cuda', or 'cuda:0'.
        """
        self.model = None
        self.model_name = model_name
        self.conf_threshold = confidence
        self.iou_threshold = iou_threshold
        self.input_size = input_size
        self.last_inference_time = 0.0
        self.last_fps = 0.0

        # Resolve device
        self.device = self._resolve_device(device)
        logger.info(f"Detector will use device: {self.device}")

        # Load the model
        self.load_model(model_name)

    def _resolve_device(self, device: str) -> str:
        """
        Determine the inference device.
        
        'auto' checks for CUDA availability.
        Returns 'cuda:0' or 'cpu'.
        """
        if device == "auto":
            if torch.cuda.is_available():
                resolved = "cuda:0"
                gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"GPU detected: {gpu_name}")
            else:
                resolved = "cpu"
                logger.info("No GPU detected, using CPU")
            return resolved
        return device

    def load_model(self, model_name: str) -> bool:
        """
        Load a YOLO model, releasing the previous one first.

        Args:
            model_name: Model file to load (e.g., 'yolov8n.pt').

        Returns:
            True if loaded successfully, False otherwise.
        """
        try:
            # Release existing model to free memory
            if self.model is not None:
                del self.model
                self.model = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info(f"Released previous model from memory")

            logger.info(f"Loading model: {model_name} on {self.device}")
            self.model = YOLO(model_name)

            # Run a small warmup inference to initialize the model on the device
            # This ensures the model is loaded onto GPU before real inference
            dummy = np.zeros((64, 64, 3), dtype=np.uint8)
            self.model.predict(
                dummy,
                device=self.device,
                verbose=False,
                imgsz=64,
            )

            self.model_name = model_name
            logger.info(f"Model '{model_name}' loaded successfully on {self.device}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model '{model_name}': {e}")
            # Attempt fallback to CPU if GPU failed
            if "cuda" in self.device:
                logger.warning("Attempting CPU fallback...")
                self.device = "cpu"
                try:
                    self.model = YOLO(model_name)
                    self.model_name = model_name
                    logger.info(f"Model '{model_name}' loaded on CPU (fallback)")
                    return True
                except Exception as e2:
                    logger.error(f"CPU fallback also failed: {e2}")
            return False

    def detect(self, frame: np.ndarray) -> dict:
        """
        Run person detection on a single frame.

        Args:
            frame: BGR image as a NumPy array (H, W, 3).

        Returns:
            Dictionary with keys:
                'boxes': np.ndarray of shape (N, 4) — [x1, y1, x2, y2]
                'confidences': np.ndarray of shape (N,) — confidence scores
                'class_ids': np.ndarray of shape (N,) — all zeros (person)
                'inference_time': float — seconds for this inference
                'fps': float — instantaneous FPS
                'num_detections': int — number of people detected
                'model_name': str — which model produced these results
        """
        if self.model is None:
            logger.error("No model loaded. Cannot run detection.")
            return self._empty_result()

        # Measure inference time
        start_time = time.perf_counter()

        # Run YOLO inference
        # classes=[0] filters for person only (COCO class 0)
        results = self.model.predict(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=[0],          # Person class only
            device=self.device,
            imgsz=self.input_size,
            verbose=False,        # Suppress per-frame console output
        )

        end_time = time.perf_counter()
        inference_time = end_time - start_time
        fps = 1.0 / inference_time if inference_time > 0 else 0.0

        self.last_inference_time = inference_time
        self.last_fps = fps

        # Parse results from the first (and only) image
        result = results[0]

        if result.boxes is not None and len(result.boxes) > 0:
            # Extract bounding boxes [x1, y1, x2, y2] in pixel coords
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
        else:
            boxes = np.empty((0, 4), dtype=np.float32)
            confidences = np.empty((0,), dtype=np.float32)
            class_ids = np.empty((0,), dtype=np.int32)

        return {
            "boxes": boxes,
            "confidences": confidences,
            "class_ids": class_ids,
            "inference_time": inference_time,
            "fps": fps,
            "num_detections": len(boxes),
            "model_name": self.model_name,
        }

    def _empty_result(self) -> dict:
        """Return an empty detection result."""
        return {
            "boxes": np.empty((0, 4), dtype=np.float32),
            "confidences": np.empty((0,), dtype=np.float32),
            "class_ids": np.empty((0,), dtype=np.int32),
            "inference_time": 0.0,
            "fps": 0.0,
            "num_detections": 0,
            "model_name": self.model_name,
        }

    def get_model_info(self) -> dict:
        """Return information about the currently loaded model."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "conf_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold,
            "input_size": self.input_size,
        }

    def release(self):
        """Release model and free GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Detector resources released")
