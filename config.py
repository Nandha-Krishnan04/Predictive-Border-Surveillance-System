"""
Border Surveillance System - Central Configuration
===================================================
All configurable parameters for the surveillance system.
Thresholds, paths, and model settings are defined here.
"""

import os

# ─── Project Paths ───────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SAMPLE_VIDEOS_DIR = os.path.join(PROJECT_ROOT, "sample_videos")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# ─── YOLO Detection Settings ────────────────────────────────────────────────
# Model file names (Ultralytics auto-downloads if missing)
YOLO_NANO_MODEL = "yolov8n.pt"
YOLO_SMALL_MODEL = "yolov8s.pt"

# Default model to load at startup
DEFAULT_MODEL = YOLO_NANO_MODEL

# COCO class ID for 'person'
PERSON_CLASS_ID = 0

# Minimum detection confidence threshold
DETECTION_CONFIDENCE_THRESHOLD = 0.35

# NMS IoU threshold
NMS_IOU_THRESHOLD = 0.45

# Input image size for YOLO inference (pixels, square)
YOLO_INPUT_SIZE = 640

# ─── Device Settings ────────────────────────────────────────────────────────
# "auto" = use GPU if available, else CPU
# "cpu" = force CPU
# "cuda" or "cuda:0" = force GPU
DEVICE = "auto"

# ─── Environment Analysis Thresholds ────────────────────────────────────────
BRIGHTNESS_LOW_THRESHOLD = 60       # Mean brightness below this = low-light
BRIGHTNESS_HIGH_THRESHOLD = 190     # Mean brightness above this = overexposed
CONTRAST_LOW_THRESHOLD = 30         # Std-dev of grayscale below this = low contrast
BLUR_THRESHOLD = 50                 # Laplacian variance below this = blurry

# ─── Dynamic Model Selection ────────────────────────────────────────────────
# Rolling window size for confidence tracking (number of frames)
CONFIDENCE_WINDOW_SIZE = 30

# If average confidence drops below this, consider upgrading model
LOW_CONFIDENCE_THRESHOLD = 0.45

# Minimum frames between model switches (cooldown)
MODEL_SWITCH_COOLDOWN_FRAMES = 60

# GPU utilization threshold (%) — above this, prefer Nano
GPU_UTILIZATION_HIGH_THRESHOLD = 80

# GPU memory free threshold (MB) — below this, prefer Nano
GPU_MEMORY_FREE_THRESHOLD_MB = 500

# ─── Tracking Settings ──────────────────────────────────────────────────────
# ByteTrack parameters
TRACK_HIGH_THRESH = 0.4       # High detection threshold for tracking
TRACK_LOW_THRESH = 0.1        # Low detection threshold for second association
TRACK_BUFFER = 30             # Frames to keep lost tracks before deletion
MATCH_THRESH = 0.8            # Matching threshold for IoU

# Maximum trail length (points to draw for movement trail)
MAX_TRAIL_LENGTH = 50

# ─── Prediction Settings ────────────────────────────────────────────────────
# Number of future steps to predict
PREDICTION_HORIZON = 10

# Rolling window for prediction error comparison
PREDICTION_ERROR_WINDOW = 20

# Minimum track length before starting predictions
MIN_TRACK_LENGTH_FOR_PREDICTION = 5

# ─── Restricted Zone (Default) ──────────────────────────────────────────────
# Default rectangular zone as [x1, y1, x2, y2] in pixel coordinates
# Users can reconfigure via the dashboard
DEFAULT_RESTRICTED_ZONE = [100, 100, 500, 400]

# Alert cooldown per track ID (frames) — avoid repeated alerts for same person
ALERT_COOLDOWN_FRAMES = 30

# ─── Video Output ───────────────────────────────────────────────────────────
OUTPUT_VIDEO_CODEC = "mp4v"
OUTPUT_VIDEO_FPS = 30.0

# ─── Display / Visualization ────────────────────────────────────────────────
# Bounding box colors (BGR format for OpenCV)
BBOX_COLOR = (0, 255, 0)           # Green
BBOX_THICKNESS = 2
TRAIL_COLOR = (255, 255, 0)        # Cyan
PREDICTION_COLOR = (0, 0, 255)     # Red
RESTRICTED_ZONE_COLOR = (0, 0, 255)  # Red
RESTRICTED_ZONE_ALPHA = 0.3        # Transparency for zone overlay
TEXT_COLOR = (255, 255, 255)       # White
TEXT_FONT_SCALE = 0.5
TEXT_THICKNESS = 1
