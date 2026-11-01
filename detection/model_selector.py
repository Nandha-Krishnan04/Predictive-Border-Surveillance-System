"""
Dynamic Model Selector Module
==============================
Dynamically selects between YOLOv8 Nano and YOLOv8 Small based on:
1. Environmental conditions (brightness, blur, contrast)
2. Detection confidence over a rolling frame window
3. GPU resource availability (utilization, free VRAM)

Features:
- Configurable thresholds
- Rolling confidence window
- Switching cooldown and hysteresis to prevent oscillation
- Detailed logging of decision rationale
- Safe fallback mechanism
"""

import logging
from collections import deque
from typing import Tuple, Dict, Any

import config
from detection.detector import PersonDetector

logger = logging.getLogger(__name__)


class ModelSelector:
    """
    Manages dynamic switching between YOLO Nano and YOLO Small models.
    """

    def __init__(
        self,
        detector: PersonDetector,
        confidence_window_size: int = config.CONFIDENCE_WINDOW_SIZE,
        low_confidence_threshold: float = config.LOW_CONFIDENCE_THRESHOLD,
        cooldown_frames: int = config.MODEL_SWITCH_COOLDOWN_FRAMES,
        gpu_util_high: float = config.GPU_UTILIZATION_HIGH_THRESHOLD,
        gpu_mem_free_min_mb: float = config.GPU_MEMORY_FREE_THRESHOLD_MB,
    ):
        self.detector = detector
        self.confidence_window_size = confidence_window_size
        self.low_confidence_threshold = low_confidence_threshold
        self.cooldown_frames = cooldown_frames
        self.gpu_util_high = gpu_util_high
        self.gpu_mem_free_min_mb = gpu_mem_free_min_mb

        # Rolling history of mean frame confidence scores
        self.confidence_history = deque(maxlen=confidence_window_size)

        # Frame counters for cooldown
        self.frames_since_switch = cooldown_frames  # allow immediate switch if needed
        self.total_switches = 0
        self.last_decision_reason = "Initial model loaded (Nano)"

    def update_confidence_history(self, detection_result: dict):
        """Record the average confidence of the current frame if any person was detected."""
        confs = detection_result.get("confidences", [])
        if len(confs) > 0:
            avg_conf = float(np.mean(confs))
            self.confidence_history.append(avg_conf)

    def evaluate_and_switch(
        self,
        env_metrics: Dict[str, Any],
        gpu_stats: Dict[str, Any],
        detection_result: dict,
    ) -> Tuple[str, str]:
        """
        Evaluate criteria and switch YOLO model if appropriate.

        Args:
            env_metrics: Output from EnvironmentAnalyzer.analyze().
            gpu_stats: Output from GPUMonitor.get_stats().
            detection_result: Latest detection result dict.

        Returns:
            Tuple of (current_model_name, decision_reason).
        """
        import numpy as np  # Ensure imported in scope

        self.frames_since_switch += 1
        confs = detection_result.get("confidences", [])
        if len(confs) > 0:
            self.confidence_history.append(float(np.mean(confs)))

        current_model = self.detector.model_name
        is_nano = "nano" in current_model.lower() or "8n" in current_model.lower()

        # Check rolling confidence
        has_confidence_data = len(self.confidence_history) >= 5
        rolling_avg_conf = (
            float(np.mean(self.confidence_history)) if has_confidence_data else 1.0
        )
        is_low_confidence = has_confidence_data and (rolling_avg_conf < self.low_confidence_threshold)

        # Resource safety check
        gpu_busy = False
        gpu_low_memory = False
        if gpu_stats.get("available", False):
            gpu_busy = gpu_stats.get("gpu_utilization_percent", 0.0) >= self.gpu_util_high
            gpu_low_memory = gpu_stats.get("memory_free_mb", 1000.0) < self.gpu_mem_free_min_mb

        # Rule evaluation
        target_model = current_model
        reason = "Maintaining current model"

        # Rule A: High GPU load or low VRAM -> Force/Prefer Nano
        if gpu_busy or gpu_low_memory:
            target_model = config.YOLO_NANO_MODEL
            reason = (
                f"GPU constrained (util: {gpu_stats.get('gpu_utilization_percent', 0):.0f}%, "
                f"free VRAM: {gpu_stats.get('memory_free_mb', 0):.0f}MB). Enforcing Nano."
            )

        # Rule B: Difficult environment OR low confidence -> Consider Small (if GPU has headroom)
        elif env_metrics.get("is_difficult", False) or is_low_confidence:
            trigger_factors = []
            if env_metrics.get("is_difficult", False):
                trigger_factors.append(f"adverse conditions [{env_metrics.get('condition_summary')}]")
            if is_low_confidence:
                trigger_factors.append(f"low rolling confidence ({rolling_avg_conf:.2f})")

            triggers_str = " and ".join(trigger_factors)

            if is_nano:
                target_model = config.YOLO_SMALL_MODEL
                reason = f"Upgrading to Small due to {triggers_str} with sufficient GPU capacity."
            else:
                target_model = config.YOLO_SMALL_MODEL
                reason = f"Retaining Small due to {triggers_str}."

        # Rule C: Normal environment AND good confidence -> Prefer Nano for optimal efficiency
        else:
            if not is_nano:
                target_model = config.YOLO_NANO_MODEL
                reason = "Conditions clear and confidence stable; downgrading to Nano for speed/efficiency."
            else:
                target_model = config.YOLO_NANO_MODEL
                reason = "Normal conditions; retaining Nano."

        # Apply Cooldown & Hysteresis to prevent frequent flipping
        if target_model != current_model:
            if self.frames_since_switch < self.cooldown_frames:
                # Cooldown active; do not switch yet
                suppressed_reason = (
                    f"Switch to {target_model} requested ({reason}), but held by cooldown "
                    f"({self.frames_since_switch}/{self.cooldown_frames} frames)."
                )
                self.last_decision_reason = suppressed_reason
                return current_model, suppressed_reason

            # Execute model switch
            logger.info(f"Dynamic Model Switch: {current_model} -> {target_model}. Reason: {reason}")
            success = self.detector.load_model(target_model)
            if success:
                self.frames_since_switch = 0
                self.total_switches += 1
                self.last_decision_reason = reason
                return target_model, reason
            else:
                fallback_reason = f"Failed to switch to {target_model}; safely falling back to {current_model}."
                logger.error(fallback_reason)
                self.last_decision_reason = fallback_reason
                return current_model, fallback_reason

        self.last_decision_reason = reason
        return current_model, reason

    def get_status(self) -> Dict[str, Any]:
        """Return diagnostic metrics of the model selector."""
        import numpy as np

        rolling_avg = float(np.mean(self.confidence_history)) if self.confidence_history else 0.0
        return {
            "current_model": self.detector.model_name,
            "rolling_confidence": rolling_avg,
            "confidence_samples": len(self.confidence_history),
            "cooldown_progress": f"{self.frames_since_switch}/{self.cooldown_frames}",
            "total_switches": self.total_switches,
            "last_reason": self.last_decision_reason,
        }
