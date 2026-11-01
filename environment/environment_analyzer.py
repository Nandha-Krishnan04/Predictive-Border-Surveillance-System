"""
Environment Analyzer Module
============================
Analyzes video frame environmental quality:
- Brightness (mean pixel luminance)
- Contrast (standard deviation of grayscale values)
- Blur / Visibility (Laplacian variance)
- Condition classification (normal, low-light, low-visibility, blurry, overexposed)
"""

import cv2
import numpy as np
import config


class EnvironmentAnalyzer:
    """
    Computes visual quality heuristics for incoming video frames.
    Provides condition tags used by the dynamic model selector.
    """

    def __init__(
        self,
        brightness_low: float = config.BRIGHTNESS_LOW_THRESHOLD,
        brightness_high: float = config.BRIGHTNESS_HIGH_THRESHOLD,
        contrast_low: float = config.CONTRAST_LOW_THRESHOLD,
        blur_thresh: float = config.BLUR_THRESHOLD,
    ):
        self.brightness_low = brightness_low
        self.brightness_high = brightness_high
        self.contrast_low = contrast_low
        self.blur_thresh = blur_thresh

    def analyze(self, frame: np.ndarray) -> dict:
        """
        Analyze environmental metrics for a given BGR frame.

        Args:
            frame: BGR frame (H, W, 3).

        Returns:
            Dict containing:
                'brightness': float (0-255)
                'contrast': float (0-128+)
                'blur_score': float (Laplacian variance)
                'is_low_light': bool
                'is_overexposed': bool
                'is_low_contrast': bool
                'is_blurry': bool
                'is_difficult': bool (True if any adverse condition exists)
                'condition_summary': str (Human-readable summary)
        """
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # 1. Brightness: Mean pixel intensity
        brightness = float(np.mean(gray))

        # 2. Contrast: Standard deviation of pixel intensities
        contrast = float(np.std(gray))

        # 3. Blur: Variance of the Laplacian
        # Low variance indicates low high-frequency edges (blurry or foggy)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        blur_score = float(laplacian.var())

        # Classify conditions
        is_low_light = brightness < self.brightness_low
        is_overexposed = brightness > self.brightness_high
        is_low_contrast = contrast < self.contrast_low
        is_blurry = blur_score < self.blur_thresh

        # Difficult environment heuristic
        is_difficult = is_low_light or is_blurry or is_low_contrast or is_overexposed

        # Human-readable tag
        conditions = []
        if is_low_light:
            conditions.append("Low-Light")
        if is_overexposed:
            conditions.append("Overexposed")
        if is_low_contrast:
            conditions.append("Low-Contrast")
        if is_blurry:
            conditions.append("Blurry/Foggy")

        if not conditions:
            summary = "Clear / Normal"
        else:
            summary = ", ".join(conditions)

        return {
            "brightness": brightness,
            "contrast": contrast,
            "blur_score": blur_score,
            "is_low_light": is_low_light,
            "is_overexposed": is_overexposed,
            "is_low_contrast": is_low_contrast,
            "is_blurry": is_blurry,
            "is_difficult": is_difficult,
            "condition_summary": summary,
        }
