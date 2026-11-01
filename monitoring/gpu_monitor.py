"""
GPU Monitor Module
==================
Monitors NVIDIA GPU utilization, memory, and temperature using pynvml / nvidia_ml_py
with safe fallback to torch.cuda or CPU mode.
"""

import logging
import warnings
from typing import Dict, Any

warnings.filterwarnings("ignore", message=".*pynvml.*deprecated.*")

logger = logging.getLogger(__name__)

# Check if NVML is available
_NVML_AVAILABLE = False
try:
    try:
        import pynvml
        pynvml.nvmlInit()
        _NVML_AVAILABLE = True
        logger.info("NVML initialized successfully via pynvml.")
    except Exception:
        import nvidia_ml_py as pynvml
        pynvml.nvmlInit()
        _NVML_AVAILABLE = True
        logger.info("NVML initialized successfully via nvidia_ml_py.")
except Exception as e:
    logger.warning(f"NVML could not be initialized: {e}. GPU hardware stats will be simulated/fallback.")
    _NVML_AVAILABLE = False


class GPUMonitor:
    """
    Monitors GPU resources and status.
    Provides memory availability, utilization percentage, and health status.
    """

    def __init__(self, device_index: int = 0):
        self.device_index = device_index
        self.handle = None
        self.available = False
        self.device_name = "CPU (No GPU available)"

        if _NVML_AVAILABLE:
            try:
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
                name = pynvml.nvmlDeviceGetName(self.handle)
                self.device_name = name.decode("utf-8") if isinstance(name, bytes) else str(name)
                self.available = True
                logger.info(f"GPUMonitor tracking: {self.device_name}")
            except Exception as e:
                logger.warning(f"Could not get NVML handle for device {device_index}: {e}")
                self.available = False

    def get_stats(self) -> Dict[str, Any]:
        """
        Query current GPU metrics.

        Returns:
            Dict containing:
                'available': bool
                'device_name': str
                'gpu_utilization_percent': float
                'memory_total_mb': float
                'memory_used_mb': float
                'memory_free_mb': float
                'memory_percent': float
                'temperature_c': float
        """
        if not self.available or self.handle is None:
            return {
                "available": False,
                "device_name": self.device_name,
                "gpu_utilization_percent": 0.0,
                "memory_total_mb": 0.0,
                "memory_used_mb": 0.0,
                "memory_free_mb": 0.0,
                "memory_percent": 0.0,
                "temperature_c": 0.0,
            }

        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            temp = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)

            total_mb = mem.total / (1024 ** 2)
            used_mb = mem.used / (1024 ** 2)
            free_mb = mem.free / (1024 ** 2)
            mem_pct = (used_mb / total_mb * 100.0) if total_mb > 0 else 0.0

            return {
                "available": True,
                "device_name": self.device_name,
                "gpu_utilization_percent": float(util.gpu),
                "memory_total_mb": float(total_mb),
                "memory_used_mb": float(used_mb),
                "memory_free_mb": float(free_mb),
                "memory_percent": float(mem_pct),
                "temperature_c": float(temp),
            }
        except Exception as e:
            logger.error(f"Error querying GPU metrics: {e}")
            return {
                "available": False,
                "device_name": self.device_name,
                "gpu_utilization_percent": 0.0,
                "memory_total_mb": 0.0,
                "memory_used_mb": 0.0,
                "memory_free_mb": 0.0,
                "memory_percent": 0.0,
                "temperature_c": 0.0,
            }

    def close(self):
        """Shutdown NVML cleanly."""
        if _NVML_AVAILABLE:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
