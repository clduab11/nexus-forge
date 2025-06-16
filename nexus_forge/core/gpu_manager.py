import json
import logging
import platform
import re
import subprocess
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class GPUManager:
    """Manages GPU detection and VRAM allocation for Ollama models"""

    # Mapping of Ollama models to their approximate VRAM requirements (in GB)
    MODEL_VRAM_REQUIREMENTS = {
        "llama2": 8,
        "llama2:13b": 13,
        "llama2:70b": 35,
        "mistral": 8,
        "mixtral": 24,
        "codellama": 8,
        "phi": 4,
        "neural-chat": 8,
        "stable-beluga": 13,
        "orca-mini": 4,
        "vicuna": 8,
    }

    def __init__(self):
        self.has_nvidia_gpu = False
        self.has_metal_gpu = False
        self.gpu_info: Optional[Dict] = None
        self._detect_gpu()

    def _detect_gpu(self) -> None:
        """Detect GPU and get its specifications"""
        system = platform.system()

        if system == "Darwin":  # macOS
            try:
                # Use system_profiler to get GPU info
                result = subprocess.run(
                    ["system_profiler", "SPDisplaysDataType", "-json"],
                    capture_output=True,
                    text=True,
                    check=True,
                )

                if result.returncode == 0:
                    data = json.loads(result.stdout)
                    gpu_data = data.get("SPDisplaysDataType", [{}])[0]

                    # Check if it's Apple Silicon with Metal support
                    if (
                        "spdisplays_metalv2"
                        in gpu_data.get("spdisplays_metalfamily", "").lower()
                    ):
                        self.has_metal_gpu = True
                        # Apple Silicon typically shares system memory, assume 33% for GPU
                        total_memory = int(
                            subprocess.check_output(["sysctl", "-n", "hw.memsize"])
                            .decode()
                            .strip()
                        ) / (1024 * 1024 * 1024)
                        gpu_memory = total_memory * 0.33  # 33% of system memory

                        self.gpu_info = {
                            "name": gpu_data.get(
                                "spdisplays_device_name", "Apple Silicon GPU"
                            ),
                            "total_memory": gpu_memory,
                            "free_memory": gpu_memory * 0.8,  # Estimate 80% available
                            "used_memory": gpu_memory * 0.2,  # Estimate 20% used
                            "metal_support": True,
                        }
                        logger.info(f"Detected Metal GPU: {self.gpu_info['name']}")
                        return

            except (
                subprocess.SubprocessError,
                FileNotFoundError,
                json.JSONDecodeError,
            ) as e:
                logger.info(f"Error detecting Metal GPU: {str(e)}")

        # Try NVIDIA detection if not on macOS or Metal detection failed
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=gpu_name,memory.total,memory.free,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            if result.returncode == 0:
                self.has_nvidia_gpu = True
                # Parse the CSV output
                gpu_data = result.stdout.strip().split(",")
                self.gpu_info = {
                    "name": gpu_data[0].strip(),
                    "total_memory": float(gpu_data[1].strip()),
                    "free_memory": float(gpu_data[2].strip()),
                    "used_memory": float(gpu_data[3].strip()),
                    "metal_support": False,
                }
                logger.info(f"Detected NVIDIA GPU: {self.gpu_info['name']}")
            else:
                self.has_nvidia_gpu = False
                logger.info("No NVIDIA GPU detected")

        except (subprocess.SubprocessError, FileNotFoundError):
            self.has_nvidia_gpu = False
            logger.info("nvidia-smi not found, assuming no GPU")

    def get_suitable_models(self) -> List[Dict[str, any]]:
        """Get list of Ollama models suitable for the detected GPU"""
        if not (self.has_nvidia_gpu or self.has_metal_gpu) or not self.gpu_info:
            return [
                {
                    "model": model,
                    "vram_required": vram,
                    "suitable": False,
                    "reason": "No compatible GPU detected",
                }
                for model, vram in self.MODEL_VRAM_REQUIREMENTS.items()
            ]

        available_vram = self.gpu_info["total_memory"] / 1024  # Convert to GB
        gpu_type = "Metal" if self.has_metal_gpu else "NVIDIA"

        # For Metal, we're more conservative with VRAM requirements due to shared memory
        vram_buffer = 4 if self.has_metal_gpu else 2
        effective_vram = available_vram - vram_buffer

        return [
            {
                "model": model,
                "vram_required": vram,
                "suitable": vram <= effective_vram,
                "reason": (
                    f"Sufficient {gpu_type} VRAM"
                    if vram <= effective_vram
                    else f"Insufficient {gpu_type} VRAM"
                ),
            }
            for model, vram in self.MODEL_VRAM_REQUIREMENTS.items()
        ]

    def get_recommended_model(self) -> Tuple[str, float]:
        """Get the recommended Ollama model based on available VRAM"""
        if not (self.has_nvidia_gpu or self.has_metal_gpu) or not self.gpu_info:
            return "phi", 4  # Most conservative model for CPU usage

        available_vram = self.gpu_info["total_memory"] / 1024  # Convert to GB

        # More conservative buffer for Metal due to shared memory
        vram_buffer = 4 if self.has_metal_gpu else 2
        effective_vram = available_vram - vram_buffer

        # Find the largest model that fits in available VRAM with buffer
        suitable_models = [
            (model, vram)
            for model, vram in self.MODEL_VRAM_REQUIREMENTS.items()
            if vram <= effective_vram
        ]

        if not suitable_models:
            return "phi", 4  # Fallback to smallest model

        # Return the model with highest VRAM requirement that fits
        return max(suitable_models, key=lambda x: x[1])

    def get_gpu_status(self) -> Dict:
        """Get current GPU status"""
        if not (self.has_nvidia_gpu or self.has_metal_gpu) or not self.gpu_info:
            return {
                "has_gpu": False,
                "gpu_name": None,
                "total_vram": None,
                "free_vram": None,
                "used_vram": None,
                "gpu_type": None,
                "metal_support": False,
            }

        return {
            "has_gpu": True,
            "gpu_name": self.gpu_info["name"],
            "total_vram": self.gpu_info["total_memory"] / 1024,  # Convert to GB
            "free_vram": self.gpu_info["free_memory"] / 1024,
            "used_vram": self.gpu_info["used_memory"] / 1024,
            "gpu_type": "Metal" if self.has_metal_gpu else "NVIDIA",
            "metal_support": self.gpu_info.get("metal_support", False),
        }
