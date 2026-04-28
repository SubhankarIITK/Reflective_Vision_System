"""
Phase 8: Depth Awareness Module
Monocular depth estimation using DPT-Large or simulated depth fallback.
"""

import cv2
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.logger import get_logger

logger = get_logger("depth")


class DepthEstimator:
    def __init__(self, model_name: str = "Intel/dpt-large",
                 use_simulated: bool = False, device: str = "cuda"):
        self.use_simulated = use_simulated
        self.model = None

        if not use_simulated:
            try:
                from transformers import DPTForDepthEstimation, DPTImageProcessor
                import torch
                self.processor = DPTImageProcessor.from_pretrained(model_name)
                self.model = DPTForDepthEstimation.from_pretrained(model_name)
                self.device = device
                self.model.to(device)
                self.model.eval()
                self.torch = torch
                logger.info(f"DPT depth model loaded: {model_name}")
            except Exception as e:
                logger.warning(f"DPT load failed ({e}). Using simulated depth.")
                self.use_simulated = True

    def estimate(self, image: np.ndarray) -> np.ndarray:
        if self.use_simulated:
            return self._simulated_depth(image)
        return self._dpt_depth(image)

    def _dpt_depth(self, image: np.ndarray) -> np.ndarray:
        from PIL import Image as PILImage
        pil_img = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        inputs = self.processor(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with self.torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth

        depth = predicted_depth.squeeze().cpu().numpy()
        depth = cv2.resize(depth, (image.shape[1], image.shape[0]))
        depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return depth

    def _simulated_depth(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Vertical gradient (objects higher in frame = farther)
        h, w = gray.shape
        gradient = np.linspace(255, 80, h, dtype=np.float32)
        depth_base = np.tile(gradient.reshape(h, 1), (1, w))

        # Blur for smooth depth transitions
        depth_blurred = cv2.GaussianBlur(depth_base, (51, 51), 0)

        # Modulate by brightness (bright = reflective = closer)
        brightness = gray.astype(np.float32)
        depth_mod = depth_blurred * 0.7 + brightness * 0.3
        depth = cv2.normalize(depth_mod, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return depth

    def get_object_depth(self, depth_map: np.ndarray, bbox: list[int]) -> dict:
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(depth_map.shape[1], x2), min(depth_map.shape[0], y2)
        region = depth_map[y1:y2, x1:x2]
        if region.size == 0:
            return {"mean": 0.0, "min": 0.0, "max": 0.0, "range": 0.0}
        return {
            "mean": float(np.mean(region)),
            "min": float(np.min(region)),
            "max": float(np.max(region)),
            "range": float(np.max(region) - np.min(region)),
        }

    def colorize_depth(self, depth_map: np.ndarray) -> np.ndarray:
        return cv2.applyColorMap(depth_map, cv2.COLORMAP_PLASMA)

    def overlay_depth(self, image: np.ndarray, depth_map: np.ndarray,
                      alpha: float = 0.35) -> np.ndarray:
        colored = self.colorize_depth(depth_map)
        if colored.shape[:2] != image.shape[:2]:
            colored = cv2.resize(colored, (image.shape[1], image.shape[0]))
        return cv2.addWeighted(image, 1 - alpha, colored, alpha, 0)
