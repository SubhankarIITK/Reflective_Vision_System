"""
Phase 4: Segmentation Module using Segment Anything Model (SAM)
Generates precise masks for detected objects.
Run: python src/segmentation.py --image sample.jpg --weights models/sam_vit_h_4b8939.pth
"""

import cv2
import numpy as np
import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.config import load_config
from utils.logger import get_logger

logger = get_logger("segmentation")


class SAMSegmentor:
    def __init__(self, checkpoint: str, model_type: str = "vit_h", device: str = "cuda"):
        try:
            from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
            self.SamPredictor = SamPredictor
            self.SamAutomaticMaskGenerator = SamAutomaticMaskGenerator

            if not Path(checkpoint).exists():
                raise FileNotFoundError(
                    f"SAM checkpoint not found: {checkpoint}\n"
                    "Download: wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P models/"
                )

            sam = sam_model_registry[model_type](checkpoint=checkpoint)
            sam.to(device=device)
            self.predictor = SamPredictor(sam)
            self.auto_generator = SamAutomaticMaskGenerator(
                sam,
                points_per_side=32,
                pred_iou_thresh=0.88,
                stability_score_thresh=0.95,
                crop_n_layers=1,
                crop_n_points_downscale_factor=2,
            )
            self.available = True
            logger.info(f"SAM loaded: {model_type} on {device}")

        except ImportError:
            logger.warning("segment_anything not installed. Using fallback segmentation.")
            self.available = False

    def segment_from_boxes(self, image: np.ndarray, boxes: list[list[int]]) -> list[np.ndarray]:
        if not self.available or not boxes:
            return self._fallback_masks(image, boxes)

        self.predictor.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        masks = []
        for box in boxes:
            x1, y1, x2, y2 = box
            import torch
            box_tensor = np.array([x1, y1, x2, y2])
            mask, score, _ = self.predictor.predict(
                box=box_tensor,
                multimask_output=False,
            )
            masks.append(mask[0].astype(np.uint8) * 255)
        return masks

    def segment_auto(self, image: np.ndarray) -> list[dict]:
        if not self.available:
            return []
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self.auto_generator.generate(rgb)

    def _fallback_masks(self, image: np.ndarray, boxes: list[list[int]]) -> list[np.ndarray]:
        masks = []
        h, w = image.shape[:2]
        for box in boxes:
            x1, y1, x2, y2 = box
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[y1:y2, x1:x2] = 255
            # Refine with GrabCut
            mask = self._grabcut_refine(image, mask, box)
            masks.append(mask)
        return masks

    def _grabcut_refine(self, image: np.ndarray, rough_mask: np.ndarray,
                        box: list[int]) -> np.ndarray:
        try:
            x1, y1, x2, y2 = box
            rect = (x1, y1, x2 - x1, y2 - y1)
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            gc_mask = np.zeros(image.shape[:2], np.uint8)
            cv2.grabCut(image, gc_mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            result = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
            return result
        except Exception:
            return rough_mask

    def draw_masks(self, image: np.ndarray, masks: list[np.ndarray],
                   alpha: float = 0.4) -> np.ndarray:
        out = image.copy()
        colors = [
            (0, 120, 255), (0, 200, 100), (255, 180, 0), (200, 0, 200),
            (0, 200, 200), (255, 100, 0), (100, 100, 255), (50, 200, 50)
        ]
        for i, mask in enumerate(masks):
            color = colors[i % len(colors)]
            colored = np.zeros_like(image)
            colored[:] = color
            mask_bool = mask > 127
            out[mask_bool] = (
                out[mask_bool] * (1 - alpha) + colored[mask_bool] * alpha
            ).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, contours, -1, color, 2)
        return out


def run_segmentation(image_path: str, weights: str, output_dir: str = "outputs/images",
                     config_path: str = "configs/config.yaml"):
    cfg = load_config(config_path)
    seg_cfg = cfg["segmentation"]
    os.makedirs(output_dir, exist_ok=True)

    segmentor = SAMSegmentor(weights, seg_cfg["sam_model_type"])
    image = cv2.imread(image_path)

    auto_masks = segmentor.segment_auto(image)
    logger.info(f"Auto-generated {len(auto_masks)} masks")

    masks = [m["segmentation"].astype(np.uint8) * 255 for m in auto_masks[:8]]
    result = segmentor.draw_masks(image, masks)

    out_path = Path(output_dir) / f"seg_{Path(image_path).name}"
    cv2.imwrite(str(out_path), result)
    logger.info(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--weights", default="models/sam_vit_h_4b8939.pth")
    parser.add_argument("--output", default="outputs/images")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    run_segmentation(args.image, args.weights, args.output, args.config)
