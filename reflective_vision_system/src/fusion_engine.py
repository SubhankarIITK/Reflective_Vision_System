"""
Phase 6: Multi-Module Fusion Engine
Combines YOLO detections, SAM masks, and CV features with confidence-based fusion.
"""

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.config import load_config
from utils.logger import get_logger

logger = get_logger("fusion_engine")


@dataclass
class FusedDetection:
    bbox: list[int]
    class_id: int
    class_name: str
    detection_conf: float
    segmentation_conf: float
    cv_conf: float
    fused_conf: float
    mask: Optional[np.ndarray] = None
    has_reflection: bool = False
    transparency_score: float = 0.0
    track_id: int = -1


class FusionEngine:
    def __init__(self, config_path: str = "configs/config.yaml"):
        cfg = load_config(config_path)
        fusion_cfg = cfg["fusion"]
        self.det_w = fusion_cfg["detection_weight"]
        self.seg_w = fusion_cfg["segmentation_weight"]
        self.cv_w = fusion_cfg["cv_weight"]
        self.min_conf = fusion_cfg["min_confidence"]

    def _iou(self, box_a: list[int], box_b: list[int]) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        intersection = inter_w * inter_h
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union = area_a + area_b - intersection
        return intersection / union if union > 0 else 0.0

    def _seg_confidence(self, det_box: list[int], masks: list[np.ndarray]) -> float:
        if not masks:
            return 0.0
        x1, y1, x2, y2 = det_box
        best = 0.0
        for mask in masks:
            if mask is None or mask.size == 0:
                continue
            region = mask[y1:y2, x1:x2]
            if region.size == 0:
                continue
            coverage = np.count_nonzero(region) / region.size
            best = max(best, coverage)
        return float(np.clip(best, 0.0, 1.0))

    def _cv_confidence(self, det_box: list[int], cv_features) -> float:
        x1, y1, x2, y2 = det_box
        score = 0.0

        # Edge density inside box
        edge_crop = cv_features.edges[y1:y2, x1:x2]
        if edge_crop.size > 0:
            edge_density = np.count_nonzero(edge_crop) / edge_crop.size
            score += edge_density * 0.4

        # Highlight presence
        hl_crop = cv_features.highlight_mask[y1:y2, x1:x2]
        if hl_crop.size > 0:
            hl_density = np.count_nonzero(hl_crop) / hl_crop.size
            score += min(hl_density * 2.0, 0.4)

        # Transparency
        trans_crop = cv_features.transparency_map[y1:y2, x1:x2]
        if trans_crop.size > 0:
            trans_score = np.mean(trans_crop) / 255.0
            score += trans_score * 0.2

        return float(np.clip(score, 0.0, 1.0))

    def _has_reflection(self, det_box: list[int], cv_features) -> bool:
        for region in cv_features.reflection_regions:
            cx, cy = region["center"]
            x1, y1, x2, y2 = det_box
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                return True
        return False

    def _transparency_score(self, det_box: list[int], cv_features) -> float:
        x1, y1, x2, y2 = det_box
        trans_crop = cv_features.transparency_map[y1:y2, x1:x2]
        if trans_crop.size == 0:
            return 0.0
        return float(np.mean(trans_crop) / 255.0)

    def fuse(self, detections: list[dict], masks: list[np.ndarray],
             cv_features, image_shape: tuple) -> list[FusedDetection]:
        fused = []

        for det in detections:
            box = det["bbox"]
            det_conf = det["confidence"]
            seg_conf = self._seg_confidence(box, masks)
            cv_conf = self._cv_confidence(box, cv_features)

            fused_conf = (
                self.det_w * det_conf +
                self.seg_w * seg_conf +
                self.cv_w * cv_conf
            )

            if fused_conf < self.min_conf:
                continue

            # Find best mask for this detection
            best_mask = None
            best_iou = 0.0
            for mask in masks:
                if mask is None:
                    continue
                h, w = image_shape[:2]
                mask_bbox = self._mask_to_bbox(mask, h, w)
                if mask_bbox:
                    iou = self._iou(box, mask_bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_mask = mask

            fused.append(FusedDetection(
                bbox=box,
                class_id=det["class_id"],
                class_name=det["class_name"],
                detection_conf=det_conf,
                segmentation_conf=seg_conf,
                cv_conf=cv_conf,
                fused_conf=fused_conf,
                mask=best_mask,
                has_reflection=self._has_reflection(box, cv_features),
                transparency_score=self._transparency_score(box, cv_features),
            ))

        fused.sort(key=lambda x: x.fused_conf, reverse=True)
        logger.debug(f"Fused {len(detections)} detections → {len(fused)} kept")
        return fused

    def _mask_to_bbox(self, mask: np.ndarray, h: int, w: int) -> Optional[list[int]]:
        try:
            ys, xs = np.where(mask > 127)
            if len(xs) == 0:
                return None
            return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        except Exception:
            return None

    def draw_fused(self, image: np.ndarray, fused_detections: list[FusedDetection],
                   mask_alpha: float = 0.4) -> np.ndarray:
        out = image.copy()
        colors = [
            (0, 120, 255), (0, 200, 100), (255, 180, 0), (200, 0, 200),
            (0, 200, 200), (255, 100, 0), (100, 100, 255), (50, 200, 50)
        ]

        for fd in fused_detections:
            color = colors[fd.class_id % len(colors)]
            x1, y1, x2, y2 = fd.bbox

            # Draw mask
            if fd.mask is not None:
                mask_bool = fd.mask > 127
                overlay = out.copy()
                overlay[mask_bool] = color
                out = cv2.addWeighted(out, 1 - mask_alpha, overlay, mask_alpha, 0)

            # Draw box
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            # Label
            label = f"{fd.class_name} [{fd.fused_conf:.2f}]"
            if fd.has_reflection:
                label += " R"
            if fd.transparency_score > 0.5:
                label += " T"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(out, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            # Conf bar
            bar_w = x2 - x1
            cv2.rectangle(out, (x1, y2 + 2), (x1 + int(bar_w * fd.fused_conf), y2 + 6), color, -1)

        return out
