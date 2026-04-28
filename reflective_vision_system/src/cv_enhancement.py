"""
Phase 5: Classical CV Enhancement
Edge detection, contour extraction, reflection/highlight detection.
Improves detection of transparent and reflective objects.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.logger import get_logger

logger = get_logger("cv_enhancement")


@dataclass
class CVFeatures:
    edges: np.ndarray
    contours: list
    contour_bboxes: list[list[int]]
    highlight_mask: np.ndarray
    reflection_regions: list[dict]
    transparency_map: np.ndarray
    sharpness_score: float


class CVEnhancer:
    def __init__(self):
        pass

    def extract_edges(self, image: np.ndarray,
                      low_thresh: int = 50, high_thresh: int = 150) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, low_thresh, high_thresh)

        # Multi-scale: merge edges from two scales
        blurred2 = cv2.GaussianBlur(gray, (3, 3), 0)
        edges2 = cv2.Canny(blurred2, low_thresh // 2, high_thresh // 2)
        edges = cv2.bitwise_or(edges, edges2)

        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        return edges

    def extract_contours(self, edges: np.ndarray,
                         min_area: int = 500) -> tuple[list, list[list[int]]]:
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered = [c for c in contours if cv2.contourArea(c) > min_area]
        bboxes = []
        for c in filtered:
            x, y, w, h = cv2.boundingRect(c)
            bboxes.append([x, y, x + w, y + h])
        return filtered, bboxes

    def detect_highlights(self, image: np.ndarray,
                          brightness_thresh: int = 220) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        _, s, v = cv2.split(hsv)
        # High V, low S = specular highlight
        highlight_mask = cv2.bitwise_and(
            cv2.threshold(v, brightness_thresh, 255, cv2.THRESH_BINARY)[1],
            cv2.threshold(s, 40, 255, cv2.THRESH_BINARY_INV)[1]
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        highlight_mask = cv2.dilate(highlight_mask, kernel, iterations=2)
        return highlight_mask

    def detect_reflections(self, image: np.ndarray,
                           highlight_mask: np.ndarray) -> list[dict]:
        contours, _ = cv2.findContours(highlight_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regions = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < 100:
                continue
            x, y, w, h = cv2.boundingRect(c)
            cx, cy = x + w // 2, y + h // 2
            aspect = w / h if h > 0 else 1.0
            regions.append({"bbox": [x, y, x + w, y + h],
                             "center": (cx, cy),
                             "area": area,
                             "aspect_ratio": aspect,
                             "intensity": float(np.mean(image[y:y+h, x:x+w]))})
        return regions

    def estimate_transparency(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Local std dev ~ texture variance; low texture + bright = transparent
        kernel = np.ones((15, 15), np.float32) / 225
        local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        local_sq_mean = cv2.filter2D((gray.astype(np.float32))**2, -1, kernel)
        local_var = np.clip(local_sq_mean - local_mean**2, 0, None)
        local_std = np.sqrt(local_var)

        norm_std = cv2.normalize(local_std, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        transparency_map = 255 - norm_std  # Low texture → high transparency score
        return transparency_map

    def compute_sharpness(self, image: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return float(laplacian.var())

    def analyze(self, image: np.ndarray) -> CVFeatures:
        edges = self.extract_edges(image)
        contours, contour_bboxes = self.extract_contours(edges)
        highlight_mask = self.detect_highlights(image)
        reflection_regions = self.detect_reflections(image, highlight_mask)
        transparency_map = self.estimate_transparency(image)
        sharpness = self.compute_sharpness(image)

        logger.debug(f"Edges: {edges.sum()//255} px | "
                     f"Contours: {len(contours)} | "
                     f"Reflections: {len(reflection_regions)} | "
                     f"Sharpness: {sharpness:.1f}")

        return CVFeatures(
            edges=edges,
            contours=contours,
            contour_bboxes=contour_bboxes,
            highlight_mask=highlight_mask,
            reflection_regions=reflection_regions,
            transparency_map=transparency_map,
            sharpness_score=sharpness,
        )

    def draw_cv_overlay(self, image: np.ndarray, features: CVFeatures) -> np.ndarray:
        out = image.copy()

        # Edge overlay (blue)
        edge_overlay = cv2.cvtColor(features.edges, cv2.COLOR_GRAY2BGR)
        edge_overlay[:, :, 0] = 255
        edge_overlay[:, :, 1] = 0
        edge_overlay[:, :, 2] = 0
        mask = features.edges > 0
        out[mask] = (out[mask] * 0.5 + edge_overlay[mask] * 0.5).astype(np.uint8)

        # Highlight overlay (yellow)
        hl_mask = features.highlight_mask > 0
        out[hl_mask] = (out[hl_mask] * 0.4 + np.array([0, 255, 255]) * 0.6).astype(np.uint8)

        # Contour bboxes (green)
        for bbox in features.contour_bboxes:
            cv2.rectangle(out, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 1)

        cv2.putText(out, f"Sharpness: {features.sharpness_score:.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(out, f"Reflections: {len(features.reflection_regions)}",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return out


if __name__ == "__main__":
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="outputs/images")
    args = parser.parse_args()

    enhancer = CVEnhancer()
    image = cv2.imread(args.image)
    features = enhancer.analyze(image)
    result = enhancer.draw_cv_overlay(image, features)
    os.makedirs(args.output, exist_ok=True)
    out_path = Path(args.output) / f"cv_{Path(args.image).name}"
    cv2.imwrite(str(out_path), result)
    logger.info(f"Saved: {out_path}")
