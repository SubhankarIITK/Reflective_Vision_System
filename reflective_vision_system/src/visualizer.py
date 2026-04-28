"""
Phase 10: Visualization & Output
Overlays bounding boxes, masks, IDs, depth. Saves results as images/videos.
"""

import cv2
import numpy as np
import os
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.logger import get_logger

logger = get_logger("visualizer")

PALETTE = [
    (0, 120, 255), (0, 200, 100), (255, 180, 0), (200, 0, 200),
    (0, 200, 200), (255, 100, 0), (100, 100, 255), (50, 200, 50),
    (255, 50, 50), (50, 255, 200), (200, 100, 255), (255, 200, 50),
]


def color_for_id(track_id: int) -> tuple:
    return PALETTE[track_id % len(PALETTE)]


def draw_rounded_rect(img: np.ndarray, pt1: tuple, pt2: tuple,
                      color: tuple, thickness: int = 2, radius: int = 8):
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
    cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
    cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
    cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
    cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)


class Visualizer:
    def __init__(self, config: dict):
        vis_cfg = config.get("visualization", {})
        self.show_boxes = vis_cfg.get("show_boxes", True)
        self.show_masks = vis_cfg.get("show_masks", True)
        self.show_ids = vis_cfg.get("show_ids", True)
        self.show_depth = vis_cfg.get("show_depth", False)
        self.mask_alpha = vis_cfg.get("mask_alpha", 0.4)
        self.box_thickness = vis_cfg.get("box_thickness", 2)
        self.font_scale = vis_cfg.get("font_scale", 0.6)

    def draw_tracks(self, image: np.ndarray, tracked_detections: list[dict],
                    fused_map: dict = None) -> np.ndarray:
        out = image.copy()

        for track in tracked_detections:
            track_id = track["track_id"]
            bbox = track["bbox"]
            color = color_for_id(track_id)
            x1, y1, x2, y2 = bbox
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.shape[1] - 1, x2), min(image.shape[0] - 1, y2)

            fused = (fused_map or {}).get(track_id)

            # Draw mask
            if self.show_masks and fused and fused.mask is not None:
                mask_bool = fused.mask > 127
                overlay = out.copy()
                overlay[mask_bool] = color
                out = cv2.addWeighted(out, 1 - self.mask_alpha, overlay, self.mask_alpha, 0)

            # Bounding box
            if self.show_boxes:
                draw_rounded_rect(out, (x1, y1), (x2, y2), color, self.box_thickness)

            # Label
            parts = [track["class_name"]]
            if self.show_ids:
                parts.insert(0, f"#{track_id}")
            if fused:
                parts.append(f"{fused.fused_conf:.2f}")
                if fused.has_reflection:
                    parts.append("REFL")
                if fused.transparency_score > 0.5:
                    parts.append("TRANSP")
            label = " | ".join(parts)

            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX,
                                                  self.font_scale, 1)
            lx, ly = x1, max(y1 - 1, th + 8)
            cv2.rectangle(out, (lx, ly - th - 6), (lx + tw + 6, ly + 2), color, -1)
            cv2.putText(out, label, (lx + 3, ly - 2),
                        cv2.FONT_HERSHEY_DUPLEX, self.font_scale, (255, 255, 255), 1,
                        cv2.LINE_AA)

            # Confidence bar
            if fused:
                bar_len = x2 - x1
                bar_fill = int(bar_len * fused.fused_conf)
                cv2.rectangle(out, (x1, y2 + 2), (x2, y2 + 5), (60, 60, 60), -1)
                cv2.rectangle(out, (x1, y2 + 2), (x1 + bar_fill, y2 + 5), color, -1)

        return out

    def draw_stats(self, image: np.ndarray, stats: dict) -> np.ndarray:
        out = image.copy()
        h, w = out.shape[:2]
        panel_h = 22 * (len(stats) + 1)
        cv2.rectangle(out, (0, 0), (220, panel_h), (20, 20, 20), -1)
        cv2.rectangle(out, (0, 0), (220, panel_h), (80, 80, 80), 1)
        for i, (k, v) in enumerate(stats.items()):
            text = f"{k}: {v}"
            cv2.putText(out, text, (6, 18 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 220, 255), 1, cv2.LINE_AA)
        return out


class VideoWriter:
    def __init__(self, output_path: str, fps: float = 25.0, frame_size: tuple = (1280, 720)):
        os.makedirs(str(Path(output_path).parent), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
        self.path = output_path
        logger.info(f"VideoWriter opened: {output_path} @ {fps}fps {frame_size}")

    def write(self, frame: np.ndarray):
        self.writer.write(frame)

    def release(self):
        self.writer.release()
        logger.info(f"Video saved: {self.path}")


def save_image(image: np.ndarray, output_dir: str, name: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = str(Path(output_dir) / f"{name}_{timestamp}.jpg")
    cv2.imwrite(out_path, image)
    return out_path
