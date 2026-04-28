"""
Phase 3: YOLOv8 Inference Script
Run: python src/detector.py --source data/raw/sample.jpg --weights models/best.pt
"""

import cv2
import numpy as np
import argparse
import os
from pathlib import Path
from ultralytics import YOLO
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.config import load_config
from utils.logger import get_logger

logger = get_logger("detector")

CLASS_NAMES = [
    "glass_bottle", "plastic_container", "reflective_surface",
    "semi_transparent_object", "glass_cup", "plastic_bag", "mirror", "window"
]

CLASS_COLORS = [
    (0, 120, 255), (0, 200, 100), (255, 180, 0), (200, 0, 200),
    (0, 200, 200), (255, 100, 0), (100, 100, 255), (50, 200, 50)
]


class Detector:
    def __init__(self, weights: str, config_path: str = "configs/config.yaml"):
        cfg = load_config(config_path)
        det = cfg["detection"]
        self.conf = det["conf_threshold"]
        self.iou = det["iou_threshold"]
        self.device = det["device"]
        self.model = YOLO(weights)
        logger.info(f"Detector loaded: {weights}")

    def detect(self, image: np.ndarray) -> list[dict]:
        results = self.model.predict(
            source=image,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cls_id = int(box.cls[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                detections.append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "class_id": cls_id,
                    "class_name": CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id),
                    "confidence": conf,
                })
        return detections

    def draw_detections(self, image: np.ndarray, detections: list[dict]) -> np.ndarray:
        out = image.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cls_id = det["class_id"]
            color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]
            label = f"{det['class_name']} {det['confidence']:.2f}"
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(out, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        return out


def run_inference(source: str, weights: str, output_dir: str = "outputs/images"):
    detector = Detector(weights)
    os.makedirs(output_dir, exist_ok=True)

    src = Path(source)
    if src.is_file():
        image = cv2.imread(str(src))
        detections = detector.detect(image)
        logger.info(f"Detected {len(detections)} objects in {src.name}")
        for d in detections:
            logger.info(f"  {d['class_name']}: {d['confidence']:.3f} @ {d['bbox']}")
        result = detector.draw_detections(image, detections)
        out_path = Path(output_dir) / f"det_{src.name}"
        cv2.imwrite(str(out_path), result)
        logger.info(f"Saved: {out_path}")
    else:
        logger.error(f"Source not found: {source}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--weights", default="models/reflective_detector/weights/best.pt")
    parser.add_argument("--output", default="outputs/images")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    run_inference(args.source, args.weights, args.output)
