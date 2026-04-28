"""
Phase 3: YOLOv8 Training Script
Run: python src/train.py --config configs/config.yaml
"""

import argparse
import os
from pathlib import Path
from ultralytics import YOLO
import yaml
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.config import load_config
from utils.logger import get_logger

logger = get_logger("train")


def train(config_path: str = "configs/config.yaml"):
    cfg = load_config(config_path)
    det = cfg["detection"]
    paths = cfg["paths"]

    os.makedirs(paths["models"], exist_ok=True)
    os.makedirs(paths["logs"], exist_ok=True)

    logger.info(f"Loading base model: {det['model']}")
    model = YOLO(det["model"])

    logger.info("Starting training...")
    results = model.train(
        data="configs/dataset.yaml",
        epochs=det["epochs"],
        imgsz=det["imgsz"],
        batch=cfg["data"]["batch_size"],
        device=det["device"],
        workers=cfg["data"]["num_workers"],
        project=paths["models"],
        name="reflective_detector",
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        flipud=0.1,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.1,
        conf=det["conf_threshold"],
        iou=det["iou_threshold"],
        plots=True,
        save=True,
        save_period=10,
        val=True,
        verbose=True,
    )

    best_weights = Path(paths["models"]) / "reflective_detector" / "weights" / "best.pt"
    logger.info(f"Training complete. Best weights: {best_weights}")
    return str(best_weights)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    train(args.config)
