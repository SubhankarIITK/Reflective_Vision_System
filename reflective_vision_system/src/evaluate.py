"""
Phase 3: YOLOv8 Evaluation Script
Run: python src/evaluate.py --weights models/best.pt --data configs/dataset.yaml
"""

import argparse
import json
from pathlib import Path
from ultralytics import YOLO
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.logger import get_logger

logger = get_logger("evaluate")


def evaluate(weights: str, data: str = "configs/dataset.yaml",
             imgsz: int = 640, device: str = "cuda"):
    model = YOLO(weights)
    logger.info(f"Evaluating: {weights}")

    metrics = model.val(data=data, imgsz=imgsz, device=device, verbose=True)

    results = {
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "per_class_AP": {
            str(i): float(v) for i, v in enumerate(metrics.box.ap50)
        }
    }

    out_path = Path("outputs/logs/eval_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"mAP@0.5     : {results['mAP50']:.4f}")
    logger.info(f"mAP@0.5:0.95: {results['mAP50_95']:.4f}")
    logger.info(f"Precision   : {results['precision']:.4f}")
    logger.info(f"Recall      : {results['recall']:.4f}")
    logger.info(f"Results saved to {out_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", default="configs/dataset.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    evaluate(args.weights, args.data, args.imgsz, args.device)
