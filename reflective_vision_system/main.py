"""
Phase 9: Main End-to-End Pipeline
input → detection → segmentation → CV enhancement → fusion → tracking → output

Usage:
  # Image
  python main.py --source data/raw/sample.jpg --weights models/best.pt

  # Video
  python main.py --source data/raw/sample.mp4 --weights models/best.pt

  # Webcam
  python main.py --source 0 --weights models/best.pt

  # Generate synthetic data + run demo
  python main.py --demo
"""

import cv2
import numpy as np
import argparse
import os
import time
from pathlib import Path

from src.detector import Detector
from src.segmentation import SAMSegmentor
from src.cv_enhancement import CVEnhancer
from src.fusion_engine import FusionEngine, FusedDetection
from src.tracking import SORTTracker
from src.depth import DepthEstimator
from src.visualizer import Visualizer, VideoWriter, save_image
from utils.config import load_config
from utils.logger import get_logger

logger = get_logger("main_pipeline")


class SmartVisionPipeline:
    def __init__(self, config_path: str = "configs/config.yaml",
                 det_weights: str = None,
                 sam_weights: str = None,
                 use_depth: bool = False):
        self.cfg = load_config(config_path)
        self.use_depth = use_depth

        det_cfg = self.cfg["detection"]
        seg_cfg = self.cfg["segmentation"]
        trk_cfg = self.cfg["tracking"]

        det_weights = det_weights or str(
            Path(self.cfg["paths"]["models"]) / "reflective_detector" / "weights" / "best.pt"
        )
        sam_weights = sam_weights or seg_cfg["sam_checkpoint"]

        logger.info("Initializing pipeline modules...")

        self.detector = Detector(det_weights, config_path)
        self.segmentor = SAMSegmentor(
            sam_weights,
            model_type=seg_cfg["sam_model_type"],
            device=det_cfg["device"],
        )
        self.cv_enhancer = CVEnhancer()
        self.fusion = FusionEngine(config_path)
        self.tracker = SORTTracker(
            max_age=trk_cfg["max_age"],
            min_hits=trk_cfg["min_hits"],
            iou_threshold=trk_cfg["iou_threshold"],
        )
        self.depth_estimator = DepthEstimator(
            use_simulated=self.cfg["depth"].get("use_simulated", True),
            device=det_cfg["device"],
        ) if use_depth else None
        self.visualizer = Visualizer(self.cfg)

        logger.info("Pipeline ready.")

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, dict]:
        t0 = time.perf_counter()

        # 1. Detection
        detections = self.detector.detect(frame)
        t1 = time.perf_counter()

        # 2. Segmentation
        boxes = [d["bbox"] for d in detections]
        masks = self.segmentor.segment_from_boxes(frame, boxes)
        t2 = time.perf_counter()

        # 3. CV Enhancement
        cv_features = self.cv_enhancer.analyze(frame)
        t3 = time.perf_counter()

        # 4. Depth (optional)
        depth_map = None
        if self.use_depth and self.depth_estimator:
            depth_map = self.depth_estimator.estimate(frame)

        # 5. Fusion
        fused_detections = self.fusion.fuse(detections, masks, cv_features, frame.shape)
        t4 = time.perf_counter()

        # 6. Tracking — convert FusedDetections to dict for tracker
        det_dicts = [{
            "bbox": fd.bbox,
            "class_id": fd.class_id,
            "class_name": fd.class_name,
            "confidence": fd.fused_conf,
        } for fd in fused_detections]
        tracked = self.tracker.update(det_dicts)
        t5 = time.perf_counter()

        # Map track_id → FusedDetection for visualization
        fused_map = {}
        for fd in fused_detections:
            for trk in tracked:
                if trk["class_id"] == fd.class_id:
                    fused_map[trk["track_id"]] = fd
                    fd.track_id = trk["track_id"]
                    break

        # 7. Visualize
        result = frame.copy()
        if self.use_depth and depth_map is not None:
            result = self.depth_estimator.overlay_depth(result, depth_map, alpha=0.2)
        result = self.visualizer.draw_tracks(result, tracked, fused_map)

        stats = {
            "Objects": len(tracked),
            "Det ms": f"{(t1-t0)*1000:.1f}",
            "Seg ms": f"{(t2-t1)*1000:.1f}",
            "CV ms":  f"{(t3-t2)*1000:.1f}",
            "Fuse ms":f"{(t4-t3)*1000:.1f}",
            "Trk ms": f"{(t5-t4)*1000:.1f}",
            "Total ms":f"{(t5-t0)*1000:.1f}",
        }
        result = self.visualizer.draw_stats(result, stats)

        return result, {"tracked": tracked, "fused": fused_detections,
                        "cv": cv_features, "depth": depth_map}

    def run(self, source, output_dir: str = "outputs"):
        os.makedirs(output_dir + "/images", exist_ok=True)
        os.makedirs(output_dir + "/videos", exist_ok=True)

        # Determine source type
        if str(source).isdigit():
            cap = cv2.VideoCapture(int(source))
            src_type = "webcam"
        else:
            src_path = Path(source)
            if not src_path.exists():
                logger.error(f"Source not found: {source}")
                return
            suffix = src_path.suffix.lower()
            if suffix in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
                src_type = "image"
            else:
                src_type = "video"
                cap = cv2.VideoCapture(str(src_path))

        if src_type == "image":
            frame = cv2.imread(str(source))
            result, info = self.process_frame(frame)
            out_path = save_image(result, output_dir + "/images", "result")
            logger.info(f"Saved result: {out_path}")
            logger.info(f"Tracked {len(info['tracked'])} objects")
            return result

        # Video / webcam
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vid_name = Path(str(source)).stem if src_type == "video" else "webcam"
        writer = VideoWriter(f"{output_dir}/videos/{vid_name}_result.mp4", fps, (w, h))

        frame_count = 0
        self.tracker.reset()

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                result, info = self.process_frame(frame)
                writer.write(result)
                frame_count += 1
                if frame_count % 30 == 0:
                    logger.info(f"Frame {frame_count} | Objects: {len(info['tracked'])}")

        except KeyboardInterrupt:
            logger.info("Interrupted by user.")
        finally:
            cap.release()
            writer.release()

        logger.info(f"Processed {frame_count} frames.")


def run_demo():
    """Generate synthetic frames and run the full pipeline as a demo."""
    import sys
    sys.path.insert(0, ".")
    from src.synthetic_gen import generate_image

    logger.info("Running DEMO: generating synthetic frames...")
    os.makedirs("data/demo", exist_ok=True)
    frames = []
    for i in range(5):
        img, _ = generate_image(size=640, max_objects=4)
        path = f"data/demo/demo_{i:03d}.jpg"
        cv2.imwrite(path, img)
        frames.append(path)
    logger.info(f"Generated {len(frames)} demo frames in data/demo/")
    return frames[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart Vision System Pipeline")
    parser.add_argument("--source", type=str, default=None,
                        help="Image/video path or webcam index (0)")
    parser.add_argument("--weights", type=str,
                        default="models/reflective_detector/weights/best.pt",
                        help="YOLO weights path")
    parser.add_argument("--sam_weights", type=str,
                        default="models/sam_vit_h_4b8939.pth")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--output", type=str, default="outputs")
    parser.add_argument("--depth", action="store_true", help="Enable depth estimation")
    parser.add_argument("--demo", action="store_true",
                        help="Generate synthetic demo and run pipeline")
    args = parser.parse_args()

    if args.demo:
        source = run_demo()
    elif args.source is None:
        parser.error("Provide --source or use --demo")
    else:
        source = args.source

    pipeline = SmartVisionPipeline(
        config_path=args.config,
        det_weights=args.weights,
        sam_weights=args.sam_weights,
        use_depth=args.depth,
    )
    pipeline.run(source, args.output)
