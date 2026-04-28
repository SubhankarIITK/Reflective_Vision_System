"""
Phase 7: Temporal Object Tracking
SORT-based tracker assigning unique IDs across frames.
"""

import numpy as np
from filterpy.kalman import KalmanFilter
from dataclasses import dataclass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.logger import get_logger

logger = get_logger("tracking")


def iou(bb_a: np.ndarray, bb_b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = bb_a
    bx1, by1, bx2, by2 = bb_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def linear_assignment(cost_matrix: np.ndarray) -> np.ndarray:
    try:
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(row_ind, col_ind)))
    except ImportError:
        return np.empty((0, 2), dtype=int)


class KalmanBoxTracker:
    _id_counter = 0

    def __init__(self, bbox: list[int], class_id: int, class_name: str):
        KalmanBoxTracker._id_counter += 1
        self.id = KalmanBoxTracker._id_counter
        self.class_id = class_id
        self.class_name = class_name
        self.hits = 0
        self.no_loss_count = 0
        self.age = 0

        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        # State: [x, y, s, r, dx, dy, ds] where s=area, r=aspect
        self.kf.F = np.array([
            [1,0,0,0,1,0,0],
            [0,1,0,0,0,1,0],
            [0,0,1,0,0,0,1],
            [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1],
        ], dtype=np.float32)
        self.kf.H = np.array([
            [1,0,0,0,0,0,0],
            [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0],
        ], dtype=np.float32)
        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        self.kf.x[:4] = self._bbox_to_z(bbox)

    @staticmethod
    def _bbox_to_z(bbox: list[int]) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        s = (x2 - x1) * (y2 - y1)
        r = (x2 - x1) / float(y2 - y1) if (y2 - y1) > 0 else 1.0
        return np.array([[cx], [cy], [s], [r]], dtype=np.float32)

    @staticmethod
    def _z_to_bbox(z: np.ndarray) -> list[int]:
        cx, cy, s, r = float(z[0]), float(z[1]), float(z[2]), float(z[3])
        s = max(s, 1.0)
        r = max(r, 0.1)
        w = np.sqrt(s * r)
        h = s / w
        return [int(cx - w/2), int(cy - h/2), int(cx + w/2), int(cy + h/2)]

    def predict(self) -> list[int]:
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] = 0.0
        self.kf.predict()
        self.age += 1
        self.no_loss_count += 1
        return self._z_to_bbox(self.kf.x[:4])

    def update(self, bbox: list[int]):
        self.hits += 1
        self.no_loss_count = 0
        self.kf.update(self._bbox_to_z(bbox))

    def get_state(self) -> list[int]:
        return self._z_to_bbox(self.kf.x[:4])


class SORTTracker:
    def __init__(self, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers: list[KalmanBoxTracker] = []
        self.frame_count = 0

    def reset(self):
        self.trackers = []
        self.frame_count = 0
        KalmanBoxTracker._id_counter = 0

    def update(self, detections: list[dict]) -> list[dict]:
        self.frame_count += 1

        predicted_boxes = []
        for t in self.trackers:
            pred = t.predict()
            predicted_boxes.append(pred)

        # Match detections to existing trackers
        matched, unmatched_dets, unmatched_trks = self._associate(
            detections, predicted_boxes
        )

        for det_idx, trk_idx in matched:
            self.trackers[trk_idx].update(detections[det_idx]["bbox"])

        # Create new trackers for unmatched detections
        for i in unmatched_dets:
            det = detections[i]
            self.trackers.append(
                KalmanBoxTracker(det["bbox"], det.get("class_id", 0), det.get("class_name", ""))
            )

        # Remove dead trackers
        self.trackers = [t for t in self.trackers if t.no_loss_count <= self.max_age]

        # Return active tracks
        results = []
        for t in self.trackers:
            if t.hits >= self.min_hits or self.frame_count <= self.min_hits:
                state = t.get_state()
                results.append({
                    "bbox": state,
                    "track_id": t.id,
                    "class_id": t.class_id,
                    "class_name": t.class_name,
                    "age": t.age,
                    "hits": t.hits,
                })
        return results

    def _associate(self, detections: list[dict], predicted_boxes: list[list[int]]):
        if not predicted_boxes:
            return [], list(range(len(detections))), []
        if not detections:
            return [], [], list(range(len(predicted_boxes)))

        iou_matrix = np.zeros((len(detections), len(predicted_boxes)))
        for d_i, det in enumerate(detections):
            for t_i, pred in enumerate(predicted_boxes):
                iou_matrix[d_i, t_i] = iou(np.array(det["bbox"]), np.array(pred))

        cost_matrix = 1.0 - iou_matrix
        matched_indices = linear_assignment(cost_matrix)

        matched, unmatched_dets, unmatched_trks = [], [], []

        for d_i in range(len(detections)):
            if d_i not in matched_indices[:, 0]:
                unmatched_dets.append(d_i)

        for t_i in range(len(predicted_boxes)):
            if t_i not in matched_indices[:, 1]:
                unmatched_trks.append(t_i)

        for d_i, t_i in matched_indices:
            if iou_matrix[d_i, t_i] < self.iou_threshold:
                unmatched_dets.append(d_i)
                unmatched_trks.append(t_i)
            else:
                matched.append((d_i, t_i))

        return matched, unmatched_dets, unmatched_trks
