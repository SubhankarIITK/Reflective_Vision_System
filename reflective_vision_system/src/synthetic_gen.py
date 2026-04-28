"""
Synthetic data generator for reflective/transparent objects.
Generates labeled YOLO-format training images when real datasets are unavailable.
Run: python src/synthetic_gen.py --num_images 500 --output data/processed
"""

import cv2
import numpy as np
import argparse
import os
import random
from pathlib import Path
from tqdm import tqdm

CLASS_NAMES = [
    "glass_bottle", "plastic_container", "reflective_surface",
    "semi_transparent_object", "glass_cup", "plastic_bag", "mirror", "window"
]


def random_color(alpha=False):
    c = [random.randint(100, 255) for _ in range(3)]
    return (*c, random.randint(80, 180)) if alpha else tuple(c)


def draw_glass_bottle(img, cx, cy, w, h):
    overlay = img.copy()
    color = (random.randint(100, 200), random.randint(150, 230), random.randint(180, 255))
    pts = np.array([
        [cx, cy - h // 2], [cx + w // 6, cy - h // 4],
        [cx + w // 3, cy + h // 2], [cx - w // 3, cy + h // 2],
        [cx - w // 6, cy - h // 4]
    ], dtype=np.int32)
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)
    # Highlight
    cv2.line(img, (cx - w // 8, cy - h // 3), (cx - w // 8, cy + h // 3),
             (255, 255, 255), max(1, w // 12))


def draw_glass_cup(img, cx, cy, w, h):
    overlay = img.copy()
    color = (random.randint(180, 230), random.randint(200, 240), random.randint(220, 255))
    cv2.ellipse(overlay, (cx, cy + h // 2), (w // 2, h // 8), 0, 0, 180, color, -1)
    pts = np.array([
        [cx - w // 2, cy + h // 2], [cx - w // 3, cy - h // 2],
        [cx + w // 3, cy - h // 2], [cx + w // 2, cy + h // 2]
    ], dtype=np.int32)
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)
    cv2.line(img, (cx - w // 5, cy - h // 3), (cx - w // 5, cy + h // 3),
             (255, 255, 255), max(1, w // 15))


def draw_reflective_surface(img, cx, cy, w, h):
    overlay = img.copy()
    cv2.rectangle(overlay, (cx - w // 2, cy - h // 2), (cx + w // 2, cy + h // 2),
                  (200, 200, 220), -1)
    # Gradient shimmer
    for i in range(0, w, max(1, w // 8)):
        alpha = random.uniform(0.1, 0.5)
        x = cx - w // 2 + i
        cv2.line(overlay, (x, cy - h // 2), (x + h // 4, cy + h // 2),
                 (255, 255, 255), max(1, w // 20))
    cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)


def draw_plastic_container(img, cx, cy, w, h):
    overlay = img.copy()
    color = tuple(random.randint(100, 220) for _ in range(3))
    cv2.rectangle(overlay, (cx - w // 2, cy - h // 2), (cx + w // 2, cy + h // 2), color, -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    cv2.rectangle(img, (cx - w // 2, cy - h // 2), (cx + w // 2, cy + h // 2),
                  (255, 255, 255), 2)


def draw_plastic_bag(img, cx, cy, w, h):
    overlay = img.copy()
    pts = np.array([
        [cx, cy - h // 2], [cx + w // 2, cy],
        [cx + w // 3, cy + h // 2], [cx - w // 3, cy + h // 2],
        [cx - w // 2, cy]
    ], dtype=np.int32)
    cv2.fillPoly(overlay, [pts], (220, 230, 240))
    cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
    cv2.polylines(img, [pts], True, (180, 190, 200), 1)


def draw_mirror(img, cx, cy, w, h):
    overlay = img.copy()
    cv2.rectangle(overlay, (cx - w // 2, cy - h // 2), (cx + w // 2, cy + h // 2),
                  (230, 230, 240), -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
    cv2.rectangle(img, (cx - w // 2, cy - h // 2), (cx + w // 2, cy + h // 2),
                  (160, 160, 180), 3)


def draw_window(img, cx, cy, w, h):
    overlay = img.copy()
    cv2.rectangle(overlay, (cx - w // 2, cy - h // 2), (cx + w // 2, cy + h // 2),
                  (180, 210, 230), -1)
    cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)
    cv2.rectangle(img, (cx - w // 2, cy - h // 2), (cx + w // 2, cy + h // 2),
                  (120, 140, 160), 3)
    cv2.line(img, (cx, cy - h // 2), (cx, cy + h // 2), (120, 140, 160), 2)
    cv2.line(img, (cx - w // 2, cy), (cx + w // 2, cy), (120, 140, 160), 2)


def draw_semi_transparent(img, cx, cy, w, h):
    overlay = img.copy()
    color = tuple(random.randint(150, 220) for _ in range(3))
    cv2.ellipse(overlay, (cx, cy), (w // 2, h // 2), 0, 0, 360, color, -1)
    cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)


DRAW_FUNCS = [
    draw_glass_bottle, draw_plastic_container, draw_reflective_surface,
    draw_semi_transparent, draw_glass_cup, draw_plastic_bag, draw_mirror, draw_window
]


def generate_background(size=640):
    bg_type = random.choice(["solid", "gradient", "noise", "table"])
    img = np.zeros((size, size, 3), dtype=np.uint8)
    if bg_type == "solid":
        img[:] = tuple(random.randint(30, 200) for _ in range(3))
    elif bg_type == "gradient":
        for i in range(size):
            val = int(50 + 150 * i / size)
            img[i, :] = (val, val + 20, val + 40)
    elif bg_type == "noise":
        img = np.random.randint(60, 180, (size, size, 3), dtype=np.uint8)
        img = cv2.GaussianBlur(img, (21, 21), 0)
    elif bg_type == "table":
        img[:] = (random.randint(80, 160),) * 3
        for i in range(0, size, random.randint(20, 60)):
            cv2.line(img, (0, i), (size, i), (img[0, 0, 0] + 20,) * 3, 1)
    return img


def generate_image(size=640, max_objects=5):
    img = generate_background(size)
    labels = []
    n_objects = random.randint(1, max_objects)

    for _ in range(n_objects):
        cls_id = random.randint(0, 7)
        w = random.randint(size // 8, size // 3)
        h = random.randint(size // 8, size // 3)
        cx = random.randint(w // 2, size - w // 2)
        cy = random.randint(h // 2, size - h // 2)

        DRAW_FUNCS[cls_id](img, cx, cy, w, h)

        xc = cx / size
        yc = cy / size
        wn = w / size
        hn = h / size
        xc = np.clip(xc, 0, 1)
        yc = np.clip(yc, 0, 1)
        wn = np.clip(wn, 0, 1)
        hn = np.clip(hn, 0, 1)
        labels.append(f"{cls_id} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")

    return img, labels


def generate_dataset(output_dir: str, num_images: int = 500, splits=(0.8, 0.1, 0.1), size=640):
    output_dir = Path(output_dir)
    split_names = ["train", "val", "test"]
    counts = [int(num_images * s) for s in splits]
    counts[-1] = num_images - sum(counts[:-1])

    for split, count in zip(split_names, counts):
        img_dir = output_dir / "images" / split
        lbl_dir = output_dir / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for i in tqdm(range(count), desc=f"Generating {split}"):
            img, labels = generate_image(size=size)
            fname = f"synthetic_{split}_{i:05d}"
            cv2.imwrite(str(img_dir / f"{fname}.jpg"), img)
            with open(lbl_dir / f"{fname}.txt", "w") as f:
                f.write("\n".join(labels))

    print(f"\nDone. Dataset saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_images", type=int, default=500)
    parser.add_argument("--output", type=str, default="data/processed")
    parser.add_argument("--size", type=int, default=640)
    args = parser.parse_args()
    generate_dataset(args.output, args.num_images, size=args.size)
