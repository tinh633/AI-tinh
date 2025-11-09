#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_drowsy.py
----------------
Build YOLO dataset + train model detect open_eye / closed_eye cho drowsy detection.

Cấu trúc thư mục (như bạn yêu cầu):

DATA_ROOT=/home/gess/Documents/sub/TrainModel/
OPEN_EYES=/home/gess/Documents/sub/TrainModel/Open_Eyes/
CLOSED_EYES=/home/gess/Documents/sub/TrainModel/Closed_Eyes/
VIDEOS=/home/gess/Documents/sub/TrainModel/train_vid/
NEW_REAL=/home/gess/Documents/sub/TrainModel/Webcam/
YOLO_DATASET=/home/gess/Documents/sub/TrainModel/YOLO_Dataset/
OUTPUT=/home/gess/Documents/sub/TrainModel/outputs
MODEL_DIR=/home/gess/Documents/sub/TrainModel/models
TMP_DIR=/home/gess/Documents/sub/TrainModel/tmp

Cách dùng (ví dụ):
  python train_drowsy.py --prepare        # build YOLO_Dataset (auto-label)
  python train_drowsy.py --train          # train model, lưu best_drowsy.pt
  python train_drowsy.py --export         # export ONNX

Có thể combine:
  python train_drowsy.py --prepare --train --export
"""

from __future__ import annotations
import os
import sys
import json
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Mediapipe cho auto-label mắt
try:
    import mediapipe as mp  # type: ignore
    HAS_MEDIAPIPE = True
except Exception:
    mp = None  # type: ignore
    HAS_MEDIAPIPE = False

from ultralytics import YOLO

SEED = 42
np.random.seed(SEED)

# --------------------- ĐƯỜNG DẪN CỐ ĐỊNH CHO BẠN --------------------- #

DATA_ROOT = Path("/home/gess/Documents/sub/TrainModel/").resolve()
OPEN_EYES = Path("/home/gess/Documents/sub/TrainModel/Open_Eyes/").resolve()
CLOSED_EYES = Path("/home/gess/Documents/sub/TrainModel/Closed_Eyes/").resolve()
VIDEOS = Path("/home/gess/Documents/sub/TrainModel/train_vid/").resolve()
NEW_REAL = Path("/home/gess/Documents/sub/TrainModel/Webcam/").resolve()  # ảnh real-life

YOLO_DATASET = Path("/home/gess/Documents/sub/TrainModel/YOLO_Dataset/").resolve()
OUTPUT = Path("/home/gess/Documents/sub/TrainModel/outputs/").resolve()
MODEL_DIR = Path("/home/gess/Documents/sub/TrainModel/models/").resolve()
TMP_DIR = Path("/home/gess/Documents/sub/TrainModel/tmp/").resolve()

for p in [OPEN_EYES, CLOSED_EYES, VIDEOS, NEW_REAL, YOLO_DATASET, OUTPUT, MODEL_DIR, TMP_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------- #

IMG_EXTS = {".jpg", ".jpeg", ".png"}
VID_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

# Từ khoá infer nhãn open/closed từ tên file (cho VIDEO + NEW_REAL)
KW_OPEN = ["open", "mo"]
KW_CLOSED = ["closed", "dong", "nham", "sleep", "drowsy"]

CLASS_MAP = {"open_eye": 0, "closed_eye": 1}
CLASS_NAMES = ["open_eye", "closed_eye"]


def natural_key(s: str) -> list:
    import re
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', str(s))]


def write_yolo_label(label_path: Path, cls_id: int,
                     xyxy: Tuple[int, int, int, int],
                     img_w: int, img_h: int) -> bool:
    x1, y1, x2, y2 = xyxy
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_w - 1, x2), min(img_h - 1, y2)
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    if w <= 1 or h <= 1:
        return False
    cx = x1 + w / 2
    cy = y1 + h / 2
    nx, ny, nw, nh = cx / img_w, cy / img_h, w / img_w, h / img_h
    label_path.write_text(f"{cls_id} {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}\n")
    return True


# -------------------- ROI bằng Mediapipe FaceMesh -------------------- #

class ROIExtractor:
    """Dùng MediaPipe FaceMesh để lấy bbox mắt (trái/phải)."""

    def __init__(self):
        if not HAS_MEDIAPIPE:
            raise RuntimeError(
                "Chưa cài mediapipe. Hãy chạy:\n  pip install mediapipe"
            )
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, refine_landmarks=True
        )
        # Landmarks bao quanh mắt trái/phải
        self.left_ids = [33, 7, 163, 144, 145, 153, 154, 155,
                         133, 173, 157, 158, 159, 160, 161, 246]
        self.right_ids = [263, 249, 390, 373, 374, 380, 381, 382,
                          362, 398, 384, 385, 386, 387, 388, 466]

    def infer_eye_boxes(self, img: np.ndarray) -> List[Tuple[int, int, int, int]]:
        h, w = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = self.face_mesh.process(img_rgb)
        if not res.multi_face_landmarks:
            return []
        lm = res.multi_face_landmarks[0]
        boxes = []
        for ids in (self.left_ids, self.right_ids):
            xs = [int(lm.landmark[i].x * w) for i in ids]
            ys = [int(lm.landmark[i].y * h) for i in ids]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            side = int(1.4 * max(x2 - x1, y2 - y1))  # pad 40%
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            x1, y1 = max(0, cx - side // 2), max(0, cy - side // 2)
            x2, y2 = min(w - 1, cx + side // 2), min(h - 1, cy + side // 2)
            if (x2 - x1) >= 5 and (y2 - y1) >= 5:
                boxes.append((x1, y1, x2, y2))
        return boxes


# ---------------------- Build YOLO Dataset ---------------------- #

@dataclass
class DatasetBuilder:
    img_size: int = 640

    def __post_init__(self):
        self.roi: Optional[ROIExtractor] = None
        (YOLO_DATASET / "images" / "train").mkdir(parents=True, exist_ok=True)
        (YOLO_DATASET / "images" / "val").mkdir(parents=True, exist_ok=True)
        (YOLO_DATASET / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (YOLO_DATASET / "labels" / "val").mkdir(parents=True, exist_ok=True)
        TMP_DIR.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _infer_label_from_name(name: str) -> Optional[int]:
        low = name.lower()
        if any(k in low for k in KW_OPEN):
            return CLASS_MAP["open_eye"]
        if any(k in low for k in KW_CLOSED):
            return CLASS_MAP["closed_eye"]
        return None

    def _collect_images(self) -> List[Tuple[Path, int]]:
        pairs: List[Tuple[Path, int]] = []
        # open_eyes
        for p in sorted(OPEN_EYES.glob("**/*.*"), key=natural_key):
            if p.suffix.lower() in IMG_EXTS:
                pairs.append((p, CLASS_MAP["open_eye"]))
        # closed_eyes
        for p in sorted(CLOSED_EYES.glob("**/*.*"), key=natural_key):
            if p.suffix.lower() in IMG_EXTS:
                pairs.append((p, CLASS_MAP["closed_eye"]))
        # NEW_REAL: infer label từ tên file
        if NEW_REAL.exists():
            for p in sorted(NEW_REAL.glob("**/*.*"), key=natural_key):
                if p.suffix.lower() not in IMG_EXTS:
                    continue
                lbl = self._infer_label_from_name(p.stem)
                if lbl is not None:
                    pairs.append((p, lbl))
        return pairs

    def _frames_from_videos(self) -> List[Tuple[Path, int]]:
        if not VIDEOS.exists():
            return []
        results: List[Tuple[Path, int]] = []
        for v in sorted(VIDEOS.glob("**/*.*"), key=natural_key):
            if v.suffix.lower() not in VID_EXTS:
                continue
            lbl = self._infer_label_from_name(v.stem)
            if lbl is None:
                continue
            limit = 80 if lbl == CLASS_MAP["closed_eye"] else 40
            cap = cv2.VideoCapture(str(v))
            if not cap.isOpened():
                print(f"⚠️ Không mở được video: {v}")
                continue
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            step = max(frame_count // max(1, limit), 1)
            idx, saved = 0, 0
            while True:
                ok = cap.grab()
                if not ok:
                    break
                if idx % step == 0:
                    ok, frame = cap.retrieve()
                    if not ok or frame is None:
                        break
                    fp = TMP_DIR / f"{v.stem}_{idx}.jpg"
                    cv2.imwrite(str(fp), frame)
                    results.append((fp, lbl))
                    saved += 1
                    if saved >= limit:
                        break
                idx += 1
            cap.release()
        return results

    def _lazy_roi(self):
        if self.roi is None:
            self.roi = ROIExtractor()
    def _auto_label_one(self, src: Path, cls_id: int,
                        out_img: Path, out_lbl: Path) -> bool:
        img = cv2.imread(str(src))
        if img is None:
            return False

        cv2.imwrite(str(out_img), img)
        h, w = img.shape[:2]

        self._lazy_roi()
        boxes = self.roi.infer_eye_boxes(img)

        ok_any = False
        if boxes:
            # dùng bbox từ Mediapipe nếu có
            for box in boxes:
                ok_any |= write_yolo_label(out_lbl, cls_id, box, w, h)
        else:
            # ❗ Fallback: dùng cả ảnh làm bbox (ảnh crop mắt/face)
            full_box = (0, 0, w - 1, h - 1)
            ok_any = write_yolo_label(out_lbl, cls_id, full_box, w, h)

        if not ok_any:
            if out_img.exists():
                out_img.unlink(missing_ok=True)
            if out_lbl.exists():
                out_lbl.unlink(missing_ok=True)
            return False

        return True


    def build(self, val_ratio: float = 0.2):
        if not HAS_MEDIAPIPE:
            raise RuntimeError("Không có mediapipe -> không thể auto-label. "
                               "Cài mediapipe trước.")

        print("🔄 Gom ảnh + frame video...")
        pairs = self._collect_images()
        pairs += self._frames_from_videos()

        if len(pairs) == 0:
            print("❌ Không tìm thấy ảnh/video để build dataset.")
            return

        labels = [c for _, c in pairs]
        stratify = labels if len(set(labels)) > 1 else None
        train_pairs, val_pairs = train_test_split(
            pairs, test_size=val_ratio, random_state=SEED, stratify=stratify
        )

        stats = {"train_kept": 0, "train_skipped": 0,
                 "val_kept": 0, "val_skipped": 0}

        for split, items in [("train", train_pairs), ("val", val_pairs)]:
            img_dir = YOLO_DATASET / "images" / split
            lbl_dir = YOLO_DATASET / "labels" / split
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)

            for src, cls_id in items:
                out_img = img_dir / f"{src.stem}.jpg"
                out_lbl = lbl_dir / f"{src.stem}.txt"
                try:
                    ok = self._auto_label_one(src, cls_id, out_img, out_lbl)
                    if ok:
                        stats[f"{split}_kept"] += 1
                    else:
                        stats[f"{split}_skipped"] += 1
                except Exception as e:
                    stats[f"{split}_skipped"] += 1
                    print(f"⚠️ Lỗi auto-label {src}: {e}")
                    for p in [out_img, out_lbl]:
                        if p.exists():
                            p.unlink(missing_ok=True)

        # Ghi data.yaml cho YOLO
        data_yaml = YOLO_DATASET / "data.yaml"
        data_yaml.write_text(
            "\n".join([
                f"path: {YOLO_DATASET}",
                "train: images/train",
                "val: images/val",
                f"nc: {len(CLASS_NAMES)}",
                f"names: {CLASS_NAMES}",
            ]) + "\n"
        )
        print(f"✅ Build xong YOLO_Dataset → {YOLO_DATASET}")
        print(f"📈 Stats: {json.dumps(stats, ensure_ascii=False)}")

        # Xoá frame tạm
        for f in TMP_DIR.glob("*.jpg"):
            f.unlink(missing_ok=True)


# --------------------------- Train YOLO --------------------------- #

@dataclass
class Trainer:
    weights: str = "yolov8n.pt"
    imgsz: int = 768  # lớn chút cho mắt rõ

    def __post_init__(self):
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        print(f"🚀 Loading model: {self.weights}")
        self.model = YOLO(self.weights)

    def train(self, epochs: int = 100, batch: int = 8, workers: int = 4):
        data_yaml = YOLO_DATASET / "data.yaml"
        assert data_yaml.exists(), "Chưa có data.yaml, hãy chạy --prepare trước."

        # chọn device
        device = "0"

        args = dict(
            data=str(data_yaml),
            imgsz=self.imgsz,
            epochs=epochs,
            batch=batch,
            workers=workers,
            device=device,
            seed=SEED,
            project=str(OUTPUT),
            name="drowsy_det",
            cos_lr=True,
            lr0=0.002,
            lrf=0.12,
            optimizer="SGD",
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3.0,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            amp=True,
            patience=30,
            cache="ram",
            mosaic=0.3,
            mixup=0.0,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=5.0,
            translate=0.05,
            scale=0.2,
            shear=1.0,
            erasing=0.0,
            box=7.5,
            cls=0.5,
            save_json=False,
            val=True,
        )
        print("📊 Train args (tóm tắt):")
        print(json.dumps({k: v for k, v in args.items() if k != "data"}, indent=2))

        results = self.model.train(**args)
        save_dir = Path(results.save_dir)
        best = save_dir / "weights" / "best.pt"
        if best.exists():
            dst = MODEL_DIR / "best_drowsy.pt"
            dst.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy2(best, dst)
            print(f"✅ Lưu best model → {dst}")
        else:
            print("⚠️ Không tìm thấy best.pt sau train.")

    def export(self, fmt: str = "onnx"):
        src = MODEL_DIR / "best_drowsy.pt"
        assert src.exists(), "Không tìm thấy best_drowsy.pt, hãy train trước."
        m = YOLO(str(src))
        file = m.export(format=fmt)
        import shutil
        dst = MODEL_DIR / f"drowsy_export.{fmt}"
        shutil.copy2(file, dst)
        print(f"✅ Export → {dst}")


# --------------------------- CLI --------------------------- #

def parse_args():
    ap = argparse.ArgumentParser(description="Train drowsy detection (open/closed eyes)")
    ap.add_argument("--prepare", action="store_true", help="Build YOLO_Dataset (auto-label)")
    ap.add_argument("--train", action="store_true", help="Train YOLO model")
    ap.add_argument("--export", action="store_true", help="Export best model (ONNX)")
    ap.add_argument("--imgsz", type=int, default=768, help="Image size train")
    return ap.parse_args()


def main():
    args = parse_args()

    print("📂 Paths:")
    print(json.dumps({
        "DATA_ROOT": str(DATA_ROOT),
        "OPEN_EYES": str(OPEN_EYES),
        "CLOSED_EYES": str(CLOSED_EYES),
        "VIDEOS": str(VIDEOS),
        "NEW_REAL": str(NEW_REAL),
        "YOLO_DATASET": str(YOLO_DATASET),
        "OUTPUT": str(OUTPUT),
        "MODEL_DIR": str(MODEL_DIR),
        "TMP_DIR": str(TMP_DIR),
    }, indent=2, ensure_ascii=False))

    if args.prepare:
        builder = DatasetBuilder(img_size=args.imgsz)
        builder.build()

    if args.train:
        trainer = Trainer(imgsz=args.imgsz)
        trainer.train()

    if args.export:
        trainer = Trainer(imgsz=args.imgsz)
        trainer.export(fmt="onnx")


if __name__ == "__main__":
    main()
