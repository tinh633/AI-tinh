#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np

# ====== Đường dẫn gốc ======
DATA_ROOT = Path("/home/gess/Documents/sub/TrainModel")
OPEN_EYES = DATA_ROOT / "Open_Eyes"
CLOSED_EYES = DATA_ROOT / "Closed_Eyes"
VIDEOS = DATA_ROOT / "train_vid"
NEW_REAL = DATA_ROOT / "Webcam"
YOLO_DATASET = DATA_ROOT / "YOLO_Dataset"
OUTPUT = DATA_ROOT / "outputs"
MODEL_DIR = DATA_ROOT / "models"
TMP_DIR = DATA_ROOT / "tmp"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ====== YOLO ======
try:
    from ultralytics import YOLO
except Exception as e:
    YOLO = None  # sẽ báo lỗi rõ ràng khi train

# ====== MediaPipe (auto-label) ======
# Sử dụng FaceMesh để ước lượng độ mở mắt (EAR-like).
# Landmark index cho mắt phải/trái (MediaPipe FaceMesh).
LEFT_EYE_IDX  = [33, 160, 158, 133, 153, 144]   # xấp xỉ
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]  # xấp xỉ

def try_import_mediapipe():
    try:
        import mediapipe as mp
        return mp
    except Exception:
        return None

def eye_aspect_ratio(pts: np.ndarray) -> float:
    """EAR gần đúng từ 6 điểm (shape: (6,2))."""
    # D = khoảng cách dọc, W = khoảng cách ngang
    # (p2,p6), (p3,p5) là dọc; (p1,p4) là ngang
    def dist(a, b): return np.linalg.norm(a - b)
    p1, p2, p3, p4, p5, p6 = pts
    vertical = dist(p2, p6) + dist(p3, p5)
    horizontal = dist(p1, p4) + 1e-6
    return float(vertical / horizontal)

def crop_eye_bbox(img: np.ndarray, pts: np.ndarray, margin: float=0.35) -> Tuple[int,int,int,int]:
    """Lấy bbox quanh 6 điểm mắt, mở rộng margin theo tỉ lệ."""
    h, w = img.shape[:2]
    x_min = max(0, int(np.min(pts[:, 0])))
    y_min = max(0, int(np.min(pts[:, 1])))
    x_max = min(w-1, int(np.max(pts[:, 0])))
    y_max = min(h-1, int(np.max(pts[:, 1])))
    cx, cy = (x_min + x_max)//2, (y_min + y_max)//2
    bw, bh = x_max - x_min, y_max - y_min
    bw = int(bw * (1.0 + margin))
    bh = int(bh * (1.0 + margin))
    x1 = max(0, cx - bw//2); y1 = max(0, cy - bh//2)
    x2 = min(w-1, cx + bw//2); y2 = min(h-1, cy + bh//2)
    return x1, y1, x2, y2

def autolabel_image(path: Path, mp) -> Optional[Tuple[List[Tuple[int,int,int,int,int]], dict]]:
    """
    Auto-label 1 ảnh:
    - Trả về: (list_bbox_label, debug) hoặc None nếu không tìm được mắt.
    - Mỗi phần tử trong list_bbox_label: (cls_id, x1, y1, x2, y2) với cls: 0=open_eye, 1=closed_eye
    """
    img = cv2.imread(str(path))
    if img is None:
        return None
    h, w = img.shape[:2]

    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True, max_num_faces=1) as fm:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = fm.process(rgb)
        if not res.multi_face_landmarks:
            return None
        lm = res.multi_face_landmarks[0]
        # Lấy điểm mắt (left/right)
        coords = []
        for idx in LEFT_EYE_IDX + RIGHT_EYE_IDX:
            pt = lm.landmark[idx]
            coords.append([pt.x * w, pt.y * h])
        coords = np.array(coords, dtype=np.float32)

        left_pts  = coords[:6]
        right_pts = coords[6:]

        left_ear  = eye_aspect_ratio(left_pts)
        right_ear = eye_aspect_ratio(right_pts)
        # Ngưỡng auto-label (điều chỉnh nếu cần)
        # > 0.25 mở; < 0.21 nhắm; vùng giữa: bỏ qua để tránh gán nhãn mơ hồ
        TH_OPEN = 0.25
        TH_CLOSED = 0.21

        bboxes = []
        dbg = {"left_ear": left_ear, "right_ear": right_ear}

        for ear, pts in [(left_ear, left_pts), (right_ear, right_pts)]:
            if ear >= TH_OPEN:
                cls_id = 0  # open_eye
            elif ear <= TH_CLOSED:
                cls_id = 1  # closed_eye
            else:
                # vùng mơ hồ -> bỏ qua mắt này
                continue
            x1, y1, x2, y2 = crop_eye_bbox(img, pts)
            bboxes.append((cls_id, x1, y1, x2, y2))

        if not bboxes:
            return None
        return bboxes, dbg

# ========= Ghi nhãn YOLO =========
MIN_EYE_RATIO = 0.035  # bbox phải >3.5% chiều rộng ảnh
def write_yolo_line(f, cls_id, xyxy, img_w, img_h) -> bool:
    x1, y1, x2, y2 = xyxy
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_w-1, x2), min(img_h-1, y2)
    w = max(0, x2 - x1); h = max(0, y2 - y1)
    if w <= 1 or h <= 1:
        return False
    if w < MIN_EYE_RATIO * img_w or h < MIN_EYE_RATIO * img_h:
        return False
    cx, cy = x1 + w / 2, y1 + h / 2
    nx, ny, nw, nh = cx / img_w, cy / img_h, w / img_w, h / img_h
    f.write(f"{cls_id} {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}\n")
    return True

def save_preview_grid(img_paths: List[Path], out_path: Path, grid: Tuple[int,int]=(5,6), max_side: int=320):
    """Lưu lưới ảnh xem nhanh (không bắt buộc)."""
    rows, cols = grid
    sample = img_paths[: rows*cols]
    if not sample: return
    tiles = []
    for p in sample:
        im = cv2.imread(str(p))
        if im is None: continue
        h, w = im.shape[:2]
        scale = max_side / max(h, w)
        im = cv2.resize(im, (int(w*scale), int(h*scale)))
        pad = np.full((max_side, max_side, 3), 32, np.uint8)
        y = (max_side - im.shape[0])//2
        x = (max_side - im.shape[1])//2
        pad[y:y+im.shape[0], x:x+im.shape[1]] = im
        tiles.append(pad)
    if not tiles: return
    # pad đủ số lượng
    while len(tiles) < rows*cols:
        tiles.append(np.full((max_side, max_side, 3), 32, np.uint8))
    # ghép
    rows_img = [np.hstack(tiles[c*cols:(c+1)*cols]) for c in range(rows)]
    grid_img = np.vstack(rows_img)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), grid_img)

def split_and_write(
    labeled: List[Tuple[Path, List[Tuple[int,int,int,int,int]]]],
    train_ratio=0.8
) -> dict:
    """Chia train/val, copy ảnh sang YOLO_Dataset/images/split và ghi nhãn/manifest + preview."""
    (YOLO_DATASET / "images" / "train").mkdir(parents=True, exist_ok=True)
    (YOLO_DATASET / "images" / "val").mkdir(parents=True, exist_ok=True)
    (YOLO_DATASET / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (YOLO_DATASET / "labels" / "val").mkdir(parents=True, exist_ok=True)

    random.shuffle(labeled)
    n_train = int(len(labeled) * train_ratio)
    train_set = labeled[:n_train]
    val_set = labeled[n_train:]

    kept_train, kept_val = 0, 0
    train_list, val_list = [], []

    for split_name, items in [("train", train_set), ("val", val_set)]:
        for img_path, bboxes in items:
            img = cv2.imread(str(img_path))
            if img is None: 
                continue
            h, w = img.shape[:2]

            dst_img = YOLO_DATASET / "images" / split_name / img_path.name
            dst_lbl = YOLO_DATASET / "labels" / split_name / (img_path.stem + ".txt")

            # copy ảnh
            shutil.copy2(img_path, dst_img)

            # ghi nhãn
            ok = 0
            with open(dst_lbl, "w", encoding="utf-8") as f:
                for cls_id, x1, y1, x2, y2 in bboxes:
                    ok += int(write_yolo_line(f, cls_id, (x1,y1,x2,y2), w, h))
            # nếu không có bbox hợp lệ -> xóa ảnh/label
            if ok == 0:
                try:
                    dst_img.unlink(missing_ok=True)
                    dst_lbl.unlink(missing_ok=True)
                except Exception:
                    pass
                continue

            if split_name == "train":
                kept_train += 1
                train_list.append(str(dst_img))
            else:
                kept_val += 1
                val_list.append(str(dst_img))

    # manifest
    (YOLO_DATASET / "manifests").mkdir(parents=True, exist_ok=True)
    (YOLO_DATASET / "manifests" / "train_list.txt").write_text("\n".join(train_list), encoding="utf-8")
    (YOLO_DATASET / "manifests" / "val_list.txt").write_text("\n".join(val_list), encoding="utf-8")

    # preview grids
    save_preview_grid([Path(p) for p in train_list], YOLO_DATASET / "preview" / "train_grid.jpg")
    save_preview_grid([Path(p) for p in val_list], YOLO_DATASET / "preview" / "val_grid.jpg")

    return {
        "train_kept": kept_train, "val_kept": kept_val,
        "train_skipped": len(train_set) - kept_train,
        "val_skipped": len(val_set) - kept_val
    }

def build_data_yaml():
    data_yaml = YOLO_DATASET / "data.yaml"
    data = {
        "path": str(YOLO_DATASET),
        "train": "images/train",
        "val": "images/val",
        "names": ["open_eye", "closed_eye"],
        "nc": 2,
    }
    data_yaml.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data_yaml

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true", help="Gom dữ liệu + auto-label + split")
    ap.add_argument("--new-real-only", action="store_true", help="Chỉ dùng NEW_REAL cho auto-label")
    ap.add_argument("--train", action="store_true", help="Huấn luyện YOLO")
    ap.add_argument("--export", action="store_true", help="Xuất ONNX sau khi train")
    ap.add_argument("--imgsz", type=int, default=768)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--finetune", action="store_true", help="Fine-tune (lr thấp, mosaic thấp)")
    ap.add_argument("--exp-name", type=str, default="drowsy_det", help="Tên run trong outputs/")
    ap.add_argument("--freeze", type=int, default=0, help="Đóng băng N tầng backbone")
    ap.add_argument("--weights", type=str, default=str(MODEL_DIR / "best_drowsy.pt"),
                    help="Trọng số khởi tạo khi train/finetune")
    return ap.parse_args()

def collect_images_for_autolabel(new_real_only: bool) -> List[Path]:
    cands = []
    if new_real_only:
        cands += [p for p in NEW_REAL.glob("**/*") if p.suffix.lower() in [".jpg", ".png", ".jpeg"]]
    else:
        for root in [OPEN_EYES, CLOSED_EYES, NEW_REAL]:
            cands += [p for p in root.glob("**/*") if p.suffix.lower() in [".jpg", ".png", ".jpeg"]]
    # loại trùng theo tên
    uniq = {}
    for p in cands:
        uniq[p.resolve()] = True
    lst = list(uniq.keys())
    random.shuffle(lst)
    return lst

def prepare_dataset(new_real_only: bool=False):
    print("📂 Paths:")
    paths = {
        "DATA_ROOT": str(DATA_ROOT),
        "OPEN_EYES": str(OPEN_EYES),
        "CLOSED_EYES": str(CLOSED_EYES),
        "VIDEOS": str(VIDEOS),
        "NEW_REAL": str(NEW_REAL),
        "YOLO_DATASET": str(YOLO_DATASET),
        "OUTPUT": str(OUTPUT),
        "MODEL_DIR": str(MODEL_DIR),
        "TMP_DIR": str(TMP_DIR)
    }
    print(json.dumps(paths, indent=2, ensure_ascii=False))

    # dọn YOLO_Dataset
    for sub in ["images/train", "images/val", "labels/train", "labels/val", "preview", "manifests"]:
        d = YOLO_DATASET / sub
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)

    mp = try_import_mediapipe()
    if mp is None:
        print("❌ Thiếu mediapipe (pip install mediapipe). Không thể auto-label.")
        sys.exit(1)

    cands = collect_images_for_autolabel(new_real_only)
    print(f"🔄 Auto-label {len(cands)} ảnh với MediaPipe FaceMesh...")

    labeled = []
    detected = 0
    for p in cands:
        out = autolabel_image(p, mp)
        if out is None: 
            continue
        bboxes, dbg = out
        if bboxes:
            labeled.append((p, bboxes))
            detected += 1

    stats = split_and_write(labeled, train_ratio=0.8)
    data_yaml = build_data_yaml()

    print(f"✅ Build xong YOLO_Dataset → {YOLO_DATASET}")
    print(f"📈 Stats: {json.dumps(stats)}")
    print(f"📝 Manifest: {YOLO_DATASET/'manifests/train_list.txt'} , {YOLO_DATASET/'manifests/val_list.txt'}")
    print(f"🖼️ Preview: {YOLO_DATASET/'preview/train_grid.jpg'} , {YOLO_DATASET/'preview/val_grid.jpg'}")
    print(f"📄 data.yaml: {data_yaml}")

@dataclass
class Trainer:
    weights: str
    imgsz: int = 768
    finetune: bool = False
    exp_name: str = "drowsy_det"
    freeze: int = 0

    def __post_init__(self):
        if YOLO is None:
            raise RuntimeError("Ultralytics chưa được cài trong môi trường hiện tại.")
        self.model = YOLO(self.weights) if Path(self.weights).exists() else YOLO("yolov8n.pt")

    def train(self, epochs=100, batch=8, workers=4):
        data_yaml = YOLO_DATASET / "data.yaml"
        if not data_yaml.exists():
            raise FileNotFoundError(f"Không thấy {data_yaml}. Hãy chạy --prepare trước.")
        if self.finetune and Path(self.weights).exists():
            lr0, mosaic, freeze = 0.001, 0.15, self.freeze
            print(f"💡 Finetune: lr0={lr0}, mosaic={mosaic}, freeze={freeze}")
        else:
            lr0, mosaic, freeze = 0.002, 0.30, 0

        args = dict(
            data=str(data_yaml),
            imgsz=self.imgsz,
            epochs=epochs,
            batch=batch,
            workers=workers,
            device="0",
            seed=SEED,
            project=str(OUTPUT),
            name=self.exp_name,
            cos_lr=True,
            lr0=lr0, lrf=0.12,
            optimizer="SGD",
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3.0,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            amp=False,              # Quadro T2000: tắt AMP cho ổn định
            patience=20,
            cache="disk",
            mosaic=mosaic,
            mixup=0.0,
            hsv_h=0.012, hsv_s=0.6, hsv_v=0.35,
            degrees=5.0, translate=0.05, scale=0.2, shear=1.0,
            erasing=0.0,
            box=7.5, cls=0.5,
            save_json=False,
            val=True,
            freeze=freeze
        )
        print("📊 Train args (tóm tắt):")
        print(json.dumps({k:v for k,v in args.items() if k in ["imgsz","epochs","batch","device","name","lr0","mosaic","freeze","amp","cache"]}, indent=2))
        res = self.model.train(**args)

        # chọn best.pt → copy về MODEL_DIR
        run_dir = Path(res.save_dir)
        best = run_dir / "weights" / "best.pt"
        if best.exists():
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            out_best = MODEL_DIR / "best_drowsy.pt"
            shutil.copy2(best, out_best)
            print(f"✅ Lưu best model → {out_best}")

    def export(self):
        best = MODEL_DIR / "best_drowsy.pt"
        if not best.exists():
            raise FileNotFoundError(f"Không thấy {best}")
        m = YOLO(str(best))
        onnx_path = MODEL_DIR / "drowsy_export.onnx"
        m.export(format="onnx", dynamic=False, opset=22, simplify=True)
        # Ultralytics mặc định lưu cạnh weights, đặt tên theo run; chuẩn hóa:
        # tìm file onnx mới nhất trong MODEL_DIR
        latest = None
        for p in MODEL_DIR.glob("*.onnx"):
            if latest is None or p.stat().st_mtime > latest.stat().st_mtime:
                latest = p
        if latest and latest != onnx_path:
            shutil.copy2(latest, onnx_path)
        print(f"✅ Exported ONNX → {onnx_path}")

def main():
    args = parse_args()

    if args.prepare:
        prepare_dataset(new_real_only=args.new_real_only)

    if args.train:
        trainer = Trainer(
            weights=args.weights,
            imgsz=args.imgsz,
            finetune=args.finetune,
            exp_name=args.exp_name,
            freeze=args.freeze
        )
        trainer.train(epochs=args.epochs, batch=args.batch)

    if args.export:
        Trainer(weights=str(MODEL_DIR / "best_drowsy.pt")).export()

if __name__ == "__main__":
    main()
