
from __future__ import annotations
import os
import sys
import argparse
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, List, Optional, Tuple, Union

import cv2
import numpy as np
from ultralytics import YOLO

# ---------------- PATH & CONFIG ---------------- #

DATA_ROOT = Path("/home/gess/Documents/sub/TrainModel/").resolve()
MODEL_DIR = DATA_ROOT / "models"
DEFAULT_WEIGHTS = MODEL_DIR / "best_drowsy.pt"

CLASS_NAMES = ["open_eye", "closed_eye"]
CLS_OPEN = 0
CLS_CLOSED = 1


# ---------------- DROWSY STATE TRACKER ---------------- #

@dataclass
class DrowsyConfig:
    history_len: int = 60          # số frame nhớ lại (ví dụ 60 ~ 2s nếu 30 FPS)
    closed_perc_threshold: float = 0.7  # >= 70% frame closed trong cửa sổ
    closed_consec_threshold: int = 20   # >= 20 frame liên tiếp closed thì báo drowsy


class DrowsyStateTracker:
    """
    Lưu lịch sử trạng thái mắt và quyết định có drowsy hay không.
    eye_state:
      1  = closed
      0  = open
      -1 = unknown
    """

    def __init__(self, cfg: DrowsyConfig):
        self.cfg = cfg
        self.history: Deque[int] = deque(maxlen=cfg.history_len)

    def update(self, eye_state: int) -> Tuple[bool, float, int]:
        """
        Cập nhật 1 frame, trả về:
          drowsy (bool), perc_closed (0-1), closed_streak (int)
        """
        # unknown thì đừng thêm quá nhiều noise
        if eye_state not in (0, 1):
            self.history.append(-1)
        else:
            self.history.append(eye_state)

        # Đếm closed trong history (bỏ qua unknown)
        valid_states = [s for s in self.history if s >= 0]
        closed_count = sum(1 for s in valid_states if s == 1)
        total_valid = len(valid_states)
        perc_closed = (closed_count / total_valid) if total_valid > 0 else 0.0

        # Chuỗi closed liên tiếp (tính từ cuối)
        closed_streak = 0
        for s in reversed(self.history):
            if s == 1:
                closed_streak += 1
            elif s == 0:
                break
            else:
                # unknown thì cho phép nhưng không reset streak
                continue

        is_drowsy = False
        if total_valid >= 5:  # cần ít nhất 5 frame có thông tin
            if (perc_closed >= self.cfg.closed_perc_threshold and
                    closed_streak >= self.cfg.closed_consec_threshold):
                is_drowsy = True

        return is_drowsy, perc_closed, closed_streak


# ---------------- YOLO WRAPPER ---------------- #

@dataclass
class InferenceConfig:
    weights: Union[str, Path] = DEFAULT_WEIGHTS
    device: str = "0"          # "0" = GPU0, "cpu" = CPU
    imgsz: int = 640
    conf: float = 0.5          # confidence threshold


class EyeDetector:
    def __init__(self, cfg: InferenceConfig):
        self.cfg = cfg

        w = Path(cfg.weights)
        if not w.exists():
            raise FileNotFoundError(
                f"Không tìm thấy weights: {w}. "
                f"Hãy train trước hoặc chỉnh lại --weights."
            )

        print(f"🚀 Loading model: {w}")
        self.model = YOLO(str(w))

    def infer_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, int, List[Tuple[int, int, int, int, int, float]]]:
        """
        Chạy YOLO trên 1 frame.
        Trả về:
          - frame_with_draw: frame đã vẽ bbox + label
          - eye_state: 1=closed, 0=open, -1=unknown
          - dets: list (x1, y1, x2, y2, cls_id, conf)
        """
        h, w = frame.shape[:2]

        results = self.model(
            frame,
            imgsz=self.cfg.imgsz,
            conf=self.cfg.conf,
            device=self.cfg.device,
            verbose=False,
        )
        r = results[0]

        dets: List[Tuple[int, int, int, int, int, float]] = []

        if r.boxes is None or len(r.boxes) == 0:
            return frame, -1, dets

        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy().astype(int)

        cls_best_score = {-1: -1.0, 0: -1.0, 1: -1.0}
        cls_best = {-1: None, 0: None, 1: None}

        for xyxy, c, cls_id in zip(boxes, confs, clss):
            x1, y1, x2, y2 = xyxy.astype(int)
            cls_id = int(cls_id)
            if cls_id not in (CLS_OPEN, CLS_CLOSED):
                continue

            # Lưu det
            dets.append((x1, y1, x2, y2, cls_id, float(c)))

            # chọn box tốt nhất cho từng class
            if c > cls_best_score.get(cls_id, -1):
                cls_best_score[cls_id] = float(c)
                cls_best[cls_id] = (x1, y1, x2, y2)

        # eye_state theo độ tự tin cao hơn giữa open vs closed
        open_conf = cls_best_score.get(CLS_OPEN, -1.0)
        closed_conf = cls_best_score.get(CLS_CLOSED, -1.0)

        if open_conf < 0 and closed_conf < 0:
            eye_state = -1
        elif closed_conf > open_conf:
            eye_state = 1
        else:
            eye_state = 0

        # Vẽ bbox + label lên frame
        for x1, y1, x2, y2, cls_id, conf in dets:
            color = (0, 255, 0) if cls_id == CLS_OPEN else (0, 165, 255)
            label = f"{CLASS_NAMES[cls_id]} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame, label,
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA
            )

        return frame, eye_state, dets


# ---------------- DRAW OVERLAY ---------------- #

def draw_drowsy_overlay(
    frame: np.ndarray,
    is_drowsy: bool,
    perc_closed: float,
    closed_streak: int,
) -> np.ndarray:
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Thanh bar phía dưới: tỉ lệ closed
    bar_w = int(w * 0.5)
    bar_h = 18
    bar_x = int(w * 0.25)
    bar_y = h - bar_h - 10

    # Nền bar
    cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)

    # Phần closed
    closed_len = int(bar_w * perc_closed)
    cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + closed_len, bar_y + bar_h), (0, 140, 255), -1)

    # Viền
    cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 1)

    # Text bên cạnh
    text = f"closed: {perc_closed * 100:.0f}% | streak: {closed_streak}"
    cv2.putText(
        overlay, text,
        (bar_x, bar_y - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA
    )

    # DROWSY ALERT
    status_txt = "DROWSY" if is_drowsy else "AWAKE"
    status_color = (0, 0, 255) if is_drowsy else (0, 255, 0)

    cv2.putText(
        overlay, status_txt,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2, cv2.LINE_AA
    )

    # blend overlay
    alpha = 0.7
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)


# ---------------- MODE: WEBCAM ---------------- #

def run_webcam(detector: EyeDetector, drowsy_trk: DrowsyStateTracker, cam_index: int = 0):
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"❌ Không mở được webcam index {cam_index}")
        return

    print("🎥 Webcam mode. Nhấn 'q' để thoát.")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("⚠️ Không đọc được frame từ webcam.")
            break

        # Option: resize nhỏ bớt cho nhanh
        # frame = cv2.resize(frame, (640, 360))

        frame_det, eye_state, _ = detector.infer_frame(frame)
        is_drowsy, perc_closed, streak = drowsy_trk.update(eye_state)
        frame_out = draw_drowsy_overlay(frame_det, is_drowsy, perc_closed, streak)

        cv2.imshow("Drowsy Detection - Webcam", frame_out)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


# ---------------- MODE: IMAGE ---------------- #

def iter_images(path: Path):
    if path.is_file():
        yield path
    else:
        for p in sorted(path.glob("**/*.*")):
            if p.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                yield p


def run_image_mode(detector: EyeDetector, drowsy_trk: DrowsyStateTracker, path: Path):
    if not path.exists():
        print(f"❌ Không tìm thấy đường dẫn ảnh: {path}")
        return

    for img_path in iter_images(path):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"⚠️ Không đọc được ảnh: {img_path}")
            continue

        frame_det, eye_state, _ = detector.infer_frame(img)
        is_drowsy, perc_closed, streak = drowsy_trk.update(eye_state)
        frame_out = draw_drowsy_overlay(frame_det, is_drowsy, perc_closed, streak)

        print(f"🖼 {img_path.name}: "
              f"eye_state={'closed' if eye_state==1 else 'open' if eye_state==0 else 'unknown'}, "
              f"closed={perc_closed*100:.0f}%, streak={streak}, drowsy={is_drowsy}")

        cv2.imshow(f"Drowsy Detection - {img_path.name}", frame_out)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()
        if key == ord('q') or key == 27:
            break


# ---------------- MODE: VIDEO ---------------- #

def run_video_mode(detector: EyeDetector, drowsy_trk: DrowsyStateTracker, path: Path):
    if not path.exists():
        print(f"❌ Không tìm thấy video: {path}")
        return

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"❌ Không mở được video: {path}")
        return

    print(f"🎬 Video mode: {path.name}. Nhấn 'q' để thoát.")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("▶️ Hết video.")
            break

        frame_det, eye_state, _ = detector.infer_frame(frame)
        is_drowsy, perc_closed, streak = drowsy_trk.update(eye_state)
        frame_out = draw_drowsy_overlay(frame_det, is_drowsy, perc_closed, streak)

        cv2.imshow(f"Drowsy Detection - {path.name}", frame_out)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


# ---------------- CLI ---------------- #

def parse_args():
    ap = argparse.ArgumentParser(description="Drowsy detection realtime / image / video")
    ap.add_argument("--weights", type=str, default=str(DEFAULT_WEIGHTS),
                    help="Đường dẫn weights YOLO (.pt)")

    mode_group = ap.add_mutually_exclusive_group()
    mode_group.add_argument("--webcam", action="store_true", help="Dùng webcam realtime")
    mode_group.add_argument("--image", type=str, help="Path ảnh hoặc folder ảnh")
    mode_group.add_argument("--video", type=str, help="Path video")

    ap.add_argument("--cam-index", type=int, default=0, help="Chỉ số camera (mặc định 0)")
    ap.add_argument("--device", type=str, default="0", help="Thiết bị: '0' (GPU0) hoặc 'cpu'")
    ap.add_argument("--imgsz", type=int, default=640, help="Kích thước input YOLO")
    ap.add_argument("--conf", type=float, default=0.5, help="Confidence threshold YOLO")

    # Drowsy config
    ap.add_argument("--history", type=int, default=60,
                    help="Chiều dài history (số frame nhớ lại)")
    ap.add_argument("--closed-perc", type=float, default=0.7,
                    help="Ngưỡng % closed trong history để xem xét drowsy")
    ap.add_argument("--closed-consec", type=int, default=20,
                    help="Ngưỡng số frame closed liên tiếp để drowsy")

    return ap.parse_args()


def main():
    args = parse_args()

    # Nếu không truyền gì → mặc định webcam
    if len(sys.argv) == 1:
        args.webcam = True

    cfg_inf = InferenceConfig(
        weights=args.weights,
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
    )
    detector = EyeDetector(cfg_inf)

    cfg_drowsy = DrowsyConfig(
        history_len=args.history,
        closed_perc_threshold=args.closed_perc,
        closed_consec_threshold=args.closed_consec,
    )
    drowsy_trk = DrowsyStateTracker(cfg_drowsy)

    if args.webcam:
        run_webcam(detector, drowsy_trk, cam_index=args.cam_index)
    elif args.image:
        run_image_mode(detector, drowsy_trk, Path(args.image))
    elif args.video:
        run_video_mode(detector, drowsy_trk, Path(args.video))
    else:
        print("⚠️ Không có mode nào được chọn. Dùng --webcam hoặc --image hoặc --video.")
        print("Ví dụ:")
        print("  python infer_drowsy.py --webcam")
        print("  python infer_drowsy.py --image /path/to/img_or_folder")
        print("  python infer_drowsy.py --video /path/to/video.mp4")


if __name__ == "__main__":
    main()
