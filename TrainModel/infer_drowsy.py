#!/usr/bin/env python3

import argparse
import os
import time
from collections import deque, Counter
from pathlib import Path

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception as e:  # thông báo dễ hiểu nếu thiếu lib
    raise RuntimeError(
        "Không import được 'ultralytics'. Cài bằng: pip install ultralytics"
    ) from e


# ------------------------- Utils -------------------------
def put_text(img, text, org, scale=0.7, color=(255, 255, 255), thick=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_panel(frame, lines, top=6, left=8, line_h=22, color=(0, 255, 0)):
    """Vẽ panel chữ trái góc trên."""
    x, y = left, top
    for i, ln in enumerate(lines):
        put_text(frame, ln, (x, y + i * line_h), 0.6, color, 2)


def enhance_face(frame, do_clahe=False):
    """Tùy chọn tăng tương phản nhẹ để giúp phát hiện tốt hơn ở ánh sáng yếu."""
    if not do_clahe:
        return frame
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)


def pick_eye_state(dets, cls_names, conf_thres):
    """
    Từ danh sách boxes của 1 frame -> tính số open/closed.
    dets: (xyxy, conf, cls)
    Trả về: counts, best_boxes
    """
    counts = Counter()
    best_boxes = []  # (xyxy, label_text, color)

    for xyxy, conf, cls_id in dets:
        if conf < conf_thres:
            continue
        label = cls_names[int(cls_id)]
        counts[label] += 1
        color = (0, 255, 0) if label == "open_eye" else (0, 0, 255)
        text = f"{label} {conf:.2f}"
        best_boxes.append((xyxy.astype(int), text, color))

    # Chọn trạng thái frame theo đa số (tie-break bằng conf trung bình)
    if counts:
        state = "closed_eye" if counts["closed_eye"] > counts["open_eye"] else "open_eye"
    else:
        state = "none"

    return state, counts, best_boxes


class PERCLOSSmoother:
    """Giữ cửa sổ trượt N frame để tính % mắt-đóng ổn định."""

    def __init__(self, window=45):  # ~1.5s với 30 FPS
        self.win = window
        self.hist = deque(maxlen=window)  # 1 nếu closed, 0 nếu open, ignore nếu none

    def push(self, state):
        if state == "closed_eye":
            self.hist.append(1)
        elif state == "open_eye":
            self.hist.append(0)
        # 'none' -> bỏ qua (không đẩy), để không làm nhiễu PERCLOS

    def value(self):
        if not self.hist:
            return 0.0
        return float(sum(self.hist)) / len(self.hist)


# ------------------------- Core Inference -------------------------
def run(
    model_path,
    source=0,
    conf=0.35,
    iou=0.45,
    device="0",
    imgsz=768,
    save=False,
    out_path="runs/infer_out.mp4",
    show=True,
    perclos_win=45,
    sleepy_thr=0.6,
    min_closed_ms=1200,
    clahe=False,
):
    """
    source:
      - 0,1,...: webcam index
      - path to image or video
    """
    # Load model
    model = YOLO(model_path)
    names = model.names  # {0:'open_eye', 1:'closed_eye'} hoặc ngược
    # Chuẩn hóa list theo tên:
    class_names = [names[k] for k in sorted(names.keys())]

    # Video writer (nếu cần)
    vw = None

    # Nếu source là ảnh
    if isinstance(source, str) and Path(source).is_file() and source.lower().split(".")[-1] in {"jpg", "jpeg", "png", "bmp"}:
        img = cv2.imread(source)
        if img is None:
            raise FileNotFoundError(f"Không đọc được ảnh: {source}")
        img = enhance_face(img, clahe)
        res = model.predict(img, conf=conf, iou=iou, imgsz=imgsz, device=device, verbose=False)[0]
        dets = []
        if res.boxes is not None and len(res.boxes) > 0:
            for b in res.boxes:
                dets.append((b.xyxy.cpu().numpy()[0], float(b.conf.cpu().numpy()), int(b.cls.cpu().numpy())))
        state, counts, boxes = pick_eye_state(dets, class_names, conf)
        # vẽ
        for xyxy, txt, color in boxes:
            cv2.rectangle(img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
            put_text(img, txt, (xyxy[0] + 2, xyxy[1] - 6), 0.6, color, 2)
        draw_panel(
            img,
            [
                f"Model: {Path(model_path).name}",
                f"State: {state}",
                f"Counts: open={counts['open_eye']} closed={counts['closed_eye']}",
                f"Conf={conf:.2f} IOU={iou:.2f}",
                "Press any key to close",
            ],
            color=(0, 255, 255),
        )
        if save:
            out_img = Path(out_path)
            if out_img.suffix == "":
                out_img = out_img.with_suffix(".jpg")
            out_img.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_img), img)
        if show:
            cv2.imshow("Drowsy Inference (Image)", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        print({"state": state, "counts": dict(counts)})
        return

    # Webcam or Video
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được nguồn video: {source}")
    # cài writer nếu save
    if save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        vw = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    perclos = PERCLOSSmoother(window=perclos_win)
    t0 = time.time()
    last_state_change_t = t0
    is_sleepy = False
    closed_streak_ms = 0
    last_frame_t = t0

    # UI hướng dẫn
    help_lines = [
        "[q] quit   [s] save toggle   [c] CLAHE toggle",
        "[↑/↓] conf  [←/→] sleepy_thr   [space] reset PERCLOS",
    ]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        dt = max(1e-3, now - last_frame_t)
        last_frame_t = now

        frame = enhance_face(frame, clahe)
        # YOLO predict (1 frame)
        res = model.predict(frame, conf=conf, iou=iou, imgsz=imgsz, device=device, verbose=False)[0]

        dets = []
        if res.boxes is not None and len(res.boxes) > 0:
            for b in res.boxes:
                dets.append((b.xyxy.cpu().numpy()[0], float(b.conf.cpu().numpy()), int(b.cls.cpu().numpy())))

        state, counts, boxes = pick_eye_state(dets, class_names, conf)
        perclos.push(state)

        # streak đóng mắt theo thời gian
        if state == "closed_eye":
            closed_streak_ms += dt * 1000.0
        else:
            closed_streak_ms = max(0.0, closed_streak_ms - dt * 500.0)  # giảm dần để tránh jitter

        # quyết định sleepy: PERCLOS cao hoặc nhắm mắt liên tục
        sleepy = (perclos.value() >= sleepy_thr) or (closed_streak_ms >= min_closed_ms)

        # vẽ boxes
        for xyxy, txt, color in boxes:
            cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
            put_text(frame, txt, (xyxy[0] + 2, xyxy[1] - 6), 0.6, color, 2)

        # header & panel
        fps = 1.0 / dt
        panel = [
            f"Model: {Path(model_path).name}",
            f"State={state}  open={counts['open_eye']}  closed={counts['closed_eye']}",
            f"PERCLOS={perclos.value():.2f}  Thr={sleepy_thr:.2f}  ClosedStreak={int(closed_streak_ms)}ms",
            f"FPS={fps:.1f}  Conf={conf:.2f}  IOU={iou:.2f}  CLAHE={'ON' if clahe else 'OFF'}  Save={'ON' if save else 'OFF'}",
        ]
        draw_panel(frame, panel, color=(0, 255, 0))
        draw_panel(frame, help_lines, top=frame.shape[0] - 50, color=(200, 200, 0))

        # cảnh báo lớn
        if sleepy:
            put_text(frame, "SLEEPY!", (30, 70), 1.3, (0, 0, 255), 3)
            cv2.rectangle(frame, (20, 20), (frame.shape[1] - 20, frame.shape[0] - 20), (0, 0, 255), 3)

        if show:
            cv2.imshow("Drowsy Inference (Video/Webcam)", frame)

        if save and vw is not None:
            vw.write(frame)

        # phím tắt
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            save = not save
            if save and vw is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                fps = cap.get(cv2.CAP_PROP_FPS) or 30
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                vw = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        elif key == ord("c"):
            clahe = not clahe
        elif key == 82:  # Up
            conf = min(0.99, conf + 0.02)
        elif key == 84:  # Down
            conf = max(0.05, conf - 0.02)
        elif key == 83:  # Right
            sleepy_thr = min(0.95, sleepy_thr + 0.02)
        elif key == 81:  # Left
            sleepy_thr = max(0.30, sleepy_thr - 0.02)
        elif key == 32:  # Space
            perclos = PERCLOSSmoother(window=perclos_win)
            closed_streak_ms = 0

    cap.release()
    if vw is not None:
        vw.release()
    cv2.destroyAllWindows()


# ------------------------- CLI -------------------------
def parse_args():
    p = argparse.ArgumentParser("Drowsy detection – webcam / image / video")
    p.add_argument("--model", type=str,
                   default="/home/gess/Documents/sub/TrainModel/models/best_drowsy.pt")
    p.add_argument("--source", type=str, default="0",
                   help="0|1|... (webcam) hoặc đường dẫn ảnh/video")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--imgsz", type=int, default=768)
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--show", action="store_true", default=True)
    p.add_argument("--save", action="store_true")
    p.add_argument("--out", type=str, default="/home/gess/Documents/sub/TrainModel/outputs/infer_out.mp4")

    # drowsy logic
    p.add_argument("--perclos-win", type=int, default=45, help="Số frame để tính PERCLOS")
    p.add_argument("--sleepy-thr", type=float, default=0.60, help="Ngưỡng PERCLOS báo buồn ngủ")
    p.add_argument("--min-closed-ms", type=int, default=1200, help="Nhắm mắt liên tục >= ms sẽ báo buồn ngủ")
    p.add_argument("--clahe", action="store_true", help="Bật tăng tương phản (ánh sáng yếu)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # nguồn: nếu là số -> webcam index
    src = 0 if args.source.isdigit() else args.source
    run(
        model_path=args.model,
        source=src,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        imgsz=args.imgsz,
        save=args.save,
        out_path=args.out,
        show=args.show,
        perclos_win=args.perclos_win,
        sleepy_thr=args.sleepy_thr,
        min_closed_ms=args.min_closed_ms,
        clahe=args.clahe,
    )
