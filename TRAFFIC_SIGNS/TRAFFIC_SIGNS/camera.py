from ultralytics import YOLO
import cv2

model = YOLO("runs_yolo/yolov13_custom_train2/weights/best.pt")

#  Mở webcam (0 = mặc định)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Không thể mở webcam!")
    exit()

#  Vòng lặp nhận diện từng frame
while True:
    ret, frame = cap.read()
    if not ret:
        print("Không đọc được khung hình.")
        break

    # Nhận diện trực tiếp
    results = model.predict(frame, imgsz=640, conf=0.45, device=0, verbose=False)

    # Vẽ bounding box lên frame (hàm render của ultralytics)
    annotated_frame = results[0].plot()

    # Hiển thị
    cv2.imshow("YOLOv13 - Traffic Sign Detection", annotated_frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#  Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()