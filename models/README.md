# Models Directory

## YOLOv8 Model cho Nhận Diện Biển Báo

Thư mục này chứa model YOLOv8 đã được train để nhận diện biển báo giao thông.

### Cách sử dụng:

1. **Tải model đã train:**
   - Đặt file model `best.pt` vào thư mục này
   - Model có thể được train từ dataset biển báo giao thông Việt Nam

2. **Hoặc sử dụng model YOLOv8 mặc định:**
   - Nếu chưa có model custom, có thể dùng YOLOv8n pretrained:
   ```python
   from ultralytics import YOLO
   model = YOLO('yolov8n.pt')  # Tự động tải về
   ```

3. **Train model custom:**
   ```bash
   # Cài đặt ultralytics
   pip install ultralytics
   
   # Train model
   yolo detect train data=traffic_signs.yaml model=yolov8n.pt epochs=100 imgsz=640
   ```

### Lưu ý:
- File model `best.pt` cần được đặt tại: `/vercel/sandbox/models/best.pt`
- Nếu không có model, ứng dụng sẽ báo lỗi khi khởi động
- Có thể sử dụng model pretrained từ Ultralytics Hub hoặc train riêng

### Dataset gợi ý:
- [Traffic Signs Dataset](https://universe.roboflow.com/traffic-signs)
- [Vietnamese Traffic Signs](https://www.kaggle.com/datasets/vietnamesetrafficsigns)
