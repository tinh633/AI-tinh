from ultralytics import YOLO
import pandas as pd

# Load model và evaluate
model = YOLO("/home/gess/Documents/sub/TrainModel/models/best_drowsy.pt")
metrics = model.val(data="/home/gess/Documents/sub/TrainModel/YOLO_Dataset/data.yaml", imgsz=768, verbose=True)

# Lấy các giá trị cần thiết
names = list(metrics.names.values())
precision = metrics.box.p.tolist()
recall = metrics.box.r.tolist()
map50 = metrics.box.map50.tolist()
map5095 = metrics.box.map.tolist()

# Tạo dataframe gọn
df = pd.DataFrame({
    'Class': names,
    'Precision': precision,
    'Recall': recall,
    'mAP50': map50,
    'mAP50-95': map5095
})

# In ra gọn gàng
print(df.to_string(index=False, float_format="%.3f"))

# (Tùy chọn) Xuất ra file CSV gọn
df.to_csv("/home/gess/Documents/sub/TrainModel/outputs/clean_metrics.csv", index=False)
