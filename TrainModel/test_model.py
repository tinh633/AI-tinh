# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# import os
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
# import shutil
# from datetime import datetime
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
# import warnings
# warnings.filterwarnings('ignore')
# from ultralytics import YOLO
# import yaml
# from tqdm import tqdm
# import glob

# # ==================== CONFIGURATION ====================
# DATASET_PATHS = {
#     'open_eyes': '/home/gess/Documents/Data/Open_Eyes/',
#     'closed_eyes': '/home/gess/Documents/Data/Closed_Eyes/', 
#     'real_webcam': '/home/gess/Pictures/Webcam/',
#     'additional_webcam': '/home/gess/Documents/Data/Additional_Webcam/',
#     'videos': '/home/gess/Documents/Data/Fold1_part2/',
#     'new_videos': '/home/gess/Documents/Data/New_Videos/',
#     'yolo_dataset': '/home/gess/Documents/Data/YOLO_Dataset/',
#     'model_save': '/home/gess/Documents/sub/Py/hhehee/eye_detection_model/',
#     'output': '/home/gess/Documents/sub/Py/hhehee/eye_detection_results/'
# }

# # Tạo thư mục YOLO dataset structure
# YOLO_DIRS = ['images/train', 'images/val', 'labels/train', 'labels/val']
# for dir_name in YOLO_DIRS:
#     os.makedirs(os.path.join(DATASET_PATHS['yolo_dataset'], dir_name), exist_ok=True)

# for path in DATASET_PATHS.values():
#     if path.endswith('/'):
#         os.makedirs(path, exist_ok=True)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"🚀 Using device: {device}")

# # ==================== YOLO DATA PROCESSOR ====================
# class YOLODataProcessor:
#     def __init__(self, img_size=640):
#         self.img_size = img_size
#         self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#         self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
#     def prepare_yolo_dataset(self):
#         """Chuẩn bị dataset cho YOLO training"""
#         print("🔄 Preparing YOLO dataset...")
        
#         all_images = []
#         all_labels = []
        
#         # 1. Process basic images
#         print("📁 Processing basic images...")
#         all_images.extend(self._process_basic_images())
        
#         # 2. Process webcam images
#         print("📸 Processing webcam images...")
#         all_images.extend(self._process_webcam_images())
        
#         # 3. Process videos
#         print("🎥 Processing videos...")
#         all_images.extend(self._process_videos())
        
#         # Split data và tạo YOLO format
#         self._create_yolo_dataset(all_images)
        
#     def _process_basic_images(self):
#         """Xử lý ảnh cơ bản"""
#         images_info = []
        
#         # Open eyes
#         open_files = glob.glob(os.path.join(DATASET_PATHS['open_eyes'], '*.jpg')) + \
#                     glob.glob(os.path.join(DATASET_PATHS['open_eyes'], '*.png'))
        
#         for img_path in open_files[:1000]:  # Giới hạn 1000 ảnh mỗi class
#             images_info.append({
#                 'path': img_path,
#                 'label': 0,  # 0 = open eyes
#                 'source': 'basic_open'
#             })
        
#         # Closed eyes
#         closed_files = glob.glob(os.path.join(DATASET_PATHS['closed_eyes'], '*.jpg')) + \
#                       glob.glob(os.path.join(DATASET_PATHS['closed_eyes'], '*.png'))
        
#         for img_path in closed_files[:1000]:
#             images_info.append({
#                 'path': img_path,
#                 'label': 1,  # 1 = closed eyes
#                 'source': 'basic_closed'
#             })
        
#         return images_info
    
#     def _process_webcam_images(self):
#         """Xử lý ảnh webcam"""
#         images_info = []
#         webcam_folders = [DATASET_PATHS['real_webcam'], DATASET_PATHS['additional_webcam']]
        
#         for folder in webcam_folders:
#             if not os.path.exists(folder):
#                 continue
                
#             for img_path in glob.glob(os.path.join(folder, '*.jpg')) + glob.glob(os.path.join(folder, '*.png')):
#                 filename = os.path.basename(img_path).lower()
                
#                 if any(keyword in filename for keyword in ['open', 'mo', 'o_']):
#                     label = 0
#                 elif any(keyword in filename for keyword in ['closed', 'dong', 'c_', 'nhammat']):
#                     label = 1
#                 else:
#                     continue
                
#                 images_info.append({
#                     'path': img_path,
#                     'label': label,
#                     'source': 'webcam'
#                 })
        
#         return images_info
    
#     def _process_videos(self):
#         """Xử lý video - trích xuất frames"""
#         images_info = []
#         video_folders = [DATASET_PATHS['videos'], DATASET_PATHS['new_videos']]
        
#         for folder in video_folders:
#             if not os.path.exists(folder):
#                 continue
                
#             for video_path in glob.glob(os.path.join(folder, '*.mp4')) + \
#                              glob.glob(os.path.join(folder, '*.avi')) + \
#                              glob.glob(os.path.join(folder, '*.mov')):
                
#                 # Xác định label từ tên video
#                 video_name = os.path.basename(video_path).lower()
#                 if any(keyword in video_name for keyword in ['open', 'mo']):
#                     label = 0
#                 elif any(keyword in video_name for keyword in ['closed', 'dong', 'nhammat']):
#                     label = 1
#                 else:
#                     continue
                
#                 # Trích xuất frames
#                 frames = self._extract_frames_from_video(video_path, max_frames=20)
#                 for i, frame_path in enumerate(frames):
#                     images_info.append({
#                         'path': frame_path,
#                         'label': label,
#                         'source': 'video'
#                     })
        
#         return images_info
    
#     def _extract_frames_from_video(self, video_path, max_frames=20):
#         """Trích xuất frames từ video"""
#         frames = []
#         cap = cv2.VideoCapture(video_path)
        
#         if not cap.isOpened():
#             return frames
        
#         fps = cap.get(cv2.CAP_PROP_FPS)
#         frame_interval = max(1, int(fps))  # 1 frame mỗi giây
        
#         frame_count = 0
#         saved_count = 0
        
#         while saved_count < max_frames:
#             ret, frame = cap.read()
#             if not ret:
#                 break
                
#             if frame_count % frame_interval == 0:
#                 # Lưu frame tạm thời
#                 frame_filename = f"temp_frame_{os.path.basename(video_path)}_{saved_count}.jpg"
#                 frame_path = os.path.join('/tmp', frame_filename)
#                 cv2.imwrite(frame_path, frame)
#                 frames.append(frame_path)
#                 saved_count += 1
            
#             frame_count += 1
        
#         cap.release()
#         return frames
    
#     def _create_yolo_dataset(self, images_info):
#         """Tạo dataset format YOLO"""
#         print("📦 Creating YOLO dataset format...")
        
#         if len(images_info) == 0:
#             print("❌ No images found for dataset creation!")
#             return
        
#         # Split data
#         train_data, val_data = train_test_split(
#             images_info, test_size=0.2, random_state=42, 
#             stratify=[img['label'] for img in images_info]
#         )
        
#         # Tạo YOLO format cho train và val
#         self._create_yolo_split(train_data, 'train')
#         self._create_yolo_split(val_data, 'val')
        
#         # Tạo file data.yaml
#         self._create_yaml_config()
        
#         print(f"✅ YOLO dataset created! Train: {len(train_data)}, Val: {len(val_data)}")
    
#     def _create_yolo_split(self, data, split_type):
#         """Tạo dữ liệu cho train/val split"""
#         image_dir = os.path.join(DATASET_PATHS['yolo_dataset'], 'images', split_type)
#         label_dir = os.path.join(DATASET_PATHS['yolo_dataset'], 'labels', split_type)
        
#         for i, img_info in enumerate(tqdm(data, desc=f"Processing {split_type}")):
#             try:
#                 # Đọc và xử lý ảnh
#                 img = cv2.imread(img_info['path'])
#                 if img is None:
#                     continue
                
#                 # Resize ảnh
#                 img_resized = cv2.resize(img, (self.img_size, self.img_size))
                
#                 # Lưu ảnh
#                 img_filename = f"{split_type}_{i:06d}.jpg"
#                 img_save_path = os.path.join(image_dir, img_filename)
#                 cv2.imwrite(img_save_path, img_resized)
                
#                 # Tạo label file (YOLO format)
#                 label_filename = f"{split_type}_{i:06d}.txt"
#                 label_save_path = os.path.join(label_dir, label_filename)
                
#                 # YOLO format: class x_center y_center width height (normalized)
#                 # Với ảnh eye crop, coi như toàn bộ ảnh là bounding box
#                 with open(label_save_path, 'w') as f:
#                     # class_id, x_center, y_center, width, height (all normalized)
#                     f.write(f"{img_info['label']} 0.5 0.5 1.0 1.0\n")
                    
#             except Exception as e:
#                 print(f"❌ Error processing {img_info['path']}: {e}")
#                 continue
    
#     def _create_yaml_config(self):
#         """Tạo file cấu hình YOLO"""
#         yaml_content = {
#             'path': DATASET_PATHS['yolo_dataset'],
#             'train': 'images/train',
#             'val': 'images/val',
#             'nc': 2,  # number of classes
#             'names': ['open_eye', 'closed_eye']  # class names
#         }
        
#         yaml_path = os.path.join(DATASET_PATHS['yolo_dataset'], 'data.yaml')
#         with open(yaml_path, 'w') as f:
#             yaml.dump(yaml_content, f, default_flow_style=False)
        
#         print(f"✅ YAML config created: {yaml_path}")

# # ==================== YOLO MODEL TRAINER ====================
# class YOLOEyeTrainer:
#     def __init__(self, model_size='n'):  # n, s, m, l, x
#         self.model_size = model_size
#         self.model = None
#         self.data_processor = YOLODataProcessor()
        
#     def prepare_data(self):
#         """Chuẩn bị dữ liệu cho training"""
#         print("🔄 Preparing training data...")
#         self.data_processor.prepare_yolo_dataset()
        
#     def train_model(self, epochs=100, imgsz=640, batch_size=16):
#         """Training YOLO model"""
#         print("🚀 Starting YOLO Training...")
        
#         # Chuẩn bị dữ liệu
#         self.prepare_data()
        
#         # Load YOLO model
#         model_name = f'yolov8{self.model_size}.pt'
#         self.model = YOLO(model_name)
        
#         # Training configuration
#         training_args = {
#             'data': os.path.join(DATASET_PATHS['yolo_dataset'], 'data.yaml'),
#             'epochs': epochs,
#             'imgsz': imgsz,
#             'batch': batch_size,
#             'patience': 20,
#             'save': True,
#             'exist_ok': True,
#             'pretrained': True,
#             'optimizer': 'AdamW',
#             'lr0': 0.001,
#             'weight_decay': 0.0005,
#             'device': '0' if torch.cuda.is_available() else 'cpu',
#             'workers': 4,
#             'project': DATASET_PATHS['model_save'],
#             'name': f'yolov8{self.model_size}_eye_detection',
#             'verbose': True
#         }
        
#         print(f"📊 Training Configuration:")
#         print(f"   Model: YOLOv8{self.model_size}")
#         print(f"   Epochs: {epochs}")
#         print(f"   Image size: {imgsz}")
#         print(f"   Batch size: {batch_size}")
#         print(f"   Device: {training_args['device']}")
        
#         # Start training
#         results = self.model.train(**training_args)
        
#         # Save best model
#         self._save_best_model()
        
#         return results
    
#     def _save_best_model(self):
#         """Lưu best model và convert sang format phù hợp"""
#         # Model sẽ tự động lưu trong thư mục runs
#         # Copy best model đến thư mục model_save
#         best_model_path = self.model.ckpt_path
        
#         if best_model_path and os.path.exists(best_model_path):
#             # Copy best model
#             final_model_path = os.path.join(
#                 DATASET_PATHS['model_save'], 
#                 'best_eye_detection_yolo.pt'
#             )
#             shutil.copy2(best_model_path, final_model_path)
#             print(f"✅ Best model saved: {final_model_path}")
        
#     def evaluate_model(self):
#         """Đánh giá model - ĐÃ SỬA LỖI"""
#         if self.model is None:
#             print("❌ No model available for evaluation!")
#             return
        
#         # Validation dataset path
#         val_data_path = os.path.join(DATASET_PATHS['yolo_dataset'], 'data.yaml')
        
#         # Evaluate
#         metrics = self.model.val(data=val_data_path)
        
#         print(f"📊 Model Evaluation Results:")
#         print(f"   mAP50: {metrics.box.map50:.4f}")
#         print(f"   mAP50-95: {metrics.box.map:.4f}")
        
#         # SỬA LỖI: Sử dụng attributes đúng từ metrics
#         if hasattr(metrics, 'speed'):
#             print(f"   Inference Speed: {metrics.speed['inference']:.1f}ms/img")
        
#         # In kết quả chi tiết cho từng class
#         if hasattr(metrics, 'results_dict'):
#             results_dict = metrics.results_dict
#             print(f"   Precision: {results_dict.get('metrics/precision(B)', 0):.4f}")
#             print(f"   Recall: {results_dict.get('metrics/recall(B)', 0):.4f}")
        
#         return metrics
    
#     def export_for_web(self, format='torchscript'):
#         """Export model cho web deployment"""
#         if self.model is None:
#             print("❌ No model available for export!")
#             return
        
#         # Load best model để export
#         best_model_path = os.path.join(DATASET_PATHS['model_save'], 'best_eye_detection_yolo.pt')
#         if not os.path.exists(best_model_path):
#             print("❌ Best model not found for export!")
#             return
            
#         model_for_export = YOLO(best_model_path)
        
#         try:
#             if format == 'torchscript':
#                 exported_path = model_for_export.export(format='torchscript')
#             elif format == 'onnx':
#                 exported_path = model_for_export.export(format='onnx')
#             else:
#                 exported_path = model_for_export.export(format='pt')  # PyTorch
            
#             # Copy đến vị trí cuối cùng
#             final_export_path = os.path.join(DATASET_PATHS['model_save'], f'eye_detection_web.{format}')
#             shutil.copy2(exported_path, final_export_path)
            
#             print(f"✅ Model exported for web: {final_export_path}")
#             return final_export_path
            
#         except Exception as e:
#             print(f"❌ Export failed: {e}")
#             return None

# # ==================== REAL-TIME TESTING ====================
# class RealTimeTester:
#     def __init__(self, model_path):
#         self.model = YOLO(model_path)
#         self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
#     def test_webcam(self):
#         """Test real-time với webcam"""
#         print("🎥 Starting real-time webcam test...")
        
#         cap = cv2.VideoCapture(0)
#         if not cap.isOpened():
#             print("❌ Cannot open webcam!")
#             return
        
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
            
#             # Detect faces
#             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
#             for (x, y, w, h) in faces:
#                 # Extract face ROI
#                 face_roi = frame[y:y+h, x:x+w]
                
#                 # Run YOLO detection on face ROI
#                 results = self.model(face_roi, verbose=False)
                
#                 for result in results:
#                     boxes = result.boxes
#                     if boxes is not None:
#                         for box in boxes:
#                             cls = int(box.cls[0])
#                             conf = float(box.conf[0])
                            
#                             if conf > 0.5:  # Confidence threshold
#                                 label = "OPEN" if cls == 0 else "CLOSED"
#                                 color = (0, 255, 0) if cls == 0 else (0, 0, 255)
                                
#                                 # Draw on original frame
#                                 cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
#                                 cv2.putText(frame, f'{label} {conf:.2f}', 
#                                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
#                                           0.7, color, 2)
            
#             cv2.imshow('Real-time Eye Detection - YOLO', frame)
            
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
        
#         cap.release()
#         cv2.destroyAllWindows()

# # ==================== MAIN ====================
# def main():
#     print("👁️ YOLO EYE DETECTION TRAINING")
#     print("=" * 60)
#     print("🎯 Training YOLO model for eye open/closed detection")
#     print("💾 Will export .pt file for web deployment")
#     print("=" * 60)
    
#     try:
#         # Khởi tạo trainer
#         trainer = YOLOEyeTrainer(model_size='n')  # n = nano (nhỏ, nhanh)
        
#         # Training
#         results = trainer.train_model(epochs=100, batch_size=16)
        
#         # Đánh giá
#         trainer.evaluate_model()
        
#         # Export cho web
#         trainer.export_for_web(format='torchscript')  # Hoặc 'onnx', 'pt'
        
#         print(f"\n🎊 YOLO TRAINING COMPLETED!")
#         print(f"📁 Model saved in: {DATASET_PATHS['model_save']}")
#         print("🚀 Ready for web deployment!")
        
#     except Exception as e:
#         print(f"❌ Training failed: {e}")
#         import traceback
#         traceback.print_exc()

# def test_real_time():
#     """Test real-time detection"""
#     model_path = os.path.join(DATASET_PATHS['model_save'], 'best_eye_detection_yolo.pt')
    
#     if os.path.exists(model_path):
#         tester = RealTimeTester(model_path)
#         tester.test_webcam()
#     else:
#         print("❌ Model not found! Please train first.")

# if __name__ == "__main__":
#     main()
    
#     # Uncomment để test real-time sau khi training
#     # test_real_time()


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import shutil
from datetime import datetime
import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
import yaml
from tqdm import tqdm
import glob

# ==================== CONFIGURATION ====================
DATASET_PATHS = {
    'open_eyes': '/home/gess/Documents/Data/Open_Eyes/',
    'closed_eyes': '/home/gess/Documents/Data/Closed_Eyes/', 
    'real_webcam': '/home/gess/Pictures/Webcam/',
    'additional_webcam': '/home/gess/Documents/Data/Additional_Webcam/',
    'videos': '/home/gess/Documents/Data/Fold1_part2/',
    'new_videos': '/home/gess/Documents/Data/New_Videos/',
    'yolo_dataset': '/home/gess/Documents/Data/YOLO_Dataset/',
    'model_save': '/home/gess/Documents/sub/Py/hhehee/eye_detection_model/',
    'output': '/home/gess/Documents/sub/Py/hhehee/eye_detection_results/'
}

# Tạo thư mục YOLO dataset structure
YOLO_DIRS = ['images/train', 'images/val', 'labels/train', 'labels/val']
for dir_name in YOLO_DIRS:
    os.makedirs(os.path.join(DATASET_PATHS['yolo_dataset'], dir_name), exist_ok=True)

for path in DATASET_PATHS.values():
    if path.endswith('/'):
        os.makedirs(path, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# ==================== YOLO DATA PROCESSOR ====================
class YOLODataProcessor:
    def __init__(self, img_size=640):
        self.img_size = img_size
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
    def prepare_yolo_dataset(self):
        """Chuẩn bị dataset cho YOLO training"""
        print("🔄 Preparing YOLO dataset...")
        
        all_images = []
        all_labels = []
        
        # 1. Process basic images
        print("📁 Processing basic images...")
        all_images.extend(self._process_basic_images())
        
        # 2. Process webcam images
        print("📸 Processing webcam images...")
        all_images.extend(self._process_webcam_images())
        
        # 3. Process videos
        print("🎥 Processing videos...")
        all_images.extend(self._process_videos())
        
        # Split data và tạo YOLO format
        self._create_yolo_dataset(all_images)
        
    def _process_basic_images(self):
        """Xử lý ảnh cơ bản"""
        images_info = []
        
        # Open eyes
        open_files = glob.glob(os.path.join(DATASET_PATHS['open_eyes'], '*.jpg')) + \
                    glob.glob(os.path.join(DATASET_PATHS['open_eyes'], '*.png'))
        
        for img_path in open_files[:1000]:  # Giới hạn 1000 ảnh mỗi class
            images_info.append({
                'path': img_path,
                'label': 0,  # 0 = open eyes
                'source': 'basic_open'
            })
        
        # Closed eyes
        closed_files = glob.glob(os.path.join(DATASET_PATHS['closed_eyes'], '*.jpg')) + \
                      glob.glob(os.path.join(DATASET_PATHS['closed_eyes'], '*.png'))
        
        for img_path in closed_files[:1000]:
            images_info.append({
                'path': img_path,
                'label': 1,  # 1 = closed eyes
                'source': 'basic_closed'
            })
        
        return images_info
    
    def _process_webcam_images(self):
        """Xử lý ảnh webcam"""
        images_info = []
        webcam_folders = [DATASET_PATHS['real_webcam'], DATASET_PATHS['additional_webcam']]
        
        for folder in webcam_folders:
            if not os.path.exists(folder):
                continue
                
            for img_path in glob.glob(os.path.join(folder, '*.jpg')) + glob.glob(os.path.join(folder, '*.png')):
                filename = os.path.basename(img_path).lower()
                
                if any(keyword in filename for keyword in ['open', 'mo', 'o_']):
                    label = 0
                elif any(keyword in filename for keyword in ['closed', 'dong', 'c_', 'nhammat']):
                    label = 1
                else:
                    continue
                
                images_info.append({
                    'path': img_path,
                    'label': label,
                    'source': 'webcam'
                })
        
        return images_info
    
    def _process_videos(self):
        """Xử lý video - trích xuất frames"""
        images_info = []
        video_folders = [DATASET_PATHS['videos'], DATASET_PATHS['new_videos']]
        
        for folder in video_folders:
            if not os.path.exists(folder):
                continue
                
            for video_path in glob.glob(os.path.join(folder, '*.mp4')) + \
                             glob.glob(os.path.join(folder, '*.avi')) + \
                             glob.glob(os.path.join(folder, '*.mov')):
                
                # Xác định label từ tên video
                video_name = os.path.basename(video_path).lower()
                if any(keyword in video_name for keyword in ['open', 'mo']):
                    label = 0
                elif any(keyword in video_name for keyword in ['closed', 'dong', 'nhammat']):
                    label = 1
                else:
                    continue
                
                # Trích xuất frames
                frames = self._extract_frames_from_video(video_path, max_frames=20)
                for i, frame_path in enumerate(frames):
                    images_info.append({
                        'path': frame_path,
                        'label': label,
                        'source': 'video'
                    })
        
        return images_info
    
    def _extract_frames_from_video(self, video_path, max_frames=20):
        """Trích xuất frames từ video"""
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return frames
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, int(fps))  # 1 frame mỗi giây
        
        frame_count = 0
        saved_count = 0
        
        while saved_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                # Lưu frame tạm thời
                frame_filename = f"temp_frame_{os.path.basename(video_path)}_{saved_count}.jpg"
                frame_path = os.path.join('/tmp', frame_filename)
                cv2.imwrite(frame_path, frame)
                frames.append(frame_path)
                saved_count += 1
            
            frame_count += 1
        
        cap.release()
        return frames
    
    def _create_yolo_dataset(self, images_info):
        """Tạo dataset format YOLO"""
        print("📦 Creating YOLO dataset format...")
        
        if len(images_info) == 0:
            print("❌ No images found for dataset creation!")
            return
        
        # Split data
        train_data, val_data = train_test_split(
            images_info, test_size=0.2, random_state=42, 
            stratify=[img['label'] for img in images_info]
        )
        
        # Tạo YOLO format cho train và val
        self._create_yolo_split(train_data, 'train')
        self._create_yolo_split(val_data, 'val')
        
        # Tạo file data.yaml
        self._create_yaml_config()
        
        print(f"✅ YOLO dataset created! Train: {len(train_data)}, Val: {len(val_data)}")
    
    def _create_yolo_split(self, data, split_type):
        """Tạo dữ liệu cho train/val split"""
        image_dir = os.path.join(DATASET_PATHS['yolo_dataset'], 'images', split_type)
        label_dir = os.path.join(DATASET_PATHS['yolo_dataset'], 'labels', split_type)
        
        for i, img_info in enumerate(tqdm(data, desc=f"Processing {split_type}")):
            try:
                # Đọc và xử lý ảnh
                img = cv2.imread(img_info['path'])
                if img is None:
                    continue
                
                # Resize ảnh
                img_resized = cv2.resize(img, (self.img_size, self.img_size))
                
                # Lưu ảnh
                img_filename = f"{split_type}_{i:06d}.jpg"
                img_save_path = os.path.join(image_dir, img_filename)
                cv2.imwrite(img_save_path, img_resized)
                
                # Tạo label file (YOLO format)
                label_filename = f"{split_type}_{i:06d}.txt"
                label_save_path = os.path.join(label_dir, label_filename)
                
                # YOLO format: class x_center y_center width height (normalized)
                # Với ảnh eye crop, coi như toàn bộ ảnh là bounding box
                with open(label_save_path, 'w') as f:
                    # class_id, x_center, y_center, width, height (all normalized)
                    f.write(f"{img_info['label']} 0.5 0.5 1.0 1.0\n")
                    
            except Exception as e:
                print(f"❌ Error processing {img_info['path']}: {e}")
                continue
    
    def _create_yaml_config(self):
        """Tạo file cấu hình YOLO"""
        yaml_content = {
            'path': DATASET_PATHS['yolo_dataset'],
            'train': 'images/train',
            'val': 'images/val',
            'nc': 2,  # number of classes
            'names': ['open_eye', 'closed_eye']  # class names
        }
        
        yaml_path = os.path.join(DATASET_PATHS['yolo_dataset'], 'data.yaml')
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
        
        print(f"✅ YAML config created: {yaml_path}")

# ==================== YOLO MODEL TRAINER ====================
class YOLOEyeTrainer:
    def __init__(self, model_size='n'):  # n, s, m, l, x
        self.model_size = model_size
        self.model = None
        self.data_processor = YOLODataProcessor()
        
    def prepare_data(self):
        """Chuẩn bị dữ liệu cho training"""
        print("🔄 Preparing training data...")
        self.data_processor.prepare_yolo_dataset()
        
    def train_model(self, epochs=100, imgsz=640, batch_size=16):
        """Training YOLO model"""
        print("🚀 Starting YOLO Training...")
        
        # Chuẩn bị dữ liệu
        self.prepare_data()
        
        # Load YOLO model
        model_name = f'yolov8{self.model_size}.pt'
        self.model = YOLO(model_name)
        
        # Training configuration
        training_args = {
            'data': os.path.join(DATASET_PATHS['yolo_dataset'], 'data.yaml'),
            'epochs': epochs,
            'imgsz': imgsz,
            'batch': batch_size,
            'patience': 20,
            'save': True,
            'exist_ok': True,
            'pretrained': True,
            'optimizer': 'AdamW',
            'lr0': 0.001,
            'weight_decay': 0.0005,
            'device': '0' if torch.cuda.is_available() else 'cpu',
            'workers': 4,
            'project': DATASET_PATHS['model_save'],
            'name': f'yolov8{self.model_size}_eye_detection',
            'verbose': True
        }
        
        print(f"📊 Training Configuration:")
        print(f"   Model: YOLOv8{self.model_size}")
        print(f"   Epochs: {epochs}")
        print(f"   Image size: {imgsz}")
        print(f"   Batch size: {batch_size}")
        print(f"   Device: {training_args['device']}")
        
        # Start training
        results = self.model.train(**training_args)
        
        # Save best model
        self._save_best_model()
        
        return results
    
    def _save_best_model(self):
        """Lưu best model và convert sang format phù hợp"""
        # Model sẽ tự động lưu trong thư mục runs
        # Copy best model đến thư mục model_save
        best_model_path = self.model.ckpt_path
        
        if best_model_path and os.path.exists(best_model_path):
            # Copy best model
            final_model_path = os.path.join(
                DATASET_PATHS['model_save'], 
                'best_eye_detection_yolo.pt'
            )
            shutil.copy2(best_model_path, final_model_path)
            print(f"✅ Best model saved: {final_model_path}")
        
    def evaluate_model(self):
        """Đánh giá model - ĐÃ SỬA LỖI"""
        if self.model is None:
            print("❌ No model available for evaluation!")
            return
        
        # Validation dataset path
        val_data_path = os.path.join(DATASET_PATHS['yolo_dataset'], 'data.yaml')
        
        # Evaluate
        metrics = self.model.val(data=val_data_path)
        
        print(f"📊 Model Evaluation Results:")
        print(f"   mAP50: {metrics.box.map50:.4f}")
        print(f"   mAP50-95: {metrics.box.map:.4f}")
        
        # SỬA LỖI: Sử dụng attributes đúng từ metrics
        if hasattr(metrics, 'speed'):
            print(f"   Inference Speed: {metrics.speed['inference']:.1f}ms/img")
        
        # In kết quả chi tiết cho từng class
        if hasattr(metrics, 'results_dict'):
            results_dict = metrics.results_dict
            print(f"   Precision: {results_dict.get('metrics/precision(B)', 0):.4f}")
            print(f"   Recall: {results_dict.get('metrics/recall(B)', 0):.4f}")
        
        return metrics
    
    def export_for_web(self, format='torchscript'):
        """Export model cho web deployment"""
        if self.model is None:
            print("❌ No model available for export!")
            return
        
        # Load best model để export
        best_model_path = os.path.join(DATASET_PATHS['model_save'], 'best_eye_detection_yolo.pt')
        if not os.path.exists(best_model_path):
            print("❌ Best model not found for export!")
            return
            
        model_for_export = YOLO(best_model_path)
        
        try:
            if format == 'torchscript':
                exported_path = model_for_export.export(format='torchscript')
            elif format == 'onnx':
                exported_path = model_for_export.export(format='onnx')
            else:
                exported_path = model_for_export.export(format='pt')  # PyTorch
            
            # Copy đến vị trí cuối cùng
            final_export_path = os.path.join(DATASET_PATHS['model_save'], f'eye_detection_web.{format}')
            shutil.copy2(exported_path, final_export_path)
            
            print(f"✅ Model exported for web: {final_export_path}")
            return final_export_path
            
        except Exception as e:
            print(f"❌ Export failed: {e}")
            return None

# ==================== REAL-TIME TESTING ====================
class RealTimeTester:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
    def test_webcam(self):
        """Test real-time với webcam"""
        print("🎥 Starting real-time webcam test...")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot open webcam!")
            return
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect faces
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            for (x, y, w, h) in faces:
                # Extract face ROI
                face_roi = frame[y:y+h, x:x+w]
                
                # Run YOLO detection on face ROI
                results = self.model(face_roi, verbose=False)
                
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            cls = int(box.cls[0])
                            conf = float(box.conf[0])
                            
                            if conf > 0.5:  # Confidence threshold
                                label = "OPEN" if cls == 0 else "CLOSED"
                                color = (0, 255, 0) if cls == 0 else (0, 0, 255)
                                
                                # Draw on original frame
                                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                                cv2.putText(frame, f'{label} {conf:.2f}', 
                                          (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                                          0.7, color, 2)
            
            cv2.imshow('Real-time Eye Detection - YOLO', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

# ==================== MAIN ====================
def main():
    print("👁️ YOLO EYE DETECTION TRAINING")
    print("=" * 60)
    print("🎯 Training YOLO model for eye open/closed detection")
    print("💾 Will export .pt file for web deployment")
    print("=" * 60)
    
    try:
        # Khởi tạo trainer
        trainer = YOLOEyeTrainer(model_size='n')  # n = nano (nhỏ, nhanh)
        
        # Training
        results = trainer.train_model(epochs=100, batch_size=16)
        
        # Đánh giá
        trainer.evaluate_model()
        
        # Export cho web
        trainer.export_for_web(format='torchscript')  # Hoặc 'onnx', 'pt'
        
        print(f"\n🎊 YOLO TRAINING COMPLETED!")
        print(f"📁 Model saved in: {DATASET_PATHS['model_save']}")
        print("🚀 Ready for web deployment!")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

def test_real_time():
    """Test real-time detection"""
    model_path = os.path.join(DATASET_PATHS['model_save'], 'best_eye_detection_yolo.pt')
    
    if os.path.exists(model_path):
        tester = RealTimeTester(model_path)
        tester.test_webcam()
    else:
        print("❌ Model not found! Please train first.")

if __name__ == "__main__":
    main()
    
    # Uncomment để test real-time sau khi training
    # test_real_time()