import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import shutil
from datetime import datetime
import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
DATASET_PATHS = {
    'open_eyes': '/home/gess/Documents/Data/Open_Eyes/',
    'closed_eyes': '/home/gess/Documents/Data/Closed_Eyes/', 
    'real_webcam': '/home/gess/Pictures/Webcam/',
    'videos': '/home/gess/Documents/Data/Fold1_part2/',
    'model_save': '/home/gess/Documents/sub/Py/hhehee/eye_detection_model/',
    'output': '/home/gess/Documents/sub/Py/hhehee/eye_detection_results/'
}

# Tạo thư mục
for path in [DATASET_PATHS['model_save'], DATASET_PATHS['output']]:
    os.makedirs(path, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# ==================== ADVANCED MODEL ARCHITECTURE ====================
class AdvancedEyeDetectionModel(nn.Module):
    def __init__(self, dropout_rate=0.4):
        super(AdvancedEyeDetectionModel, self).__init__()
        
        # Feature Extraction với Residual Connections
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
        )
        self.conv1_skip = nn.Conv2d(1, 32, 1, bias=False)
        
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(dropout_rate/3)
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
        )
        self.conv2_skip = nn.Conv2d(32, 64, 1, bias=False)
        
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout2d(dropout_rate/2)
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
        )
        self.conv3_skip = nn.Conv2d(64, 128, 1, bias=False)
        
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout2d(dropout_rate)
        
        # Global Context
        self.global_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate/2),
            
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate/3),
            
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        # Residual block 1
        identity = self.conv1_skip(x)
        x = self.conv1(x)
        x += identity
        x = F.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Residual block 2
        identity = self.conv2_skip(x)
        x = self.conv2(x)
        x += identity
        x = F.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Residual block 3
        identity = self.conv3_skip(x)
        x = self.conv3(x)
        x += identity
        x = F.relu(x)
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # Global pooling and classification
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x

# ==================== DATA PROCESSING - WEBCAM & VIDEO ====================
class DataProcessor:
    def __init__(self, img_size=(64, 64)):
        self.img_size = img_size
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    def extract_eyes_from_webcam_images(self, webcam_folder):
        """Trích xuất mắt từ ảnh webcam"""
        print("📸 Processing webcam images...")
        
        eye_images = []
        labels = []
        file_info = []
        
        if not os.path.exists(webcam_folder):
            print("   ❌ Webcam folder not found")
            return eye_images, labels, file_info
        
        webcam_files = [f for f in os.listdir(webcam_folder) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"   📁 Found {len(webcam_files)} webcam images")
        
        for filename in webcam_files:
            img_path = os.path.join(webcam_folder, filename)
            img = cv2.imread(img_path)
            
            if img is None:
                continue
            
            # Xác định label từ tên file
            filename_lower = filename.lower()
            if any(keyword in filename_lower for keyword in ['open', 'mo', 'o_']):
                label = 1
                true_state = "OPEN"
            elif any(keyword in filename_lower for keyword in ['closed', 'dong', 'c_', 'nhammat']):
                label = 0
                true_state = "CLOSED"
            else:
                # Skip files với label không rõ ràng
                continue
            
            # Trích xuất mắt từ ảnh
            eyes = self._extract_eyes_from_image(img)
            
            for i, eye_img in enumerate(eyes):
                if eye_img is not None:
                    # Preprocess eye image
                    processed_eye = self._preprocess_eye(eye_img)
                    eye_images.append(processed_eye)
                    labels.append(label)
                    file_info.append(f"WEBCAM_{true_state}_{filename}_eye{i}")
            
            print(f"   ✅ {filename}: {true_state} - {len(eyes)} eyes extracted")
        
        print(f"   📊 Total webcam eyes: {len(eye_images)}")
        return eye_images, labels, file_info
    
    def extract_eyes_from_videos(self, video_folder, max_frames_per_video=50):
        """Trích xuất mắt từ video"""
        print("🎥 Processing videos...")
        
        eye_images = []
        labels = []
        file_info = []
        
        if not os.path.exists(video_folder):
            print("   ❌ Video folder not found")
            return eye_images, labels, file_info
        
        video_files = [f for f in os.listdir(video_folder) 
                      if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        
        print(f"   📁 Found {len(video_files)} video files")
        
        for video_file in video_files:
            video_path = os.path.join(video_folder, video_file)
            
            # Xác định label từ tên video
            video_file_lower = video_file.lower()
            if any(keyword in video_file_lower for keyword in ['open', 'mo']):
                label = 1
                true_state = "OPEN"
            elif any(keyword in video_file_lower for keyword in ['closed', 'dong', 'nhammat']):
                label = 0
                true_state = "CLOSED"
            else:
                print(f"   ⚠️ Unknown label for video: {video_file}")
                continue
            
            print(f"   🎬 Processing: {video_file} ({true_state})")
            
            # Trích xuất frames từ video
            frames_processed = 0
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                print(f"   ❌ Cannot open video: {video_file}")
                continue
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = max(1, int(fps / 2))  # Lấy frame mỗi 0.5 giây
            
            frame_count = 0
            while frames_processed < max_frames_per_video:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                if frame_count % frame_interval != 0:
                    continue
                
                # Trích xuất mắt từ frame
                eyes = self._extract_eyes_from_image(frame)
                
                for i, eye_img in enumerate(eyes):
                    if eye_img is not None:
                        processed_eye = self._preprocess_eye(eye_img)
                        eye_images.append(processed_eye)
                        labels.append(label)
                        file_info.append(f"VIDEO_{true_state}_{video_file}_frame{frame_count}_eye{i}")
                        frames_processed += 1
                
                if frames_processed >= max_frames_per_video:
                    break
            
            cap.release()
            print(f"   ✅ {video_file}: {frames_processed} frames processed")
        
        print(f"   📊 Total video eyes: {len(eye_images)}")
        return eye_images, labels, file_info
    
    def _extract_eyes_from_image(self, image):
        """Trích xuất mắt từ ảnh sử dụng face detection"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        eyes = []
        
        # Phát hiện khuôn mặt
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            
            # Phát hiện mắt trong khuôn mặt
            detected_eyes = self.eye_cascade.detectMultiScale(face_roi, 1.1, 4)
            
            for (ex, ey, ew, eh) in detected_eyes:
                # Trích xuất vùng mắt
                eye_roi = face_roi[ey:ey+eh, ex:ex+ew]
                
                # Kiểm tra chất lượng mắt (kích thước tối thiểu)
                if ew >= 20 and eh >= 10:
                    eyes.append(eye_roi)
        
        return eyes
    
    def _preprocess_eye(self, eye_img):
        """Tiền xử lý ảnh mắt"""
        # Resize về kích thước chuẩn
        resized = cv2.resize(eye_img, self.img_size)
        
        # CLAHE để cải thiện độ tương phản
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(resized)
        
        # Gaussian blur nhẹ
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        # Chuẩn hóa
        normalized = blurred.astype('float32') / 255.0
        
        return normalized

# ==================== ENHANCED DATASET CLASS ====================
class EnhancedEyeDataset(Dataset):
    def __init__(self, images, labels, transform=None, is_training=True):
        self.images = images
        self.labels = labels
        self.transform = transform
        self.is_training = is_training
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # Áp dụng augmentation
        if self.transform:
            # Chuyển đổi image sang uint8 cho albumentations
            image_uint8 = (image * 255).astype(np.uint8)
            augmented = self.transform(image=image_uint8)
            image = augmented['image'].float() / 255.0  # Chuyển lại về [0,1]
        else:
            image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        
        return image, torch.tensor(label, dtype=torch.float32)

# ==================== PROFESSIONAL TRAINER ====================
class ProfessionalEyeTrainer:
    def __init__(self):
        self.model = None
        self.img_size = (64, 64)
        self.best_accuracy = 0
        self.data_processor = DataProcessor(self.img_size)
        
    def load_all_data_sources(self):
        """Load dữ liệu từ TẤT CẢ nguồn: ảnh + webcam + video"""
        print("🔧 Loading data from ALL sources...")
        
        def load_basic_images(folder_path, label, source_name):
            images, count = [], 0
            if not os.path.exists(folder_path):
                return images
            
            files = [f for f in os.listdir(folder_path) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            for filename in files[:2000]:  # Giới hạn 2000 ảnh mỗi class
                img_path = os.path.join(folder_path, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                if img is not None:
                    # Preprocessing cơ bản
                    img = cv2.resize(img, self.img_size)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                    img = clahe.apply(img)
                    img = cv2.GaussianBlur(img, (3, 3), 0)
                    img = img.astype('float32') / 255.0
                    
                    images.append(img)
                    count += 1
            
            print(f"   ✅ {source_name}: {count} images")
            return images
        
        all_images, all_labels, all_sources = [], [], []
        
        # 1. Ảnh cơ bản (Open Eyes)
        print("\n📁 1. Basic Images:")
        open_images = load_basic_images(DATASET_PATHS['open_eyes'], 1, "OPEN_EYES")
        closed_images = load_basic_images(DATASET_PATHS['closed_eyes'], 0, "CLOSED_EYES")
        
        all_images.extend(open_images)
        all_labels.extend([1] * len(open_images))
        all_sources.extend(["BASIC_OPEN"] * len(open_images))
        
        all_images.extend(closed_images)
        all_labels.extend([0] * len(closed_images))
        all_sources.extend(["BASIC_CLOSED"] * len(closed_images))
        
        # 2. Ảnh Webcam (đã trích xuất mắt)
        print("\n📸 2. Webcam Images:")
        webcam_eyes, webcam_labels, webcam_info = self.data_processor.extract_eyes_from_webcam_images(
            DATASET_PATHS['real_webcam']
        )
        all_images.extend(webcam_eyes)
        all_labels.extend(webcam_labels)
        all_sources.extend(webcam_info)
        
        # 3. Video (đã trích xuất mắt)
        print("\n🎥 3. Video Frames:")
        video_eyes, video_labels, video_info = self.data_processor.extract_eyes_from_videos(
            DATASET_PATHS['videos'], max_frames_per_video=30
        )
        all_images.extend(video_eyes)
        all_labels.extend(video_labels)
        all_sources.extend(video_info)
        
        # Thống kê tổng quan
        print(f"\n📊 DATASET SUMMARY:")
        print(f"   👁️ Open eyes: {sum(all_labels)}")
        print(f"   🚫 Closed eyes: {len(all_labels) - sum(all_labels)}")
        print(f"   📦 Total images: {len(all_images)}")
        
        # Thống kê theo nguồn
        source_counts = {}
        for source in all_sources:
            main_source = source.split('_')[0]
            source_counts[main_source] = source_counts.get(main_source, 0) + 1
        
        print(f"   📍 Sources: {source_counts}")
        
        return np.array(all_images), np.array(all_labels), all_sources
    
    def train_complete_model(self):
        """Training hoàn chỉnh với tất cả dữ liệu"""
        print("\n🚀 Starting Complete Training with ALL Data Sources...")
        
        # Load tất cả dữ liệu
        X, y, sources = self.load_all_data_sources()
        
        if len(X) == 0:
            print("❌ No data available for training!")
            return None
        
        # Split data
        X_train, X_temp, y_train, y_temp, sources_train, sources_temp = train_test_split(
            X, y, sources, test_size=0.3, random_state=42, stratify=y
        )
        X_val, X_test, y_val, y_test, sources_val, sources_test = train_test_split(
            X_temp, y_temp, sources_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        
        print(f"\n📈 Data Split:")
        print(f"   🎯 Train: {len(X_train)}")
        print(f"   📊 Validation: {len(X_val)}")
        print(f"   🧪 Test: {len(X_test)}")
        
        # Data augmentation
        train_transforms = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(var_limit=(10.0, 30.0), p=0.2),
            A.Normalize(mean=[0.5], std=[0.5]),
            ToTensorV2(),
        ])
        
        val_transforms = A.Compose([
            A.Normalize(mean=[0.5], std=[0.5]),
            ToTensorV2(),
        ])
        
        # Tạo datasets
        train_dataset = EnhancedEyeDataset(X_train, y_train, transform=train_transforms, is_training=True)
        val_dataset = EnhancedEyeDataset(X_val, y_val, transform=val_transforms, is_training=False)
        test_dataset = EnhancedEyeDataset(X_test, y_test, transform=val_transforms, is_training=False)
        
        # Tạo dataloaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
        
        # Khởi tạo model
        self.model = AdvancedEyeDetectionModel().to(device)
        
        # Optimizer và loss
        optimizer = optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        criterion = nn.BCELoss()
        
        # Training
        best_val_loss = float('inf')
        patience, patience_counter = 15, 0
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        
        print("\n🎯 Training Progress:")
        for epoch in range(100):
            # Training
            self.model.train()
            train_loss, train_correct, train_total = 0, 0, 0
            
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output.squeeze(), target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                train_loss += loss.item()
                pred = (output > 0.5).float()
                train_correct += (pred.squeeze() == target).sum().item()
                train_total += target.size(0)
            
            # Validation
            self.model.eval()
            val_loss, val_correct, val_total = 0, 0, 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    output = self.model(data)
                    loss = criterion(output.squeeze(), target)
                    
                    val_loss += loss.item()
                    pred = (output > 0.5).float()
                    val_correct += (pred.squeeze() == target).sum().item()
                    val_total += target.size(0)
            
            # Tính metrics
            train_acc = 100 * train_correct / train_total
            val_acc = 100 * val_correct / val_total
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                self.best_accuracy = val_acc
                self._save_complete_model()
                print(f"💾 BEST Model saved! Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"🛑 Early stopping at epoch {epoch+1}")
                break
            
            if (epoch + 1) % 5 == 0:
                print(f'   Epoch {epoch+1:3d}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                      f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Đánh giá cuối cùng
        final_accuracy = self._evaluate_model(test_loader)
        self._plot_training_history(train_losses, val_losses, train_accs, val_accs)
        
        print(f"\n🎉 TRAINING COMPLETED!")
        print(f"✅ Final Test Accuracy: {final_accuracy:.2f}%")
        print(f"💾 Model saved to: {DATASET_PATHS['model_save']}")
        
        return final_accuracy
    
    def _save_complete_model(self):
        """Lưu model hoàn chỉnh"""
        model_path = os.path.join(DATASET_PATHS['model_save'], 'professional_eye_detector.pth')
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_architecture': 'AdvancedEyeDetectionModel',
            'img_size': self.img_size,
            'accuracy': self.best_accuracy,
            'timestamp': datetime.now().isoformat(),
        }, model_path)
    
    def _evaluate_model(self, test_loader):
        """Đánh giá model"""
        self.model.eval()
        all_preds, all_targets = [], []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = self.model(data)
                pred = (output > 0.5).float()
                
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        accuracy = 100 * np.mean(np.array(all_preds) == np.array(all_targets))
        
        # Print report
        print("\n📊 Classification Report:")
        print(classification_report(all_targets, all_preds, target_names=['CLOSED', 'OPEN']))
        
        return accuracy
    
    def _plot_training_history(self, train_losses, val_losses, train_accs, val_accs):
        """Vẽ biểu đồ training"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Training Loss', linewidth=2)
        plt.plot(val_losses, label='Validation Loss', linewidth=2)
        plt.title('Model Loss History', fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(train_accs, label='Training Accuracy', linewidth=2)
        plt.plot(val_accs, label='Validation Accuracy', linewidth=2)
        plt.title('Model Accuracy History', fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(DATASET_PATHS['model_save'], 'training_history.png'), dpi=300)
        plt.close()
        print("📈 Training history plot saved!")

# ==================== MAIN ====================
def main():
    print("👁️ PROFESSIONAL EYE DETECTION - COMPLETE TRAINING")
    print("=" * 60)
    print("📚 Training from: Basic Images + Webcam + Video")
    print("💾 Model will be saved for real-time detection")
    print("=" * 60)
    
    trainer = ProfessionalEyeTrainer()
    accuracy = trainer.train_complete_model()
    
    if accuracy:
        print(f"\n🎊 TRAINING SUCCESSFUL!")
        print(f"⭐ Final Accuracy: {accuracy:.2f}%")
        print(f"📁 Model: {DATASET_PATHS['model_save']}/professional_eye_detector.pth")
        print("🚀 Ready for real-time detection!")

if __name__ == "__main__":
    main()