# app.py
import os
import yaml
import logging
import numpy as np  # Cần cho xử lý ảnh
import time
import re # xóa ảnh dấu
from pathlib import Path
from flask import send_from_directory
import google.generativeai as genai
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from ultralytics import YOLO

# Optional image processing if needed later
import cv2

# --- Config & logging ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

app = Flask(__name__)
CORS(app)
#YOLO ========================================================
# Upload folder
UPLOAD_FOLDER = Path("uploads")
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

# --- NEW: HÀM XÓA DẤU TIẾNG VIỆT ---
def remove_accents(input_str):
    """
    Xóa dấu Tiếng Việt khỏi một chuỗi.
    """
    s1 = u'ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰự'
    s0 = u'AAAAEEEIIOOOOUUYaaaaeeeiioooouuyAadDiIUuOoUuAaAaAaAaAaAaAaAaAaAaAaAaEeEeEeEeEeEeEeEeIiIiOoOoOoOoOoOoOoOoOoOoOoOoUuUuUuUuUuUuUu'
    s = ''
    try:
        input_str = input_str.decode('utf-8')
    except AttributeError:
        pass # Already utf-8
        
    for c in input_str:
        try:
            if c in s1:
                s += s0[s1.index(c)]
            else:
                s += c
        except:
             s += c
    # Bỏ các ký tự đặc biệt còn sót lại (ngoài chữ và số)
    s = re.sub(r'[^\w\s]', '', s)
    return s
# --- KẾT THÚC HÀM MỚI ---


# --- Gemini (chat) setup (keep as-is, protected) ---
API_KEY = os.getenv("GOOGLE_API_KEY")
try:
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemini-2.5-flash')
    logging.info("Gemini configured.")
except Exception as e:
    logging.warning(f"Không thể cấu hình Gemini: {e}")
    model = None

# --- Paths for YOLO model (TRAFFIC SIGN) & dataset.yaml ---
MODEL_PATH = Path("C:/Users/hovan/OneDrive/Desktop/AII/TRAFFIC_SIGNS/TRAFFIC_SIGNS/runs_yolo/yolov13_custom_train2/weights/best.pt")
DATASET_YAML = Path("C:/Users/hovan/OneDrive/Desktop/AII/TRAFFIC_SIGNS/TRAFFIC_SIGNS/dataset/dataset.yaml")

# --- Load class names from dataset.yaml ---
CLASS_NAMES = []
if DATASET_YAML.exists():
    try:
        with open(DATASET_YAML, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            if isinstance(data, dict) and "names" in data:
                CLASS_NAMES = data["names"]
            elif isinstance(data, dict) and "nc" in data and "names" in data:
                CLASS_NAMES = data["names"]
            else:
                logging.warning("Không tìm thấy trường 'names' trong dataset.yaml hoặc định dạng không chuẩn.")
    except Exception as e:
        logging.warning(f"Lỗi khi đọc dataset.yaml: {e}")
else:
    logging.warning(f"dataset.yaml không tồn tại tại: {DATASET_YAML}")

if CLASS_NAMES:
    logging.info(f"Đã load {len(CLASS_NAMES)} tên lớp (Biển báo) từ dataset.yaml")
else:
    logging.info("Danh sách CLASS_NAMES (Biển báo) trống.")

# --- Load YOLO model (TRAFFIC SIGN) once at startup ---
try:
    if MODEL_PATH.exists():
        yolo_model = YOLO(str(MODEL_PATH))
        logging.info(f"Model YOLO (Biển báo) loaded from {MODEL_PATH}")
    else:
        yolo_model = None
        logging.warning(f"Model file (Biển báo) not found at {MODEL_PATH}.")
except Exception as e:
    logging.error(f"Lỗi khi load YOLO model (Biển báo): {e}")
    yolo_model = None

# --- NEW: Tải model NHẬN DIỆN KHUÔN MẶT (Tiền xử lý) ---
PROTOTXT_PATH = Path("C:/Users/hovan/OneDrive/Desktop/AII/TrainModel/TrainModel/models/deploy.prototxt.txt")
CAFFEMODEL_PATH = Path("C:/Users/hovan/OneDrive/Desktop/AII/TrainModel/TrainModel/models/res10_300x300_ssd_iter_140000.caffemodel")
try:
    if PROTOTXT_PATH.exists() and CAFFEMODEL_PATH.exists():
        face_net = cv2.dnn.readNetFromCaffe(str(PROTOTXT_PATH), str(CAFFEMODEL_PATH))
        logging.info("Model Face Detector (OpenCV DNN) loaded.")
    else:
        face_net = None
        logging.warning(f"Không tìm thấy file Face Detector tại {PROTOTXT_PATH} hoặc {CAFFEMODEL_PATH}")
except Exception as e:
    face_net = None
    logging.error(f"Lỗi khi load Face Detector: {e}")

# --- NEW: Tải model YOLOv8 NGỦ GẬT (Model chính của bạn) ---
SLEEP_MODEL_PATH = Path("C:/Users/hovan/OneDrive/Desktop/AII/TrainModel/TrainModel/models/best_drowsy.pt") 
SLEEP_CLASSES_PATH = Path("C:/Users/hovan/OneDrive/Desktop/AII/TrainModel/TrainModel/YOLO_Dataset/data.yaml") 
try:
    if SLEEP_MODEL_PATH.exists():
        sleep_model = YOLO(str(SLEEP_MODEL_PATH))
        logging.info(f"Model YOLO (Ngủ gật) loaded from {SLEEP_MODEL_PATH}")
    else:
        sleep_model = None
        logging.warning(f"Không tìm thấy file model Ngủ gật tại {SLEEP_MODEL_PATH}")
    
    if SLEEP_CLASSES_PATH.exists():
        with open(SLEEP_CLASSES_PATH, 'r', encoding='utf-8') as f:
            sleep_classes_data = yaml.safe_load(f)
            SLEEP_CLASS_NAMES = sleep_classes_data['names']
            logging.info(f"Loaded {len(SLEEP_CLASS_NAMES)} sleep class names.")
    else:
        SLEEP_CLASS_NAMES = ['awake', 'sleepy'] 
        logging.warning(f"Không tìm thấy {SLEEP_CLASSES_PATH}, dùng fallback: {SLEEP_CLASS_NAMES}")
        
except Exception as e:
    sleep_model = None
    logging.error(f"Lỗi khi load YOLO model (Ngủ gật): {e}")

# --- Utility: allowed file types ---
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Detection function (TRAFFIC SIGN) ---
# === SỬA LỖI: VẼ TIẾNG VIỆT KHÔNG DẤU ===
def detect_sign(image_path: str, save_annotated: bool = False):
    if not yolo_model:
        return {
            "result_text": "❌ Lỗi: Model YOLO (Biển báo) chưa được load.",
            "detections": [],
            "image_path": None
        }

    try:
        img = cv2.imread(image_path)
        if img is None:
            return {
                "result_text": "❌ Lỗi: Không đọc được ảnh.",
                "detections": [],
                "image_path": None
            }

        results = yolo_model.predict(
            source=image_path,
            conf=0.25,
            iou=0.45,
            save=False,
            verbose=False
        )

        detections = []

        for result in results:
            if not hasattr(result, "boxes") or len(result.boxes) == 0:
                continue
            for box in result.boxes:
                class_id = int(box.cls.item())
                confidence = float(box.conf.item())

                # 1. Lấy tên Tiếng Việt (có dấu) để HIỂN THỊ TEXT
                if CLASS_NAMES and class_id < len(CLASS_NAMES):
                    display_name = CLASS_NAMES[class_id]
                else:
                    display_name = f"Class_{class_id}"
                
                # 2. TẠO TÊN KHÔNG DẤU ĐỂ VẼ
                # Sử dụng hàm remove_accents()
                draw_name = remove_accents(display_name)
                draw_label = f"{draw_name} {confidence*100:.1f}%" # Ví dụ: "Cam dung va do xe 93.0%"


                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                detections.append({
                    "class": display_name, # Trả về tên Tiếng Việt
                    "confidence": confidence,
                    "bbox": [x1, y1, x2, y2],
                })

                # 3. Vẽ lên ảnh bằng `draw_label` (đã bỏ dấu)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                (w, h), _ = cv2.getTextSize(draw_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
                cv2.putText(img, draw_label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        if not detections:
            return {
                "result_text": (
                    "❌ Không nhận diện được biển báo nào trong hình ảnh.\n\n"
                    "🔎 Gợi ý:\n- Đảm bảo hình ảnh có biển báo rõ ràng.\n- Biển báo không bị che khuất hoặc quá nhỏ."
                ),
                "detections": [],
                "image_path": None
            }

        image_annotated_name = Path(image_path).stem + "_annotated.jpg"
        image_annotated_path = Path(app.config['UPLOAD_FOLDER']) / image_annotated_name
        cv2.imwrite(str(image_annotated_path), img)

        result_text = f"✅ Đã nhận diện được {len(detections)} biển báo giao thông:\n\n"
        for i, det in enumerate(detections, 1):
            result_text += f"{i}. {det['class']}\n"
            result_text += f"   - Độ tin cậy: {det['confidence'] * 100:.2f}%\n"

        return {
            "result_text": result_text.strip(),
            "detections": detections,
            "image_path": str(image_annotated_path.name)
        }

    except Exception as e:
        logging.exception("Lỗi khi chạy predict (Biển báo):")
        return {
            "result_text": f"❌ Lỗi khi xử lý hình ảnh (Biển báo): {e}",
            "detections": [],
            "image_path": None
        }
# === HẾT PHẦN SỬA ===


# --- Routes ---
@app.route('/uploads/<filename>')
def uploaded_file(filename): 
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def handle_chat():
    if not model:
        return jsonify({"error": "Mô hình AI chưa được khởi tạo, vui lòng kiểm tra API Key."}), 500
    try:
        user_data = request.get_json()
        user_input = user_data.get('message')
        history = user_data.get('history', [])
        if not user_input:
            return jsonify({"error": "Không nhận được tin nhắn."}), 400
        chat_session = model.start_chat(history=history)
        response = chat_session.send_message(user_input)
        return jsonify({"response": response.text})
    except Exception as e:
        logging.exception("Lỗi handle_chat:")
        return jsonify({"error": "Đã có lỗi phía server."}), 500
#... toàn bộ logic của handle_chat.
@app.route('/detect-sign', methods=['POST'])
def detect_sign_route():
    if 'file' not in request.files:
        return jsonify({"error": "Không có file được upload."}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Không có file được chọn."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Định dạng file không được hỗ trợ. Chỉ chấp nhận PNG, JPG, JPEG."}), 400

    filename = secure_filename(file.filename)
    save_path = Path(app.config['UPLOAD_FOLDER']) / filename
    try:
        file.save(str(save_path))
        detection_result = detect_sign(str(save_path), save_annotated=True)

        try:
            if save_path.exists():
                save_path.unlink()
        except Exception:
            logging.warning("Không xóa được file upload tạm thời (biển báo).")

        return jsonify({
            "result": detection_result.get("result_text"),
            "detections": detection_result.get("detections"),
            "image_path": detection_result.get("image_path")
        })
    except Exception as e:
        logging.exception("Lỗi trong route /detect-sign:")
        try:
            if save_path.exists():
                save_path.unlink()
        except Exception:
            pass
        return jsonify({"error": f"Lỗi khi xử lý file: {e}"}), 500

# --- ROUTE CHO NHẬN DIỆN NGỦ GẬT (ĐÃ SỬA LỖI VẼ BOX) ---
@app.route('/detect-sleep', methods=['POST'])
def detect_sleep_route():
    # 1. Kiểm tra model đã được tải chưa
    if not sleep_model or not face_net:
        return jsonify({"error": "Model Ngủ gật hoặc Face Detector chưa sẵn sàng."}), 500

    # 2. Kiểm tra file upload
    if 'file' not in request.files:
        return jsonify({"error": "Không có file được upload."}), 400
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"error": "File không hợp lệ."}), 400

    filename = secure_filename(file.filename)
    save_path = Path(app.config['UPLOAD_FOLDER']) / filename
    try:
        file.save(str(save_path))

        # 3. Đọc ảnh bằng OpenCV
        image = cv2.imread(str(save_path))
        if image is None:
            return jsonify({"error": "Không thể đọc file ảnh."}), 400
        
        (h, w) = image.shape[:2]

        # 4. TIỀN XỬ LÝ: Nhận diện khuôn mặt
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
        face_net.setInput(blob)
        detections = face_net.forward()

        best_face = None
        best_confidence = 0.0

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5: # Ngưỡng tin cậy
                if confidence > best_confidence:
                    best_confidence = confidence
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    (startX, startY) = (max(0, startX), max(0, startY))
                    (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
                    best_face = (startX, startY, endX, endY)

        if best_face is None:
            return jsonify({
                "result": "❌ Không tìm thấy khuôn mặt",
                "image_path": None
            })

        # 5. CROP KHUÔN MẶT
        (startX, startY, endX, endY) = best_face
        face_width = endX - startX
        if face_width < 10: # Nếu chiều rộng mặt < 10px -> lỗi
             return jsonify({
                "result": f"❌ Lỗi phát hiện khuôn mặt (width: {face_width}px)",
                "image_path": None
            })

        face_roi = image[startY:endY, startX:endX]
        if face_roi.size == 0:
             return jsonify({
                "result": "❌ Kích thước khuôn mặt không hợp lệ",
                "image_path": None
            })

        # 6. CHẠY YOLOv8 (OBJECT DETECTION) TRÊN KHUÔN MẶT ĐÃ CROP
        results = sleep_model.predict(source=face_roi, verbose=False)
        
        result_text = ""
        is_sleepy = False # Cờ để xác định trạng thái
        
        # Logic cho Object Detection
        if hasattr(results[0], 'boxes') and results[0].boxes is not None:
            
            if len(results[0].boxes) == 0:
                result_text = "✅ PHÁT HIỆN: TỈNH TÁO"
                color = (0, 255, 0) # Xanh
                
                draw_label = "AWAKE (No detections)"
                cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
                cv2.putText(image, draw_label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            else:
                for box in results[0].boxes:
                    class_id = int(box.cls.item())
                    confidence = float(box.conf.item())
                    
                    try:
                        class_name = SLEEP_CLASS_NAMES[class_id]
                    except IndexError:
                        class_name = "unknown"

                    # GIẢ ĐỊNH: Class 'Ngủ gật' (ví dụ: 'mat_nham') của bạn là index 1
                    if class_id == 1: 
                        is_sleepy = True
                        
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                    x1_abs = x1 + startX
                    y1_abs = y1 + startY
                    x2_abs = x2 + startX
                    y2_abs = y2 + startY
                    
                    box_color = (0, 0, 255) if class_id == 1 else (0, 255, 0)
                    
                    draw_label = f"{class_name} {confidence*100:.0f}%"
                    cv2.rectangle(image, (x1_abs, y1_abs), (x2_abs, y2_abs), box_color, 1)
                    cv2.putText(image, draw_label, (x1_abs, y1_abs - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

                if is_sleepy:
                    result_text = "❌ PHÁT HIỆN: CÓ NGỦ GẬT"
                    color = (0, 0, 255) # Đỏ
                    draw_label = "SLEEPY" 
                else:
                    result_text = "✅ PHÁT HIỆN: TỈNH TÁO"
                    color = (0, 255, 0) # Xanh
                    draw_label = "AWAKE" 

                cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
                cv2.putText(image, draw_label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        else:
            result_text = "❌ Lỗi: Model không phải là Object Detection hoặc không trả về kết quả."

        # 8. Lưu ảnh đã annotate và trả về
        image_annotated_name = Path(filename).stem + "_sleep_annotated.jpg"
        image_annotated_path = Path(app.config['UPLOAD_FOLDER']) / image_annotated_name
        cv2.imwrite(str(image_annotated_path), image)

        try:
            if save_path.exists():
                save_path.unlink()
        except Exception:
            logging.warning("Không xóa được file upload tạm thời (ngủ gật).")

        return jsonify({
            "result": result_text, 
            "image_path": str(image_annotated_name)
        })

    except Exception as e:
        logging.exception("Lỗi trong route /detect-sleep:")
        try:
            if save_path.exists():
                save_path.unlink()
        except Exception:
            pass
        return jsonify({"error": f"Lỗi khi xử lý file: {e}"}), 500

# --- Run app ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, port=port)