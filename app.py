# app.py
import os
import google.generativeai as genai
from flask import Flask, request, jsonify, render_template 
from flask_cors import CORS
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import cv2  # Tùy chọn cho xử lý ảnh bổ sung

load_dotenv()
app = Flask(__name__)
CORS(app)
import gunicorn

# Cấu hình upload
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# Cấu hình Gemini cho chat (giữ nguyên)
API_KEY = os.getenv("GOOGLE_API_KEY")
try:
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemini-2.5-flash') 
except Exception as e:
    print(f"Lỗi khi cấu hình Gemini: {e}")
    model = None 

# Load model YOLOv8 cho nhận diện biển báo
yolo_model = None
model_path = 'models/best.pt'

try:
    if os.path.exists(model_path):
        yolo_model = YOLO(model_path)
        print(f"✓ Model YOLOv8 loaded successfully from {model_path}")
    else:
        print(f"⚠ Model file not found at {model_path}")
        print("  Attempting to use pretrained YOLOv8n model...")
        yolo_model = YOLO('yolov8n.pt')  # Fallback to pretrained model
        print("✓ Pretrained YOLOv8n model loaded successfully")
except Exception as e:
    print(f"✗ Error loading YOLO model: {e}")
    print("  Traffic sign detection will not be available")
    yolo_model = None

# Hàm nhận diện biển báo với YOLOv8
def detect_sign(image_path):
    if not yolo_model:
        return "Lỗi: Model YOLOv8 chưa được load. Vui lòng kiểm tra file model tại thư mục 'models/'."
    
    try:
        # Chạy prediction với YOLOv8
        results = yolo_model.predict(
            source=image_path, 
            conf=0.25,  # Confidence threshold
            iou=0.45,   # IoU threshold for NMS
            save=False,
            verbose=False
        )
        
        detections = []
        for result in results:
            if len(result.boxes) == 0:
                continue
                
            for box in result.boxes:
                class_id = int(box.cls.item())
                confidence = box.conf.item()
                class_name = result.names[class_id]
                
                # Lấy tọa độ bounding box
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                detections.append({
                    'class': class_name,
                    'confidence': confidence,
                    'bbox': [x1, y1, x2, y2]
                })
        
        if not detections:
            return "❌ Không nhận diện được biển báo nào trong hình ảnh.\n\nGợi ý:\n- Đảm bảo hình ảnh có chứa biển báo giao thông\n- Biển báo cần rõ ràng và không bị che khuất\n- Thử với hình ảnh có độ phân giải tốt hơn"
        
        # Format kết quả đẹp hơn
        result_text = f"✓ Đã nhận diện được {len(detections)} biển báo:\n\n"
        for i, det in enumerate(detections, 1):
            result_text += f"{i}. {det['class']}\n"
            result_text += f"   Độ tin cậy: {det['confidence']*100:.1f}%\n"
            result_text += f"   Vị trí: ({int(det['bbox'][0])}, {int(det['bbox'][1])}) → ({int(det['bbox'][2])}, {int(det['bbox'][3])})\n\n"
        
        return result_text.strip()
        
    except Exception as e:
        return f"❌ Lỗi khi xử lý hình ảnh: {str(e)}\n\nVui lòng thử lại với hình ảnh khác."

# Hàm kiểm tra file cho phép
def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route gốc
@app.route('/')
def index():
    return render_template('index.html')

# Route chat (giữ nguyên)
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
        print(f"Đã xảy ra lỗi trong handle_chat: {e}")
        return jsonify({"error": "Đã có lỗi xảy ra phía máy chủ."}), 500

# Route detect-sign mới
@app.route('/detect-sign', methods=['POST'])
def detect_sign_route():
    if 'file' not in request.files:
        return jsonify({"error": "Không có file được upload."}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Không có file được chọn."}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            result = detect_sign(filepath)
            os.remove(filepath)  # Xóa file sau xử lý
            return jsonify({"result": result})
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({"error": f"Lỗi khi xử lý hình ảnh: {str(e)}"}), 500
    else:
        return jsonify({"error": "Định dạng file không được hỗ trợ. Chỉ chấp nhận PNG, JPG, JPEG."}), 400

# Chạy app
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, port=port)
