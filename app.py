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

# Load model YOLOv8n cho nhận diện biển báo
yolo_model = None
try:
    yolo_model = YOLO('models/best.pt')  # Thay bằng đường dẫn thực tế model YOLOv8n, ví dụ: './models/yolov8n.pt'
    print("Model YOLOv8n loaded successfully.")
except Exception as e:
    print(f"Lỗi khi load model YOLOv8n: {e}")
    yolo_model = None

# Hàm nhận diện biển báo với YOLOv8n
def detect_sign(image_path):
    if not yolo_model:
        return "Lỗi: Model YOLOv8n chưa được load."
    
    try:
        # Chạy prediction với YOLOv8n (nano version, nhanh hơn)
        results = yolo_model.predict(source=image_path, conf=0.5, save=False)  # conf=0.5 để lọc confidence thấp
        
        detections = []
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls.item())
                confidence = box.conf.item()
                class_name = result.names[class_id]  # Giả định model có class names cho biển báo
                detections.append(f"{class_name} (Confidence: {confidence:.2f})")
        
        if not detections:
            return "Không nhận diện được biển báo nào trong hình ảnh."
        
        result_text = "Nhận diện biển báo:\n" + "\n".join(detections)
        return result_text
    except Exception as e:
        return f"Lỗi khi xử lý với YOLOv8n: {str(e)}"

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
            os.remove(filepath)
            return jsonify({"error": f"Lỗi khi xử lý hình ảnh: {str(e)}"}), 500
    else:
        return jsonify({"error": "Định dạng file không được hỗ trợ. Chỉ chấp nhận PNG, JPG, JPEG."}), 400

# Chạy app
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, port=port)