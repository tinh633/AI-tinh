import os
import logging
import time
import re
import json
import base64
import yaml
import cv2
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import google.generativeai as genai
from openai import OpenAI

# ===== IMPORT MODULE TRA CỨU LUẬT =====
try:
    from law_search import search_vbpl_sync
except ImportError:
    logging.warning("⚠️ Chưa tìm thấy file law_search.py. Chức năng tra cứu luật sẽ lỗi.")
    def search_vbpl_sync(q): return "Lỗi: Thiếu file module law_search.py"

# --- Cấu hình & Logging ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

app = Flask(__name__)
CORS(app)

# Cấu hình thư mục upload
UPLOAD_FOLDER = Path("uploads")
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def remove_accents(input_str):
    """
    Xóa dấu Tiếng Việt khỏi một chuỗi để vẽ lên ảnh OpenCV (tránh lỗi ???).
    """
    if not input_str:
        return ""
    s1 = u'ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰự'
    s0 = u'AAAAEEEIIOOOOUUYaaaaeeeiioooouuyAadDiIUuOoUuAaAaAaAaAaAaAaAaAaAaAaAaEeEeEeEeEeEeEeEeIiIiOoOoOoOoOoOoOoOoOoOoOoOoUuUuUuUuUuUuUu'
    s = ''
    try:
        input_str = str(input_str)
    except:
        return str(input_str)
        
    for c in input_str:
        if c in s1:
            s += s0[s1.index(c)]
        else:
            s += c
    return s

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- 1. CẤU HÌNH AI CHAT ---
try:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')
except Exception as e:
    gemini_model = None
    logging.warning(f"⚠️ Lỗi cấu hình Gemini: {e}")

try:
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except Exception as e:
    openai_client = None
    logging.warning(f"⚠️ Lỗi cấu hình OpenAI: {e}")

try:
    deepseek_client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY")
    )
except Exception as e:
    deepseek_client = None
    logging.warning(f"⚠️ Lỗi cấu hình DeepSeek: {e}")

# --- 2. LOAD MODEL COMPUTER VISION ---

# A. Model Biển Báo
SIGN_MODEL_PATH = Path("C:/Users/hovan/OneDrive/Desktop/AII/TRAFFIC_SIGNS/runs_yolo/yolov13_custom_train2/weights/best.pt")
try:
    if SIGN_MODEL_PATH.exists():
        sign_model = YOLO(str(SIGN_MODEL_PATH))
        logging.info("✅ Loaded Sign Model")
    else:
        sign_model = None
        logging.error(f"❌ Sign Model not found at {SIGN_MODEL_PATH}")
except Exception as e:
    sign_model = None
    logging.error(f"❌ Error loading Sign Model: {e}")

# B. Model Ngủ Gật
SLEEP_MODEL_PATH = Path("C:/Users/hovan/OneDrive/Desktop/AII/TrainModel/models/best_drowsy.pt")
try:
    if SLEEP_MODEL_PATH.exists():
        sleep_model = YOLO(str(SLEEP_MODEL_PATH))
        logging.info("✅ Loaded Sleep Model")
    else:
        sleep_model = None
        logging.error(f"❌ Sleep Model not found at {SLEEP_MODEL_PATH}")
except Exception as e:
    sleep_model = None
    logging.error(f"❌ Error loading Sleep Model: {e}")

# Define sleep detection class names
SLEEP_CLASS_NAMES = {
    0: "alert",
    1: "mat_nham"  # Drowsy/sleepy
}

# Load face detector (OpenCV DNN)
FACE_PROTO = Path("C:/Users/hovan/OneDrive/Desktop/AII/TrainModel/models/deploy.prototxt.txt")
FACE_MODEL = Path("C:/Users/hovan/OneDrive/Desktop/AII/TrainModel/models/res10_300x300_ssd_iter_140000.caffemodel")
try:
    if FACE_PROTO.exists() and FACE_MODEL.exists():
        face_net = cv2.dnn.readNetFromCaffe(str(FACE_PROTO), str(FACE_MODEL))
        logging.info("✅ Loaded Face Detector (DNN)")
    else:
        face_net = None
        logging.warning(f"⚠️ Face detector files not found at {FACE_PROTO} and {FACE_MODEL}; face detection disabled.")
except Exception as e:
    face_net = None
    logging.error(f"❌ Error loading face detector: {e}")


# --- 3. HÀM XỬ LÝ LOGIC (Backend) ---

def detect_sign_logic(image_path):
    if not sign_model: 
        return {"result": "❌ Lỗi: Model chưa load.", "image_path": None, "detections": []}

    try:
        img = cv2.imread(image_path)
        if img is None: return {"result": "❌ Lỗi đọc ảnh", "image_path": None}

        # Dự đoán
        results = sign_model.predict(source=img, conf=0.25, save=False, verbose=False)
        detections = []
        found = False

        # Lấy tên class trực tiếp từ model
        names = sign_model.names

        for r in results:
            for box in r.boxes:
                found = True
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                # Lấy tên class gốc (có dấu)
                try:
                    display_name = names[cls_id]
                except Exception:
                    display_name = str(cls_id)
                
                # 1. Label dùng để trả về JSON (Giữ nguyên Tiếng Việt có dấu)
                label_full = display_name 

                # 2. Label dùng để vẽ lên ảnh (Xóa dấu để không bị lỗi ???)
                label_no_accent = remove_accents(display_name)
                draw_text = f"{label_no_accent} {conf:.0%}"

                # Vẽ khung
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Vẽ nhãn nền
                (w, h), _ = cv2.getTextSize(draw_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(img, (x1, y1 - 25), (x1 + w, y1), (0, 255, 0), -1)
                
                # Vẽ chữ (Dùng biến draw_text không dấu)
                cv2.putText(img, draw_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                
                detections.append({"class": label_full, "confidence": conf})

        filename = "checked_sign_" + os.path.basename(image_path)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        cv2.imwrite(save_path, img)

        msg = f"✅ Tìm thấy {len(detections)} biển báo." if found else "⚠️ Không tìm thấy biển báo nào."
        if found:
            # Tạo danh sách tên biển báo duy nhất để hiển thị thông báo
            unique_signs = list(set([d['class'] for d in detections]))
            msg += "\n" + ", ".join(unique_signs)

        return {"result": msg, "detections": detections, "image_path": filename}
    except Exception as e:
        logging.error(f"Lỗi detect_sign_logic: {e}")
        return {"result": f"Lỗi xử lý: {e}", "image_path": None}


def detect_sleep_logic(filepath):
    if not sleep_model or not face_net:
        return {"error": "Model Ngủ gật hoặc Face Detector chưa sẵn sàng."}

    try:
        image = cv2.imread(str(filepath))
        if image is None:
            return {"error": "Không thể đọc file ảnh."}
        
        (h, w) = image.shape[:2]

        # Face detection (OpenCV DNN)
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
        face_net.setInput(blob)
        detections = face_net.forward()

        best_face = None
        best_confidence = 0.0

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:  # threshold
                if confidence > best_confidence:
                    best_confidence = confidence
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    (startX, startY) = (max(0, startX), max(0, startY))
                    (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
                    best_face = (startX, startY, endX, endY)

        if best_face is None:
            return {"result": "❌ Không tìm thấy khuôn mặt", "image_path": None}

        (startX, startY, endX, endY) = best_face
        face_width = endX - startX
        if face_width < 10:
            return {"result": f"❌ Lỗi phát hiện khuôn mặt (width: {face_width}px)", "image_path": None}

        face_roi = image[startY:endY, startX:endX]
        if face_roi.size == 0:
            return {"result": "❌ Kích thước khuôn mặt không hợp lệ", "image_path": None}

        # Run detection on cropped face
        results = sleep_model.predict(source=face_roi, verbose=False)

        result_text = ""
        is_sleepy = False

        if hasattr(results[0], 'boxes') and results[0].boxes is not None:
            if len(results[0].boxes) == 0:
                result_text = "✅ PHÁT HIỆN: TỈNH TÁO"
                color = (0, 255, 0)
                draw_label = "AWAKE (No detections)"
                cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
                cv2.putText(image, draw_label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            else:
                for box in results[0].boxes:
                    class_id = int(box.cls.item())
                    confidence = float(box.conf.item())
                    class_name = SLEEP_CLASS_NAMES.get(class_id, "unknown")

                    if class_id == 1:
                        is_sleepy = True

                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                    # Map tọa độ từ face_roi ra ảnh gốc
                    x1_abs = x1 + startX
                    y1_abs = y1 + startY
                    x2_abs = x2 + startX
                    y2_abs = y2 + startY

                    box_color = (0, 0, 255) if class_id == 1 else (0, 255, 0)
                    # class_name ở đây thường là tiếng Anh hoặc không dấu nên vẽ được
                    draw_label = f"{class_name} {confidence*100:.0f}%"
                    
                    cv2.rectangle(image, (x1_abs, y1_abs), (x2_abs, y2_abs), box_color, 1)
                    cv2.putText(image, draw_label, (x1_abs, y1_abs - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

                if is_sleepy:
                    result_text = "❌ PHÁT HIỆN: CÓ NGỦ GẬT"
                    color = (0, 0, 255)
                    draw_label = "SLEEPY"
                else:
                    result_text = "✅ PHÁT HIỆN: TỈNH TÁO"
                    color = (0, 255, 0)
                    draw_label = "AWAKE"

                cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
                cv2.putText(image, draw_label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        else:
            result_text = "❌ Lỗi: Model không trả về kết quả hợp lệ."

        # Save annotated image
        filename = os.path.basename(filepath)
        image_annotated_name = Path(filename).stem + "_sleep_annotated.jpg"
        image_annotated_path = Path(app.config['UPLOAD_FOLDER']) / image_annotated_name
        cv2.imwrite(str(image_annotated_path), image)

        try:
            Path(filepath).unlink()
        except Exception:
            logging.warning("Không xóa được file upload tạm thời (ngủ gật).")

        return {"result": result_text, "image_path": str(image_annotated_name)}
    except Exception as e:
        logging.exception("Lỗi trong detect_sleep_logic:")
        try:
            Path(filepath).unlink()
        except Exception:
            pass
        return {"error": f"Lỗi khi xử lý file: {e}"}

# --- 4. ROUTES API ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# --- Chat Handlers ---
SYSTEM_PROMPT = "Bạn là trợ lý lái xe AI thông minh. Trả lời ngắn gọn, hữu ích."

def call_gemini(parts, history):
    if not gemini_model: return "Chưa cấu hình Gemini"
    try:
        chat = gemini_model.start_chat(history=[])
        new_parts = [p['text'] for p in parts if 'text' in p]
        return chat.send_message(new_parts).text
    except Exception as e: return f"Lỗi Gemini: {e}"

def call_openai(parts, history):
    if not openai_client: return "Chưa cấu hình OpenAI"
    try:
        text = " ".join([p.get('text', '') for p in parts])
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}]
        )
        return resp.choices[0].message.content
    except Exception as e: return f"Lỗi OpenAI: {e}"

def call_deepseek(parts, history):
    if not deepseek_client: return "Chưa cấu hình DeepSeek"
    try:
        text = " ".join([p.get('text', '') for p in parts])
        resp = deepseek_client.chat.completions.create(
            model="deepseek/deepseek-chat-v3.1:free", messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}]
        )
        return resp.choices[0].message.content
    except Exception as e: return f"Lỗi DeepSeek: {e}"

@app.route('/chat', methods=['POST'])
def handle_chat():
    data = request.json
    return jsonify({"response": call_gemini(data.get('parts', []), data.get('history', []))})

@app.route('/chat-openai', methods=['POST'])
def handle_openai():
    data = request.json
    return jsonify({"response": call_openai(data.get('parts', []), data.get('history', []))})

@app.route('/chat-deepseek', methods=['POST'])
def handle_deepseek():
    data = request.json
    return jsonify({"response": call_deepseek(data.get('parts', []), data.get('history', []))})

@app.route('/chat-aggregate', methods=['POST'])
def handle_aggregate():
    data = request.json
    parts = data.get('parts', [])
    hist = data.get('history', [])
    with ThreadPoolExecutor(max_workers=3) as exe:
        f1 = exe.submit(call_gemini, parts, hist)
        f2 = exe.submit(call_openai, parts, hist)
        f3 = exe.submit(call_deepseek, parts, hist)
        results = {"gemini": f1.result(), "openai": f2.result(), "deepseek": f3.result()}
    
    summary_prompt = f"Tổng hợp 3 câu trả lời sau thành 1 câu tốt nhất:\n1. {results['gemini']}\n2. {results['openai']}\n3. {results['deepseek']}"
    final = call_gemini([{'text': summary_prompt}], [])
    return jsonify({"final_answer": final, "sources": results})

# --- Law Lookup ---
@app.route('/chat-law-lookup', methods=['POST'])
def handle_law_lookup():
    data = request.json
    query = data.get('message', '')
    if not query: return jsonify({"error": "Nội dung trống"}), 400
    try:
        context = search_vbpl_sync(query)
        prompt = f"Dựa vào luật sau:\n{context}\n\nTrả lời câu hỏi: '{query}' ngắn gọn, chính xác."
        response = call_gemini([{'text': prompt}], [])
        return jsonify({"response": response})
    except Exception as e: return jsonify({"error": str(e)}), 500

# --- Detection Endpoints ---
@app.route('/detect-sign', methods=['POST'])
def route_detect_sign():
    if 'file' not in request.files: return jsonify({"error": "No file"}), 400
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = f"{int(time.time())}_{secure_filename(file.filename)}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return jsonify(detect_sign_logic(filepath))
    return jsonify({"error": "Invalid file"}), 400

@app.route('/detect-sleep', methods=['POST'])
def route_detect_sleep():
    if 'file' not in request.files: return jsonify({"error": "No file"}), 400
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = f"{int(time.time())}_{secure_filename(file.filename)}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return jsonify(detect_sleep_logic(filepath))
    return jsonify({"error": "Invalid file"}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)