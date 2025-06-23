# app.py

import os
import google.generativeai as genai
# Thêm `render_template` để phục vụ file HTML
from flask import Flask, request, jsonify, render_template 
from flask_cors import CORS
from dotenv import load_dotenv

# --- Cấu hình ---
load_dotenv()
app = Flask(__name__)
CORS(app)

# --- Cấu hình cho Production ---
# Gunicorn là một server mạnh mẽ hơn server mặc định của Flask
# Chúng ta cần Gunicorn để chạy trên Render
import gunicorn

# --- Cấu hình Gemini API ---
API_KEY = os.getenv("GOOGLE_API_KEY")
try:
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
    chat = model.start_chat(history=[])
except Exception as e:
    print(f"Lỗi khi cấu hình Gemini: {e}")
    # Xử lý trường hợp không có API Key để app không bị crash
    model = None 
    chat = None

# --- Định nghĩa các Route ---

# Route cho trang chủ: Phục vụ file index.html
@app.route('/')
def index():
    # Flask sẽ tự động tìm file 'index.html' trong thư mục 'templates'
    return render_template('index.html')

# Route để xử lý chat
@app.route('/chat', methods=['POST'])
def handle_chat():
    if not chat:
        return jsonify({"error": "Mô hình AI chưa được khởi tạo, vui lòng kiểm tra API Key."}), 500
    
    try:
        user_data = request.get_json()
        user_input = user_data.get('message')

        if not user_input:
            return jsonify({"error": "Không nhận được tin nhắn."}), 400

        response = chat.send_message(user_input)
        return jsonify({"response": response.text})
    except Exception as e:
        print(f"Đã xảy ra lỗi: {e}")
        return jsonify({"error": "Đã có lỗi xảy ra phía máy chủ."}), 500

# Route để xóa lịch sử
@app.route('/clear', methods=['POST'])
def clear_chat():
    global chat
    if not model:
        return jsonify({"error": "Mô hình AI chưa được khởi tạo."}), 500
    
    chat = model.start_chat(history=[])
    return jsonify({"message": "Cuộc trò chuyện đã được xóa."})

# --- Chạy ứng dụng (Chỉ dùng cho local, Render sẽ không chạy dòng này) ---
if __name__ == '__main__':
    # Render sẽ dùng biến môi trường PORT để chạy app
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, port=port)