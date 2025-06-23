# app.py

import os
import google.generativeai as genai
from flask import Flask, request, jsonify, render_template 
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)
CORS(app)
import gunicorn

# --- Cấu hình Gemini API ---
API_KEY = os.getenv("GOOGLE_API_KEY")
try:
    genai.configure(api_key=API_KEY)
    # Di chuyển việc khởi tạo model ra ngoài để dùng chung
    model = genai.GenerativeModel('gemini-1.5-flash') 
except Exception as e:
    print(f"Lỗi khi cấu hình Gemini: {e}")
    model = None 

# --- Định nghĩa các Route ---

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
        # Nhận lịch sử chat từ phía client
        history = user_data.get('history', [])

        if not user_input:
            return jsonify({"error": "Không nhận được tin nhắn."}), 400

        # Bắt đầu một cuộc hội thoại MỚI với lịch sử được cung cấp
        chat_session = model.start_chat(history=history)
        response = chat_session.send_message(user_input)
        
        return jsonify({"response": response.text})
        
    except Exception as e:
        print(f"Đã xảy ra lỗi trong handle_chat: {e}")
        return jsonify({"error": "Đã có lỗi xảy ra phía máy chủ."}), 500

# Route /clear không còn cần thiết nữa vì server không lưu gì cả
# Bạn có thể xóa nó đi nếu muốn

# --- Chạy ứng dụng ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, port=port)