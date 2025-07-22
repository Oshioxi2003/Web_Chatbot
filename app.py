from flask import Flask, request, jsonify, render_template, session
import google.generativeai as genai
import json
import os
from datetime import datetime
import secrets
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Cấu hình Flask từ environment variables
app.secret_key = os.getenv('SECRET_KEY', secrets.token_hex(16))

# Cấu hình Gemini API từ environment variables
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY not found! Please create .env file and add your API key.")

genai.configure(api_key=GEMINI_API_KEY)
print(f"✅ Gemini API configured successfully (Key: {GEMINI_API_KEY[:8]}...)")

# Tạo model Gemini
model = genai.GenerativeModel('gemini-2.5-flash')

# File lưu trữ cuộc trò chuyện
CONVERSATIONS_FILE = 'conversations.json'

def clean_markdown_formatting(text):
    """Loại bỏ các ký tự markdown formatting để text trở nên tự nhiên"""
    if not text:
        return text
    
    # Loại bỏ bold formatting (**text** hoặc __text__)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'__(.*?)__', r'\1', text)
    
    # Loại bỏ italic formatting (*text* hoặc _text_)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'_(.*?)_', r'\1', text)
    
    # Loại bỏ heading formatting (### text)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    
    # Loại bỏ code formatting (`code`)
    text = re.sub(r'`(.*?)`', r'\1', text)
    
    # Loại bỏ quote formatting (> text)
    text = re.sub(r'^>\s+', '', text, flags=re.MULTILINE)
    
    # Loại bỏ horizontal rules (---, ___, ***)
    text = re.sub(r'^[\-_\*]{3,}$', '', text, flags=re.MULTILINE)
    
    # Loại bỏ link formatting [text](url) -> text
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    
    # Loại bỏ list bullets (- item, * item, + item)
    text = re.sub(r'^[\-\*\+]\s+', '', text, flags=re.MULTILINE)
    
    # Loại bỏ numbered lists (1. item)
    text = re.sub(r'^\d+\.\s+', '', text, flags=re.MULTILINE)
    
    # Loại bỏ các dòng trống thừa
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    # Loại bỏ whitespace thừa ở đầu và cuối
    text = text.strip()
    
    return text

def load_conversations():
    """Tải cuộc trò chuyện từ file"""
    if os.path.exists(CONVERSATIONS_FILE):
        with open(CONVERSATIONS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_conversations(conversations):
    """Lưu cuộc trò chuyện vào file"""
    with open(CONVERSATIONS_FILE, 'w', encoding='utf-8') as f:
        json.dump(conversations, f, ensure_ascii=False, indent=2)

def get_session_id():
    """Lấy hoặc tạo session ID"""
    if 'session_id' not in session:
        session['session_id'] = secrets.token_hex(8)
    return session['session_id']

@app.get("/")
def index_get():
    return render_template("base.html")

@app.post("/predict")
def predict():
    try:
        text = request.get_json().get("message")
        if not text or text.strip() == "":
            return jsonify({"error": "Tin nhắn không thể để trống"}), 400

        session_id = get_session_id()
        
        # Tải cuộc trò chuyện hiện tại
        conversations = load_conversations()
        
        if session_id not in conversations:
            conversations[session_id] = {
                "created_at": datetime.now().isoformat(),
                "messages": []
            }
        
        # Thêm tin nhắn của người dùng
        user_message = {
            "role": "user",
            "content": text,
            "timestamp": datetime.now().isoformat()
        }
        conversations[session_id]["messages"].append(user_message)
        
        # Tạo context từ lịch sử cuộc trò chuyện
        context = ""
        recent_messages = conversations[session_id]["messages"][-10:]  # Lấy 10 tin nhắn gần nhất
        for msg in recent_messages:
            if msg["role"] == "user":
                context += f"Người dùng: {msg['content']}\n"
            else:
                context += f"Assistant: {msg['content']}\n"
        
        # Gửi tin nhắn đến Gemini
        prompt = f"""Bạn là một AI assistant thông minh và hữu ích. Hãy trả lời bằng tiếng Việt một cách tự nhiên và thân thiện.

QUAN TRỌNG: Hãy trả lời bằng văn bản thuần túy, KHÔNG sử dụng markdown formatting như **, ***, ###, ---, hay bất kỳ ký tự đặc biệt nào khác. Chỉ sử dụng text bình thường như khi nói chuyện.

Lịch sử cuộc trò chuyện:
{context}

Người dùng: {text}

Hãy trả lời câu hỏi của người dùng:"""

        response = model.generate_content(prompt)
        ai_response = response.text
        
        # Làm sạch markdown formatting
        ai_response = clean_markdown_formatting(ai_response)
        
        # Thêm phản hồi của AI
        ai_message = {
            "role": "assistant", 
            "content": ai_response,
            "timestamp": datetime.now().isoformat()
        }
        conversations[session_id]["messages"].append(ai_message)
        
        # Lưu cuộc trò chuyện
        save_conversations(conversations)
        
        return jsonify({"answer": ai_response})
        
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "Đã xảy ra lỗi khi xử lý tin nhắn"}), 500

@app.get("/history")
def get_history():
    """Lấy lịch sử cuộc trò chuyện"""
    session_id = get_session_id()
    conversations = load_conversations()
    
    if session_id in conversations:
        return jsonify(conversations[session_id]["messages"])
    return jsonify([])

@app.post("/clear")
def clear_history():
    """Xóa lịch sử cuộc trò chuyện"""
    session_id = get_session_id()
    conversations = load_conversations()
    
    if session_id in conversations:
        conversations[session_id]["messages"] = []
        save_conversations(conversations)
    
    return jsonify({"success": True})

if __name__ == "__main__":
    # Chạy trên tất cả interfaces để có thể truy cập từ bên ngoài Docker container
    app.run(host='0.0.0.0', port=5000, debug=False)
