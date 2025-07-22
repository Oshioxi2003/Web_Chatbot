#!/bin/bash

# Docker entrypoint script cho AI Chatbot
set -e

echo "🚀 Starting AI Chatbot with Gemini API..."

# Kiểm tra file conversations.json, tạo nếu chưa có
if [ ! -f "/app/conversations.json" ]; then
    echo "📝 Creating conversations.json file..."
    echo "{}" > /app/conversations.json
fi

# Đảm bảo quyền file
chmod 644 /app/conversations.json

# Kiểm tra kết nối internet (optional)
echo "🌐 Checking internet connection..."
if curl -s --max-time 5 https://google.com > /dev/null; then
    echo "✅ Internet connection is working"
else
    echo "⚠️  Warning: No internet connection detected"
fi

# Hiển thị thông tin hệ thống
echo "📊 System Information:"
echo "   - Python version: $(python --version)"
echo "   - Working directory: $(pwd)"
echo "   - User: $(whoami)"
echo "   - Available memory: $(free -h | grep Mem | awk '{print $7}')"

# Chạy ứng dụng
echo "🎯 Starting Flask application on 0.0.0.0:5000..."
exec python app.py 