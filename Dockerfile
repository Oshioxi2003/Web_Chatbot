# Sử dụng Python 3.11 slim image để giảm kích thước
FROM python:3.11-slim

# Đặt thông tin metadata
LABEL maintainer="AI Chatbot"
LABEL description="Modern AI Chatbot with Gemini API and Messenger-like UI"

# Đặt biến môi trường
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=app.py

# Tạo thư mục làm việc
WORKDIR /app

# Cập nhật package manager và cài đặt dependencies cần thiết
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements trước để tận dụng Docker cache
COPY requirements.txt .

# Cài đặt Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Tạo thư mục cho conversations nếu chưa có
RUN mkdir -p /app/data

# Đặt quyền cho thư mục và scripts
RUN chmod -R 755 /app && \
    chmod +x /app/docker-entrypoint.sh

# Tạo user non-root để tăng bảo mật
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/ || exit 1

# Sử dụng entrypoint script
ENTRYPOINT ["/app/docker-entrypoint.sh"] 