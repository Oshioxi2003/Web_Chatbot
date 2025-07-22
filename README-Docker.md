# 🐳 AI Chatbot Docker Deployment

Modern AI Chatbot với Gemini API và giao diện giống Messenger, sẵn sàng triển khai với Docker.

## 📋 Yêu cầu hệ thống

- Docker Engine 20.10+
- Docker Compose 2.0+
- 2GB RAM trống
- Kết nối Internet (để gọi Gemini API)

## 🚀 Cách chạy

### Option 1: Sử dụng Docker Compose (Khuyến nghị)

```bash
# Build và chạy ứng dụng
docker-compose up -d

# Xem logs
docker-compose logs -f

# Dừng ứng dụng
docker-compose down
```

### Option 2: Sử dụng Docker trực tiếp

```bash
# Build image
docker build -t ai-chatbot .

# Chạy container
docker run -d \
  --name ai-chatbot-app \
  -p 5000:5000 \
  -v $(pwd)/conversations.json:/app/conversations.json \
  ai-chatbot

# Xem logs
docker logs -f ai-chatbot-app

# Dừng container
docker stop ai-chatbot-app
docker rm ai-chatbot-app
```

## 🌐 Truy cập ứng dụng

Sau khi chạy thành công, truy cập:
- **Localhost**: http://localhost:5000
- **LAN**: http://[IP-máy-host]:5000

## 📁 Cấu trúc Files

```
├── Dockerfile              # Định nghĩa Docker image
├── docker-compose.yml      # Orchestration với Docker Compose
├── .dockerignore           # Loại trừ files không cần thiết
├── docker-entrypoint.sh    # Script khởi tạo container
├── app.py                  # Ứng dụng Flask chính
├── requirements.txt        # Python dependencies
├── static/                 # CSS, JS, images
├── templates/              # HTML templates
└── conversations.json      # Lưu trữ cuộc trò chuyện
```

## ⚙️ Cấu hình nâng cao

### Thay đổi port

Chỉnh sửa trong `docker-compose.yml`:
```yaml
ports:
  - "8080:5000"  # Thay 8080 thành port mong muốn
```

### Mount custom conversations file

```bash
docker run -d \
  -p 5000:5000 \
  -v /path/to/your/conversations.json:/app/conversations.json \
  ai-chatbot
```

### Environment variables

Có thể thêm biến môi trường trong `docker-compose.yml`:
```yaml
environment:
  - GEMINI_API_KEY=your_api_key_here
  - FLASK_ENV=production
```

## 🔧 Troubleshooting

### Container không start được
```bash
# Kiểm tra logs
docker-compose logs ai-chatbot

# Kiểm tra status
docker-compose ps
```

### Không truy cập được từ bên ngoài
```bash
# Kiểm tra port binding
docker port ai-chatbot-app

# Kiểm tra firewall (Linux)
sudo ufw allow 5000/tcp
```

### Lỗi permission denied
```bash
# Đảm bảo quyền file
chmod 644 conversations.json
chmod +x docker-entrypoint.sh
```

## 📊 Monitoring

### Health check
```bash
# Kiểm tra health
docker inspect --format='{{.State.Health.Status}}' ai-chatbot-app

# Xem health check logs
docker inspect --format='{{range .State.Health.Log}}{{.Output}}{{end}}' ai-chatbot-app
```

### Resource usage
```bash
# Xem CPU/Memory usage
docker stats ai-chatbot-app
```

## 🔄 Updates

### Cập nhật ứng dụng
```bash
# Pull code mới
git pull

# Rebuild và restart
docker-compose down
docker-compose up -d --build
```

### Backup conversations
```bash
# Backup dữ liệu trò chuyện
cp conversations.json conversations.backup.$(date +%Y%m%d).json
```

## 🛡️ Security Notes

- Container chạy với non-root user
- Health checks được enable
- Logs rotation được cấu hình
- Network isolation với bridge network
- API key được truyền qua environment variables

## 💡 Tips

1. **Performance**: Sử dụng `docker-compose` cho production
2. **Development**: Mount source code để live reload
3. **Scaling**: Có thể chạy multiple instances với load balancer
4. **Monitoring**: Integrate với Prometheus/Grafana nếu cần

## 📞 Support

Nếu gặp vấn đề, kiểm tra:
1. Docker daemon có đang chạy không
2. Port 5000 có bị conflict không
3. Đủ dung lượng disk cho image
4. Kết nối internet cho Gemini API

Happy chatting! 🤖✨ 