# 🚀 Docker Quick Start

## ⚡ Cách nhanh nhất

### Option 1: Script tự động (Khuyến nghị)
```bash
# Chạy script build và start
./run-docker.sh

# Hoặc với Docker Compose
./run-compose.sh
```

### Option 2: Docker Compose thủ công
```bash
docker-compose up -d
```

### Option 3: Docker command thủ công
```bash
docker build -t ai-chatbot .
docker run -d -p 5000:5000 --name ai-chatbot-app ai-chatbot
```

## 🌐 Truy cập
- **Local**: http://localhost:5000
- **Network**: http://[your-ip]:5000

## 🔧 Commands hữu ích
```bash
# Xem logs
docker logs -f ai-chatbot-app

# Stop container
docker stop ai-chatbot-app

# Restart
docker restart ai-chatbot-app

# Remove
docker rm ai-chatbot-app
```

## 📁 Files được tạo
- `Dockerfile` - Build instructions
- `docker-compose.yml` - Service orchestration  
- `.dockerignore` - Exclude files
- `docker-entrypoint.sh` - Container startup
- `run-docker.sh` - Quick build script
- `run-compose.sh` - Compose script

Đọc `README-Docker.md` để biết thêm chi tiết! 🐳 