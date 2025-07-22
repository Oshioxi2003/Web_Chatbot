#!/bin/bash

echo "🐳 AI Chatbot Docker Setup"
echo "=========================="

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "❌ .env file not found!"
    echo "Please create .env file first. You can:"
    echo "1. Copy .env.example to .env and edit it"
    echo "2. Run ./setup-env.sh to create it interactively"
    exit 1
fi

# Set executable permissions
chmod +x docker-entrypoint.sh
echo "✅ Set permissions for entrypoint script"

# Build Docker image
echo "🔨 Building Docker image..."
docker build -t ai-chatbot .

if [ $? -eq 0 ]; then
    echo "✅ Docker image built successfully!"
    
    # Stop existing container if running
    echo "🛑 Stopping existing container (if any)..."
    docker stop ai-chatbot-app 2>/dev/null || true
    docker rm ai-chatbot-app 2>/dev/null || true
    
    # Run new container
    echo "🚀 Starting new container..."
    docker run -d \
        --name ai-chatbot-app \
        -p 5000:5000 \
        -v $(pwd)/conversations.json:/app/conversations.json \
        --restart unless-stopped \
        ai-chatbot
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "🎉 SUCCESS! Chatbot is running!"
        echo "📱 Access at: http://localhost:5000"
        echo ""
        echo "🔍 Useful commands:"
        echo "   View logs: docker logs -f ai-chatbot-app"
        echo "   Stop app:  docker stop ai-chatbot-app"
        echo "   Remove:    docker rm ai-chatbot-app"
        echo ""
        
        # Show logs for a few seconds
        echo "📋 Container logs (last 10 lines):"
        sleep 2
        docker logs --tail 10 ai-chatbot-app
        
    else
        echo "❌ Failed to start container"
        exit 1
    fi
else
    echo "❌ Failed to build Docker image"
    exit 1
fi 