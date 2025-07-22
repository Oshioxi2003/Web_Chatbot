#!/bin/bash

echo "🐳 AI Chatbot Docker Compose Setup"
echo "================================="

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "❌ .env file not found!"
    echo "Please create .env file first. You can:"
    echo "1. Copy .env.example to .env and edit it"
    echo "2. Run ./setup-env.sh to create it interactively"
    exit 1
fi

# Set permissions
chmod +x docker-entrypoint.sh
echo "✅ Set permissions for entrypoint script"

# Check if Docker Compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose not found. Please install Docker Compose first."
    exit 1
fi

# Stop existing services
echo "🛑 Stopping existing services..."
docker-compose down 2>/dev/null || true

# Build and start services
echo "🔨 Building and starting services..."
docker-compose up -d --build

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 SUCCESS! Chatbot is running with Docker Compose!"
    echo "📱 Access at: http://localhost:5000"
    echo ""
    echo "🔍 Useful commands:"
    echo "   View logs:     docker-compose logs -f"
    echo "   Stop services: docker-compose down"
    echo "   Restart:       docker-compose restart"
    echo "   Status:        docker-compose ps"
    echo ""
    
    # Show container status
    echo "📊 Container status:"
    docker-compose ps
    
    echo ""
    echo "📋 Recent logs:"
    docker-compose logs --tail 20
    
else
    echo "❌ Failed to start services with Docker Compose"
    exit 1
fi 