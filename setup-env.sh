#!/bin/bash

echo "🔐 AI Chatbot Environment Setup"
echo "==============================="

# Check if .env already exists
if [ -f ".env" ]; then
    echo "⚠️  .env file already exists!"
    read -p "Do you want to overwrite it? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Setup cancelled."
        exit 0
    fi
fi

# Create .env file
echo "📝 Creating .env file..."

# Get API key from user
echo ""
echo "🔑 Gemini API Key Setup"
echo "Get your API key from: https://makersuite.google.com/app/apikey"
echo ""
read -p "Enter your Gemini API key: " api_key

if [ -z "$api_key" ]; then
    echo "❌ API key cannot be empty!"
    exit 1
fi

# Generate secret key
secret_key=$(python3 -c "import secrets; print(secrets.token_hex(32))" 2>/dev/null || openssl rand -hex 32)

# Create .env file
cat > .env << EOF
# AI Chatbot Environment Variables
# =================================

# Gemini API Configuration
GEMINI_API_KEY=$api_key

# Flask Configuration
FLASK_ENV=production
FLASK_DEBUG=False

# Security
SECRET_KEY=$secret_key

# Optional: Custom settings
# APP_NAME=AI Chatbot
# LOG_LEVEL=INFO
EOF

echo "✅ .env file created successfully!"
echo ""
echo "🔒 Security reminders:"
echo "   - Never commit .env file to version control"
echo "   - Keep your API key private"
echo "   - Change SECRET_KEY in production"
echo ""
echo "🚀 You can now run the application with:"
echo "   python app.py"
echo "   or"
echo "   docker-compose up -d" 