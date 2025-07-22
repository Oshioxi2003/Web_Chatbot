# 🔐 Security Guide

## 🔑 API Key Security

### ✅ DO:
- Store API keys in `.env` file
- Use environment variables in production
- Keep `.env` file in `.gitignore`
- Generate strong SECRET_KEY for Flask
- Use different API keys for dev/prod

### ❌ DON'T:
- Hardcode API keys in source code
- Commit `.env` file to version control
- Share API keys in public
- Use default secret keys
- Log API keys in console

## 📁 File Security

### Protected Files:
- `.env` - Contains API keys and secrets
- `conversations.json` - User data
- Private keys and certificates

### Public Files:
- `.env.example` - Template without real values
- Source code (without secrets)
- Documentation

## 🐳 Docker Security

### Environment Variables:
```bash
# Use env_file in docker-compose.yml
env_file:
  - .env

# Or pass individual variables
environment:
  - GEMINI_API_KEY=${GEMINI_API_KEY}
```

### Best Practices:
- Run containers as non-root user
- Use multi-stage builds
- Scan images for vulnerabilities
- Use secrets management in production

## 🚀 Production Deployment

### Environment Setup:
1. Create secure `.env` file:
   ```bash
   ./setup-env.sh
   ```

2. Set strong SECRET_KEY:
   ```bash
   python -c "import secrets; print(secrets.token_hex(32))"
   ```

3. Use environment-specific configs:
   ```bash
   # .env.production
   FLASK_ENV=production
   FLASK_DEBUG=False
   ```

### Monitoring:
- Monitor for exposed secrets
- Regular security audits
- Log access patterns
- Use rate limiting

## 🔧 Tools & Scripts

### Setup Script:
```bash
./setup-env.sh  # Interactive environment setup
```

### Validation:
```bash
# Check if .env exists
if [ ! -f ".env" ]; then
    echo "❌ .env file missing!"
fi
```

## 📞 Incident Response

If API key is compromised:
1. Immediately revoke old key
2. Generate new API key
3. Update `.env` file
4. Restart all services
5. Audit access logs

## 🔗 Resources

- [Google AI Studio API Keys](https://makersuite.google.com/app/apikey)
- [Flask Security Best Practices](https://flask.palletsprojects.com/en/security/)
- [Docker Security](https://docs.docker.com/engine/security/)

Remember: Security is not optional! 🛡️ 