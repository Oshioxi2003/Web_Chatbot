class ModernChatbot {
    constructor() {
        this.messagesContainer = document.getElementById('messagesContainer');
        this.messageInput = document.getElementById('messageInput');
        this.sendBtn = document.getElementById('sendBtn');
        this.clearBtn = document.getElementById('clearBtn');
        this.typingIndicator = document.getElementById('typingIndicator');
        this.loadingOverlay = document.getElementById('loadingOverlay');
        
        this.messages = [];
        this.isTyping = false;
        
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.loadConversationHistory();
        this.autoResizeTextarea();
    }

    setupEventListeners() {
        // Send button click
        this.sendBtn.addEventListener('click', () => this.sendMessage());
        
        // Enter key to send message
        this.messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        // Auto-resize textarea
        this.messageInput.addEventListener('input', () => {
            this.autoResizeTextarea();
        });

        // Clear conversation
        this.clearBtn.addEventListener('click', () => this.clearConversation());

        // Prevent form submission
        this.messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && e.shiftKey) {
                // Allow shift+enter for new line
                return;
            }
        });
    }

    autoResizeTextarea() {
        const textarea = this.messageInput;
        textarea.style.height = 'auto';
        const newHeight = Math.min(textarea.scrollHeight, 120);
        textarea.style.height = newHeight + 'px';
        
        // Update send button state
        this.updateSendButton();
    }

    updateSendButton() {
        const hasText = this.messageInput.value.trim().length > 0;
        this.sendBtn.disabled = !hasText || this.isTyping;
        this.sendBtn.style.opacity = hasText && !this.isTyping ? '1' : '0.5';
    }

    async sendMessage() {
        const text = this.messageInput.value.trim();
        if (!text || this.isTyping) return;

        // Add user message
        this.addMessage(text, 'user');
        this.messageInput.value = '';
        this.autoResizeTextarea();
        
        // Show typing indicator
        this.showTypingIndicator();
        
        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ message: text })
            });

            const data = await response.json();
            
            // Hide typing indicator
            this.hideTypingIndicator();

            if (response.ok) {
                // Add AI response
                this.addMessage(data.answer, 'assistant');
            } else {
                // Show error message
                this.addMessage(data.error || 'Đã xảy ra lỗi. Vui lòng thử lại.', 'assistant', true);
            }
        } catch (error) {
            console.error('Error:', error);
            this.hideTypingIndicator();
            this.addMessage('Không thể kết nối đến server. Vui lòng kiểm tra kết nối mạng.', 'assistant', true);
        }
    }

    addMessage(content, role, isError = false) {
        // Remove welcome message if it exists
        const welcomeMessage = this.messagesContainer.querySelector('.welcome-message');
        if (welcomeMessage) {
            welcomeMessage.remove();
        }

        const messageItem = document.createElement('div');
        messageItem.className = `message-item ${role}`;
        
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        
        const messageBubble = document.createElement('div');
        messageBubble.className = 'message-bubble';
        if (isError) {
            messageBubble.style.background = '#ff4757';
            messageBubble.style.color = 'white';
        }
        messageBubble.textContent = content;
        
        const messageAvatar = document.createElement('div');
        messageAvatar.className = 'message-avatar';
        
        if (role === 'user') {
            messageAvatar.innerHTML = '<i class="fas fa-user"></i>';
        } else {
            messageAvatar.innerHTML = '<i class="fas fa-robot"></i>';
        }
        
        const messageTime = document.createElement('div');
        messageTime.className = 'message-time';
        messageTime.textContent = this.formatTime(new Date());
        
        messageContent.appendChild(messageBubble);
        messageContent.appendChild(messageTime);
        
        messageItem.appendChild(messageAvatar);
        messageItem.appendChild(messageContent);
        
        this.messagesContainer.appendChild(messageItem);
        this.scrollToBottom();
        
        // Add to messages array
        this.messages.push({
            content,
            role,
            timestamp: new Date().toISOString()
        });
    }

    showTypingIndicator() {
        this.isTyping = true;
        this.typingIndicator.style.display = 'flex';
        this.updateSendButton();
        this.scrollToBottom();
    }

    hideTypingIndicator() {
        this.isTyping = false;
        this.typingIndicator.style.display = 'none';
        this.updateSendButton();
    }

    scrollToBottom() {
        setTimeout(() => {
            this.messagesContainer.scrollTo({
                top: this.messagesContainer.scrollHeight,
                behavior: 'smooth'
            });
        }, 100);
    }

    formatTime(date) {
        return date.toLocaleTimeString('vi-VN', {
            hour: '2-digit',
            minute: '2-digit'
        });
    }

    async loadConversationHistory() {
        try {
            const response = await fetch('/history');
            if (response.ok) {
                const history = await response.json();
                if (history.length > 0) {
                    // Remove welcome message
                    const welcomeMessage = this.messagesContainer.querySelector('.welcome-message');
                    if (welcomeMessage) {
                        welcomeMessage.remove();
                    }
                    
                    // Load messages
                    history.forEach(msg => {
                        this.addMessage(msg.content, msg.role);
                    });
                }
            }
        } catch (error) {
            console.error('Error loading history:', error);
        }
    }

    async clearConversation() {
        if (!confirm('Bạn có chắc chắn muốn xóa toàn bộ cuộc trò chuyện?')) {
            return;
        }
        
        this.showLoading();
        
        try {
            const response = await fetch('/clear', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            if (response.ok) {
                // Clear UI
                this.messagesContainer.innerHTML = `
                    <div class="welcome-message">
                        <div class="welcome-avatar">
                            <i class="fas fa-robot"></i>
                        </div>
                        <h3>Chào mừng đến với AI Assistant!</h3>
                        <p>Tôi có thể giúp bạn trả lời câu hỏi, hỗ trợ công việc và trò chuyện. Hãy bắt đầu cuộc trò chuyện!</p>
                    </div>
                `;
                this.messages = [];
            } else {
                alert('Không thể xóa cuộc trò chuyện. Vui lòng thử lại.');
            }
        } catch (error) {
            console.error('Error clearing conversation:', error);
            alert('Đã xảy ra lỗi. Vui lòng thử lại.');
        }
        
        this.hideLoading();
    }

    showLoading() {
        this.loadingOverlay.style.display = 'flex';
    }

    hideLoading() {
        this.loadingOverlay.style.display = 'none';
    }
}

// Initialize chatbot when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new ModernChatbot();
});

// Add some utility functions for better UX
document.addEventListener('DOMContentLoaded', () => {
    // Add ripple effect to buttons
    const buttons = document.querySelectorAll('.send-btn, .action-btn, .attach-btn');
    buttons.forEach(button => {
        button.addEventListener('click', function(e) {
            const ripple = document.createElement('span');
            const rect = this.getBoundingClientRect();
            const size = Math.max(rect.width, rect.height);
            const x = e.clientX - rect.left - size / 2;
            const y = e.clientY - rect.top - size / 2;
            
            ripple.style.width = ripple.style.height = size + 'px';
            ripple.style.left = x + 'px';
            ripple.style.top = y + 'px';
            ripple.classList.add('ripple');
            
            this.appendChild(ripple);
            
            setTimeout(() => {
                ripple.remove();
            }, 600);
        });
    });
});

// Add CSS for ripple effect
const style = document.createElement('style');
style.textContent = `
    .ripple {
        position: absolute;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.6);
        transform: scale(0);
        animation: ripple-animation 0.6s linear;
        pointer-events: none;
    }
    
    @keyframes ripple-animation {
        to {
            transform: scale(4);
            opacity: 0;
        }
    }
    
    .send-btn, .action-btn, .attach-btn {
        position: relative;
        overflow: hidden;
    }
`;
document.head.appendChild(style);