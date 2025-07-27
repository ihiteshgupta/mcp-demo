/**
 * Prompt Builder Client - Utility Functions
 * Shared JavaScript utilities for the Prompt Builder Client
 */

// API utilities
const API = {
    baseURL: window.location.origin,
    
    async request(endpoint, options = {}) {
        const url = `${this.baseURL}${endpoint}`;
        const defaultOptions = {
            headers: {
                'Content-Type': 'application/json',
            },
        };
        
        const finalOptions = { ...defaultOptions, ...options };
        
        if (finalOptions.body && typeof finalOptions.body === 'object') {
            finalOptions.body = JSON.stringify(finalOptions.body);
        }
        
        try {
            const response = await fetch(url, finalOptions);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const contentType = response.headers.get('content-type');
            if (contentType && contentType.includes('application/json')) {
                return await response.json();
            }
            
            return await response.text();
        } catch (error) {
            console.error('API request failed:', error);
            throw error;
        }
    },
    
    get(endpoint) {
        return this.request(endpoint, { method: 'GET' });
    },
    
    post(endpoint, data) {
        return this.request(endpoint, {
            method: 'POST',
            body: data,
        });
    },
    
    put(endpoint, data) {
        return this.request(endpoint, {
            method: 'PUT',
            body: data,
        });
    },
    
    delete(endpoint) {
        return this.request(endpoint, { method: 'DELETE' });
    }
};

// Notification utilities
const Notifications = {
    show(message, type = 'info', duration = 3000) {
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.innerHTML = `
            <div class="notification-content">
                <i class="fas fa-${this.getIcon(type)}"></i>
                <span>${message}</span>
                <button class="notification-close" onclick="this.parentElement.parentElement.remove()">
                    <i class="fas fa-times"></i>
                </button>
            </div>
        `;
        
        // Add styles if not already present
        if (!document.getElementById('notification-styles')) {
            const styles = document.createElement('style');
            styles.id = 'notification-styles';
            styles.textContent = `
                .notification {
                    position: fixed;
                    top: 20px;
                    right: 20px;
                    z-index: 9999;
                    padding: 16px 20px;
                    border-radius: 8px;
                    color: white;
                    font-weight: 500;
                    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
                    transform: translateX(100%);
                    transition: transform 0.3s ease;
                    max-width: 400px;
                }
                
                .notification.show {
                    transform: translateX(0);
                }
                
                .notification-success { background-color: #10b981; }
                .notification-error { background-color: #ef4444; }
                .notification-warning { background-color: #f59e0b; }
                .notification-info { background-color: #3b82f6; }
                
                .notification-content {
                    display: flex;
                    align-items: center;
                    gap: 8px;
                }
                
                .notification-close {
                    background: none;
                    border: none;
                    color: white;
                    cursor: pointer;
                    padding: 4px;
                    margin-left: auto;
                }
            `;
            document.head.appendChild(styles);
        }
        
        document.body.appendChild(notification);
        
        // Trigger animation
        setTimeout(() => notification.classList.add('show'), 100);
        
        // Auto remove
        if (duration > 0) {
            setTimeout(() => {
                notification.classList.remove('show');
                setTimeout(() => notification.remove(), 300);
            }, duration);
        }
        
        return notification;
    },
    
    getIcon(type) {
        const icons = {
            success: 'check-circle',
            error: 'exclamation-triangle',
            warning: 'exclamation-circle',
            info: 'info-circle'
        };
        return icons[type] || 'info-circle';
    },
    
    success(message, duration) {
        return this.show(message, 'success', duration);
    },
    
    error(message, duration) {
        return this.show(message, 'error', duration);
    },
    
    warning(message, duration) {
        return this.show(message, 'warning', duration);
    },
    
    info(message, duration) {
        return this.show(message, 'info', duration);
    }
};

// Storage utilities
const Storage = {
    get(key, defaultValue = null) {
        try {
            const item = localStorage.getItem(key);
            return item ? JSON.parse(item) : defaultValue;
        } catch (error) {
            console.error('Storage get error:', error);
            return defaultValue;
        }
    },
    
    set(key, value) {
        try {
            localStorage.setItem(key, JSON.stringify(value));
            return true;
        } catch (error) {
            console.error('Storage set error:', error);
            return false;
        }
    },
    
    remove(key) {
        try {
            localStorage.removeItem(key);
            return true;
        } catch (error) {
            console.error('Storage remove error:', error);
            return false;
        }
    },
    
    clear() {
        try {
            localStorage.clear();
            return true;
        } catch (error) {
            console.error('Storage clear error:', error);
            return false;
        }
    }
};

// Form utilities
const Forms = {
    serialize(form) {
        const formData = new FormData(form);
        const data = {};
        
        for (const [key, value] of formData.entries()) {
            if (data[key]) {
                if (Array.isArray(data[key])) {
                    data[key].push(value);
                } else {
                    data[key] = [data[key], value];
                }
            } else {
                data[key] = value;
            }
        }
        
        return data;
    },
    
    validate(form, rules = {}) {
        const errors = {};
        const data = this.serialize(form);
        
        for (const [field, rule] of Object.entries(rules)) {
            const value = data[field];
            
            if (rule.required && (!value || value.trim() === '')) {
                errors[field] = rule.message || `${field} is required`;
                continue;
            }
            
            if (value && rule.pattern && !rule.pattern.test(value)) {
                errors[field] = rule.message || `${field} format is invalid`;
                continue;
            }
            
            if (value && rule.minLength && value.length < rule.minLength) {
                errors[field] = rule.message || `${field} must be at least ${rule.minLength} characters`;
                continue;
            }
            
            if (value && rule.maxLength && value.length > rule.maxLength) {
                errors[field] = rule.message || `${field} must be no more than ${rule.maxLength} characters`;
                continue;
            }
        }
        
        return {
            isValid: Object.keys(errors).length === 0,
            errors,
            data
        };
    }
};

// Clipboard utilities
const Clipboard = {
    async copy(text) {
        try {
            await navigator.clipboard.writeText(text);
            Notifications.success('Copied to clipboard');
            return true;
        } catch (error) {
            console.error('Clipboard copy failed:', error);
            Notifications.error('Failed to copy to clipboard');
            return false;
        }
    },
    
    async read() {
        try {
            return await navigator.clipboard.readText();
        } catch (error) {
            console.error('Clipboard read failed:', error);
            return null;
        }
    }
};

// URL utilities
const URL_Utils = {
    getParams() {
        return Object.fromEntries(new URLSearchParams(window.location.search));
    },
    
    setParam(key, value) {
        const url = new URL(window.location);
        url.searchParams.set(key, value);
        window.history.pushState({}, '', url);
    },
    
    removeParam(key) {
        const url = new URL(window.location);
        url.searchParams.delete(key);
        window.history.pushState({}, '', url);
    }
};

// Date utilities
const DateUtils = {
    format(date, options = {}) {
        const defaultOptions = {
            year: 'numeric',
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        };
        
        return new Intl.DateTimeFormat('en-US', { ...defaultOptions, ...options })
            .format(new Date(date));
    },
    
    relative(date) {
        const now = new Date();
        const target = new Date(date);
        const diffMs = now - target;
        const diffMins = Math.floor(diffMs / 60000);
        const diffHours = Math.floor(diffMins / 60);
        const diffDays = Math.floor(diffHours / 24);
        
        if (diffMins < 1) return 'just now';
        if (diffMins < 60) return `${diffMins} minute${diffMins > 1 ? 's' : ''} ago`;
        if (diffHours < 24) return `${diffHours} hour${diffHours > 1 ? 's' : ''} ago`;
        if (diffDays < 7) return `${diffDays} day${diffDays > 1 ? 's' : ''} ago`;
        
        return this.format(date);
    }
};

// Export utilities for use in other scripts
window.PromptBuilderUtils = {
    API,
    Notifications,
    Storage,
    Forms,
    Clipboard,
    URL_Utils,
    DateUtils
};