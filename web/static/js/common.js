/**
 * MyToolbox 通用 JavaScript 工具函数
 * 
 * 提供 Toast 通知、API 调用封装等通用功能
 */

// ==================== API 调用封装 ====================

/**
 * 发送 API 请求
 * @param {string} url - API 地址
 * @param {Object} options - 请求选项
 * @returns {Promise<Object>} 响应数据
 */
async function apiRequest(url, options = {}) {
    const defaultOptions = {
        headers: {
            'Content-Type': 'application/json',
        },
    };
    
    const mergedOptions = { ...defaultOptions, ...options };
    
    // 如果是 FormData，不设置 Content-Type（让浏览器自动设置）
    if (options.body instanceof FormData) {
        delete mergedOptions.headers['Content-Type'];
    }
    
    try {
        const response = await fetch(url, mergedOptions);
        const data = await response.json();
        
        if (!response.ok) {
            throw new ApiError(
                data.error?.message || '请求失败',
                data.error?.code || 'UNKNOWN_ERROR',
                response.status
            );
        }
        
        return data;
    } catch (error) {
        if (error instanceof ApiError) {
            throw error;
        }
        throw new ApiError(error.message || '网络请求失败', 'NETWORK_ERROR', 0);
    }
}

/**
 * API 错误类
 */
class ApiError extends Error {
    constructor(message, code, status) {
        super(message);
        this.name = 'ApiError';
        this.code = code;
        this.status = status;
    }
}

/**
 * GET 请求
 * @param {string} url - API 地址
 * @returns {Promise<Object>} 响应数据
 */
async function apiGet(url) {
    return apiRequest(url, { method: 'GET' });
}

/**
 * POST 请求（JSON）
 * @param {string} url - API 地址
 * @param {Object} data - 请求数据
 * @returns {Promise<Object>} 响应数据
 */
async function apiPost(url, data) {
    return apiRequest(url, {
        method: 'POST',
        body: JSON.stringify(data),
    });
}

/**
 * POST 请求（FormData）
 * @param {string} url - API 地址
 * @param {FormData} formData - 表单数据
 * @returns {Promise<Object>} 响应数据
 */
async function apiPostForm(url, formData) {
    return apiRequest(url, {
        method: 'POST',
        body: formData,
    });
}

/**
 * DELETE 请求
 * @param {string} url - API 地址
 * @returns {Promise<Object>} 响应数据
 */
async function apiDelete(url) {
    return apiRequest(url, { method: 'DELETE' });
}


// ==================== Toast 通知 ====================

/**
 * 显示 Toast 通知
 * @param {string} message - 消息内容
 * @param {string} type - 类型 ('success', 'error', 'warning', 'info')
 * @param {number} duration - 显示时长（毫秒）
 */
function showToast(message, type = 'info', duration = 3000) {
    const container = document.getElementById('toastContainer');
    if (!container) {
        console.error('Toast container not found');
        return;
    }
    
    // 图标映射
    const icons = {
        success: 'bi-check-circle-fill',
        error: 'bi-x-circle-fill',
        warning: 'bi-exclamation-triangle-fill',
        info: 'bi-info-circle-fill'
    };
    
    // 背景色映射
    const bgColors = {
        success: 'bg-success',
        error: 'bg-danger',
        warning: 'bg-warning',
        info: 'bg-primary'
    };
    
    // 创建 Toast 元素
    const toastId = 'toast-' + Date.now();
    const toastHtml = `
        <div id="${toastId}" class="toast align-items-center text-white ${bgColors[type]} border-0" role="alert" aria-live="assertive" aria-atomic="true">
            <div class="d-flex">
                <div class="toast-body d-flex align-items-center">
                    <i class="bi ${icons[type]} me-2"></i>
                    <span>${escapeHtml(message)}</span>
                </div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
            </div>
        </div>
    `;
    
    container.insertAdjacentHTML('beforeend', toastHtml);
    
    const toastElement = document.getElementById(toastId);
    const toast = new bootstrap.Toast(toastElement, {
        animation: true,
        autohide: true,
        delay: duration
    });
    
    // 显示 Toast
    toast.show();
    
    // 监听隐藏事件，移除 DOM 元素
    toastElement.addEventListener('hidden.bs.toast', () => {
        toastElement.remove();
    });
}

/**
 * 显示成功通知
 * @param {string} message - 消息内容
 */
function showSuccess(message) {
    showToast(message, 'success');
}

/**
 * 显示错误通知
 * @param {string} message - 消息内容
 */
function showError(message) {
    showToast(message, 'error', 5000);
}

/**
 * 显示警告通知
 * @param {string} message - 消息内容
 */
function showWarning(message) {
    showToast(message, 'warning', 4000);
}

/**
 * 显示信息通知
 * @param {string} message - 消息内容
 */
function showInfo(message) {
    showToast(message, 'info');
}


// ==================== 文件操作工具 ====================

/**
 * 格式化文件大小
 * @param {number} bytes - 字节数
 * @returns {string} 格式化后的大小
 */
function formatFileSize(bytes) {
    if (bytes === 0) return '0 B';
    
    const units = ['B', 'KB', 'MB', 'GB', 'TB'];
    const k = 1024;
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + units[i];
}

/**
 * 获取文件扩展名
 * @param {string} filename - 文件名
 * @returns {string} 扩展名（小写）
 */
function getFileExtension(filename) {
    return filename.split('.').pop().toLowerCase();
}

/**
 * 获取文件类型图标
 * @param {string} filename - 文件名
 * @returns {string} Bootstrap Icons 图标类名
 */
function getFileIcon(filename) {
    const ext = getFileExtension(filename);
    
    const audioExtensions = ['mp3', 'wav', 'flac', 'aac', 'ogg', 'm4a', 'wma'];
    const videoExtensions = ['mp4', 'mkv', 'avi', 'mov', 'wmv', 'flv', 'webm', 'm4v'];
    const subtitleExtensions = ['srt', 'vtt', 'ass', 'txt'];
    
    if (audioExtensions.includes(ext)) {
        return 'bi-file-earmark-music';
    } else if (videoExtensions.includes(ext)) {
        return 'bi-file-earmark-play';
    } else if (subtitleExtensions.includes(ext)) {
        return 'bi-file-earmark-text';
    } else {
        return 'bi-file-earmark';
    }
}

/**
 * 获取文件类型
 * @param {string} filename - 文件名
 * @returns {string} 文件类型 ('audio', 'video', 'subtitle', 'unknown')
 */
function getFileType(filename) {
    const ext = getFileExtension(filename);
    
    const audioExtensions = ['mp3', 'wav', 'flac', 'aac', 'ogg', 'm4a', 'wma'];
    const videoExtensions = ['mp4', 'mkv', 'avi', 'mov', 'wmv', 'flv', 'webm', 'm4v'];
    const subtitleExtensions = ['srt', 'vtt', 'ass', 'txt'];
    
    if (audioExtensions.includes(ext)) return 'audio';
    if (videoExtensions.includes(ext)) return 'video';
    if (subtitleExtensions.includes(ext)) return 'subtitle';
    return 'unknown';
}


// ==================== 通用工具函数 ====================

/**
 * HTML 转义
 * @param {string} text - 原始文本
 * @returns {string} 转义后的文本
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

/**
 * 格式化时间
 * @param {number} seconds - 秒数
 * @returns {string} 格式化后的时间 (HH:MM:SS)
 */
function formatDuration(seconds) {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.floor(seconds % 60);
    
    if (h > 0) {
        return `${h}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
    }
    return `${m}:${s.toString().padStart(2, '0')}`;
}

/**
 * 格式化日期时间
 * @param {number} timestamp - Unix 时间戳（秒）
 * @returns {string} 格式化后的日期时间
 */
function formatDateTime(timestamp) {
    const date = new Date(timestamp * 1000);
    return date.toLocaleString('zh-CN', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit'
    });
}

/**
 * 防抖函数
 * @param {Function} func - 要防抖的函数
 * @param {number} wait - 等待时间（毫秒）
 * @returns {Function} 防抖后的函数
 */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

/**
 * 节流函数
 * @param {Function} func - 要节流的函数
 * @param {number} limit - 时间限制（毫秒）
 * @returns {Function} 节流后的函数
 */
function throttle(func, limit) {
    let inThrottle;
    return function executedFunction(...args) {
        if (!inThrottle) {
            func(...args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}


// ==================== 进度轮询 ====================

/**
 * 任务状态轮询器
 */
class TaskPoller {
    constructor(taskId, onUpdate, onComplete, onError, interval = 1000) {
        this.taskId = taskId;
        this.onUpdate = onUpdate;
        this.onComplete = onComplete;
        this.onError = onError;
        this.interval = interval;
        this.timer = null;
        this.stopped = false;
    }
    
    /**
     * 开始轮询
     */
    start() {
        this.stopped = false;
        this.poll();
    }
    
    /**
     * 停止轮询
     */
    stop() {
        this.stopped = true;
        if (this.timer) {
            clearTimeout(this.timer);
            this.timer = null;
        }
    }
    
    /**
     * 执行一次轮询
     */
    async poll() {
        if (this.stopped) return;
        
        try {
            const response = await apiGet(`/api/task/${this.taskId}/status`);
            const task = response.data;
            
            // 回调更新
            if (this.onUpdate) {
                this.onUpdate(task);
            }
            
            // 检查是否完成
            if (task.status === 'completed') {
                this.stop();
                if (this.onComplete) {
                    this.onComplete(task);
                }
                return;
            }
            
            // 检查是否失败
            if (task.status === 'failed' || task.status === 'cancelled') {
                this.stop();
                if (this.onError) {
                    this.onError(task.error || '任务失败');
                }
                return;
            }
            
            // 继续轮询
            this.timer = setTimeout(() => this.poll(), this.interval);
            
        } catch (error) {
            this.stop();
            if (this.onError) {
                this.onError(error.message);
            }
        }
    }
}


// ==================== 文件上传工具 ====================

/**
 * 文件上传器
 */
class FileUploader {
    constructor(options = {}) {
        this.url = options.url || '/api/upload';
        this.purpose = options.purpose || 'whisper';
        this.onProgress = options.onProgress || null;
        this.onSuccess = options.onSuccess || null;
        this.onError = options.onError || null;
    }
    
    /**
     * 上传单个文件
     * @param {File} file - 文件对象
     * @returns {Promise<Object>} 上传结果
     */
    async upload(file) {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('purpose', this.purpose);
        
        try {
            const response = await apiPostForm(this.url, formData);
            
            if (this.onSuccess) {
                this.onSuccess(response.data);
            }
            
            return response.data;
        } catch (error) {
            if (this.onError) {
                this.onError(error.message);
            }
            throw error;
        }
    }
    
    /**
     * 上传多个文件
     * @param {FileList|Array<File>} files - 文件列表
     * @returns {Promise<Object>} 上传结果
     */
    async uploadBatch(files) {
        const formData = new FormData();
        for (const file of files) {
            formData.append('files', file);
        }
        formData.append('purpose', this.purpose);
        
        try {
            const response = await apiPostForm(this.url + '/batch', formData);
            
            if (this.onSuccess) {
                this.onSuccess(response.data);
            }
            
            return response.data;
        } catch (error) {
            if (this.onError) {
                this.onError(error.message);
            }
            throw error;
        }
    }
}


// ==================== DOM 工具 ====================

/**
 * 等待 DOM 加载完成
 * @param {Function} callback - 回调函数
 */
function onReady(callback) {
    if (document.readyState !== 'loading') {
        callback();
    } else {
        document.addEventListener('DOMContentLoaded', callback);
    }
}

/**
 * 创建 DOM 元素
 * @param {string} tag - 标签名
 * @param {Object} attrs - 属性
 * @param {string|Array} children - 子元素
 * @returns {HTMLElement} DOM 元素
 */
function createElement(tag, attrs = {}, children = []) {
    const element = document.createElement(tag);
    
    for (const [key, value] of Object.entries(attrs)) {
        if (key === 'className') {
            element.className = value;
        } else if (key === 'style' && typeof value === 'object') {
            Object.assign(element.style, value);
        } else if (key.startsWith('on') && typeof value === 'function') {
            element.addEventListener(key.slice(2).toLowerCase(), value);
        } else {
            element.setAttribute(key, value);
        }
    }
    
    if (typeof children === 'string') {
        element.textContent = children;
    } else if (Array.isArray(children)) {
        children.forEach(child => {
            if (typeof child === 'string') {
                element.appendChild(document.createTextNode(child));
            } else if (child instanceof HTMLElement) {
                element.appendChild(child);
            }
        });
    }
    
    return element;
}
