/**
 * Whisper 字幕提取页面 JavaScript
 */

// 全局状态
let uploadedFiles = [];
let currentTaskId = null;
let taskPoller = null;

// ==================== 初始化 ====================

onReady(() => {
    initUploadZone();
    initControls();
    loadTaskHistory();
});

/**
 * 初始化上传区域
 */
function initUploadZone() {
    const uploadZone = document.getElementById('uploadZone');
    const fileInput = document.getElementById('fileInput');
    
    // 点击上传
    uploadZone.addEventListener('click', () => fileInput.click());
    
    // 文件选择
    fileInput.addEventListener('change', (e) => {
        handleFiles(e.target.files);
    });
    
    // 拖拽事件
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('dragover');
    });
    
    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('dragover');
    });
    
    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('dragover');
        handleFiles(e.dataTransfer.files);
    });
}

/**
 * 初始化控件
 */
function initControls() {
    // 开始处理按钮
    document.getElementById('startBtn').addEventListener('click', startProcessing);
    
    // 刷新任务列表按钮
    document.getElementById('refreshTasksBtn').addEventListener('click', loadTaskHistory);
}

// ==================== 文件处理 ====================

/**
 * 处理选择的文件
 */
function handleFiles(files) {
    const fileList = document.getElementById('fileList');
    const fileListContent = document.getElementById('fileListContent');
    
    for (const file of files) {
        // 检查是否已添加
        if (uploadedFiles.some(f => f.name === file.name && f.size === file.size)) {
            continue;
        }
        
        // 验证文件类型
        const fileType = getFileType(file.name);
        if (fileType !== 'audio' && fileType !== 'video') {
            showWarning(`不支持的文件格式: ${file.name}`);
            continue;
        }
        
        uploadedFiles.push(file);
    }
    
    // 更新文件列表显示
    if (uploadedFiles.length > 0) {
        fileList.style.display = 'block';
        renderFileList();
        document.getElementById('startBtn').disabled = false;
    } else {
        fileList.style.display = 'none';
        document.getElementById('startBtn').disabled = true;
    }
}

/**
 * 渲染文件列表
 */
function renderFileList() {
    const container = document.getElementById('fileListContent');
    container.innerHTML = '';
    
    uploadedFiles.forEach((file, index) => {
        const fileType = getFileType(file.name);
        const html = `
            <div class="file-item">
                <div class="file-icon ${fileType}">
                    <i class="bi ${getFileIcon(file.name)}"></i>
                </div>
                <div class="file-info">
                    <div class="file-name">${escapeHtml(file.name)}</div>
                    <div class="file-size">${formatFileSize(file.size)}</div>
                </div>
                <div class="file-actions">
                    <button class="btn btn-sm btn-outline-danger" onclick="removeFile(${index})">
                        <i class="bi bi-x"></i>
                    </button>
                </div>
            </div>
        `;
        container.insertAdjacentHTML('beforeend', html);
    });
}

/**
 * 移除文件
 */
function removeFile(index) {
    uploadedFiles.splice(index, 1);
    
    if (uploadedFiles.length === 0) {
        document.getElementById('fileList').style.display = 'none';
        document.getElementById('startBtn').disabled = true;
    } else {
        renderFileList();
    }
}

// ==================== 处理流程 ====================

/**
 * 开始处理
 */
async function startProcessing() {
    if (uploadedFiles.length === 0) {
        showWarning('请先选择文件');
        return;
    }
    
    const file = uploadedFiles[0]; // 目前处理第一个文件
    const options = {
        model: document.getElementById('modelSelect').value,
        language: document.getElementById('languageSelect').value,
        format: document.getElementById('formatSelect').value
    };
    
    // 禁用按钮
    document.getElementById('startBtn').disabled = true;
    
    // 显示进度卡片
    showProgressCard();
    updateProgress(0, 'uploading', '正在上传文件...');
    
    try {
        // 1. 上传文件
        const uploader = new FileUploader({ purpose: 'whisper' });
        const uploadResult = await uploader.upload(file);
        
        updateProgress(10, 'uploaded', '文件上传完成，正在启动处理任务...');
        
        // 2. 启动 Whisper 任务
        const taskResult = await apiPost('/api/whisper/start', {
            file_id: uploadResult.file_id,
            ...options
        });
        
        currentTaskId = taskResult.data.task_id;
        
        // 3. 开始轮询任务状态
        startTaskPolling();
        
    } catch (error) {
        showError(error.message);
        hideProgressCard();
        document.getElementById('startBtn').disabled = false;
    }
}

/**
 * 开始任务状态轮询
 */
function startTaskPolling() {
    if (taskPoller) {
        taskPoller.stop();
    }
    
    taskPoller = new TaskPoller(
        currentTaskId,
        onTaskUpdate,
        onTaskComplete,
        onTaskError,
        1000
    );
    
    taskPoller.start();
}

/**
 * 任务更新回调
 */
function onTaskUpdate(task) {
    const stageNames = {
        'queued': '等待处理',
        'extracting': '提取音频',
        'transcribing': '语音识别',
        'generating': '生成字幕',
        'download_ready': '准备下载'
    };
    
    const stageName = stageNames[task.stage] || task.stage;
    updateProgress(task.progress, task.stage, task.message || stageName);
}

/**
 * 任务完成回调
 */
function onTaskComplete(task) {
    updateProgress(100, 'completed', '处理完成！');
    
    // 显示结果
    showResult(task);
    
    // 重新加载任务历史
    loadTaskHistory();
    
    // 重置状态
    currentTaskId = null;
    uploadedFiles = [];
    document.getElementById('fileList').style.display = 'none';
    document.getElementById('fileListContent').innerHTML = '';
    document.getElementById('startBtn').disabled = true;
}

/**
 * 任务失败回调
 */
function onTaskError(error) {
    showError(error);
    hideProgressCard();
    document.getElementById('startBtn').disabled = false;
}

// ==================== UI 更新 ====================

/**
 * 显示进度卡片
 */
function showProgressCard() {
    document.getElementById('progressCard').style.display = 'block';
    document.getElementById('resultCard').style.display = 'none';
}

/**
 * 隐藏进度卡片
 */
function hideProgressCard() {
    document.getElementById('progressCard').style.display = 'none';
}

/**
 * 更新进度
 */
function updateProgress(percent, stage, message) {
    const progressBar = document.getElementById('progressBar');
    const progressPercent = document.getElementById('progressPercent');
    const progressStage = document.getElementById('progressStage');
    const progressMessage = document.getElementById('progressMessage');
    
    progressBar.style.width = `${percent}%`;
    progressPercent.textContent = `${Math.round(percent)}%`;
    
    const stageNames = {
        'uploading': '上传中',
        'uploaded': '已上传',
        'queued': '排队中',
        'extracting': '提取音频',
        'transcribing': '语音识别',
        'generating': '生成字幕',
        'download_ready': '下载就绪',
        'completed': '已完成'
    };
    
    progressStage.textContent = stageNames[stage] || stage;
    progressMessage.textContent = message;
    
    if (percent >= 100) {
        progressBar.classList.remove('progress-bar-animated');
    }
}

/**
 * 显示处理结果
 */
function showResult(task) {
    const resultCard = document.getElementById('resultCard');
    resultCard.style.display = 'block';
    
    // 填充结果数据
    if (task.result) {
        const languageNames = {
            'zh': '中文', 'en': '英语', 'ja': '日语', 'ko': '韩语',
            'fr': '法语', 'de': '德语', 'es': '西班牙语', 'ru': '俄语'
        };
        
        document.getElementById('resultLanguage').textContent = 
            languageNames[task.result.language] || task.result.language || '未知';
        document.getElementById('resultConfidence').textContent = 
            task.result.confidence ? `${(task.result.confidence * 100).toFixed(1)}%` : '-';
        document.getElementById('resultDuration').textContent = 
            task.result.duration ? formatDuration(task.result.duration) : '-';
        document.getElementById('resultSegments').textContent = 
            task.result.segments_count || '-';
    }
    
    // 生成下载链接
    const downloadList = document.getElementById('downloadList');
    downloadList.innerHTML = '';
    
    if (task.output_files) {
        const formatIcons = {
            'srt': 'bi-file-earmark-text',
            'vtt': 'bi-file-earmark-code',
            'txt': 'bi-file-earmark-font',
            'json': 'bi-file-earmark-code',
            'csv': 'bi-file-earmark-spreadsheet'
        };
        
        for (const [format, path] of Object.entries(task.output_files)) {
            const filename = path.split('/').pop();
            const fileId = filename.split('.')[0];
            const html = `
                <a href="/api/download/${fileId}?type=output" class="download-item" download>
                    <i class="bi ${formatIcons[format] || 'bi-file-earmark'} me-2 fs-4"></i>
                    <div>
                        <div class="fw-bold">${format.toUpperCase()}</div>
                        <div class="small text-muted">${filename}</div>
                    </div>
                </a>
            `;
            downloadList.insertAdjacentHTML('beforeend', html);
        }
    }
}

// ==================== 任务历史 ====================

/**
 * 加载任务历史
 */
async function loadTaskHistory() {
    try {
        const response = await apiGet('/api/tasks?type=whisper');
        renderTaskHistory(response.data.tasks);
    } catch (error) {
        console.error('加载任务历史失败:', error);
    }
}

/**
 * 渲染任务历史列表
 */
function renderTaskHistory(tasks) {
    const container = document.getElementById('taskListContent');
    
    if (!tasks || tasks.length === 0) {
        container.innerHTML = '<p class="text-muted text-center mb-0">暂无任务</p>';
        return;
    }
    
    const statusClasses = {
        'pending': 'text-secondary',
        'processing': 'text-info',
        'completed': 'text-success',
        'failed': 'text-danger',
        'cancelled': 'text-warning'
    };
    
    const statusIcons = {
        'pending': 'bi-clock',
        'processing': 'bi-arrow-repeat',
        'completed': 'bi-check-circle',
        'failed': 'bi-x-circle',
        'cancelled': 'bi-slash-circle'
    };
    
    let html = '<div class="list-group list-group-flush">';
    
    for (const task of tasks.slice(0, 10)) { // 只显示最近10个
        const statusClass = statusClasses[task.status] || 'text-secondary';
        const statusIcon = statusIcons[task.status] || 'bi-circle';
        
        html += `
            <div class="list-group-item d-flex justify-content-between align-items-center">
                <div>
                    <div class="fw-bold">${escapeHtml(task.filename || 'Unknown')}</div>
                    <small class="text-muted">
                        ${formatDateTime(task.created_at)}
                    </small>
                </div>
                <span class="${statusClass}">
                    <i class="bi ${statusIcon} me-1"></i>
                    ${task.status}
                </span>
            </div>
        `;
    }
    
    html += '</div>';
    container.innerHTML = html;
}
