/**
 * 语音训练页面 JavaScript
 */

// 全局状态
let audioFiles = [];
let subtitleFiles = [];
let currentTrainTaskId = null;
let trainTaskPoller = null;

// ==================== 初始化 ====================

onReady(() => {
    initAudioUploadZone();
    initSubtitleUploadZone();
    initTrainControls();
    loadTrainTaskHistory();
});

/**
 * 初始化音频上传区域
 */
function initAudioUploadZone() {
    const uploadZone = document.getElementById('audioUploadZone');
    const fileInput = document.getElementById('audioFileInput');
    
    uploadZone.addEventListener('click', () => fileInput.click());
    
    fileInput.addEventListener('change', (e) => {
        handleAudioFiles(e.target.files);
    });
    
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
        handleAudioFiles(e.dataTransfer.files);
    });
}

/**
 * 初始化字幕上传区域
 */
function initSubtitleUploadZone() {
    const uploadZone = document.getElementById('subtitleUploadZone');
    const fileInput = document.getElementById('subtitleFileInput');
    
    uploadZone.addEventListener('click', () => fileInput.click());
    
    fileInput.addEventListener('change', (e) => {
        handleSubtitleFiles(e.target.files);
    });
    
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
        handleSubtitleFiles(e.dataTransfer.files);
    });
}

/**
 * 初始化训练控件
 */
function initTrainControls() {
    document.getElementById('startTrainBtn').addEventListener('click', startTraining);
    document.getElementById('stopTrainBtn').addEventListener('click', stopTraining);
    document.getElementById('refreshTrainTasksBtn').addEventListener('click', loadTrainTaskHistory);
}

// ==================== 文件处理 ====================

/**
 * 处理音频文件
 */
function handleAudioFiles(files) {
    for (const file of files) {
        if (audioFiles.some(f => f.name === file.name && f.size === file.size)) {
            continue;
        }
        
        const fileType = getFileType(file.name);
        if (fileType !== 'audio') {
            showWarning(`不支持的文件格式: ${file.name}`);
            continue;
        }
        
        audioFiles.push(file);
    }
    
    updateAudioFileList();
    updateStartButton();
}

/**
 * 处理字幕文件
 */
function handleSubtitleFiles(files) {
    for (const file of files) {
        if (subtitleFiles.some(f => f.name === file.name && f.size === file.size)) {
            continue;
        }
        
        const fileType = getFileType(file.name);
        if (fileType !== 'subtitle') {
            showWarning(`不支持的文件格式: ${file.name}`);
            continue;
        }
        
        subtitleFiles.push(file);
    }
    
    updateSubtitleFileList();
}

/**
 * 更新音频文件列表显示
 */
function updateAudioFileList() {
    const container = document.getElementById('audioFileList');
    const content = document.getElementById('audioFileListContent');
    
    if (audioFiles.length === 0) {
        container.style.display = 'none';
        return;
    }
    
    container.style.display = 'block';
    content.innerHTML = '';
    
    audioFiles.forEach((file, index) => {
        const html = `
            <div class="file-item">
                <div class="file-icon audio">
                    <i class="bi bi-file-earmark-music"></i>
                </div>
                <div class="file-info">
                    <div class="file-name">${escapeHtml(file.name)}</div>
                    <div class="file-size">${formatFileSize(file.size)}</div>
                </div>
                <div class="file-actions">
                    <button class="btn btn-sm btn-outline-danger" onclick="removeAudioFile(${index})">
                        <i class="bi bi-x"></i>
                    </button>
                </div>
            </div>
        `;
        content.insertAdjacentHTML('beforeend', html);
    });
}

/**
 * 更新字幕文件列表显示
 */
function updateSubtitleFileList() {
    const container = document.getElementById('subtitleFileList');
    const content = document.getElementById('subtitleFileListContent');
    
    if (subtitleFiles.length === 0) {
        container.style.display = 'none';
        return;
    }
    
    container.style.display = 'block';
    content.innerHTML = '';
    
    subtitleFiles.forEach((file, index) => {
        const html = `
            <div class="file-item">
                <div class="file-icon subtitle">
                    <i class="bi bi-file-earmark-text"></i>
                </div>
                <div class="file-info">
                    <div class="file-name">${escapeHtml(file.name)}</div>
                    <div class="file-size">${formatFileSize(file.size)}</div>
                </div>
                <div class="file-actions">
                    <button class="btn btn-sm btn-outline-danger" onclick="removeSubtitleFile(${index})">
                        <i class="bi bi-x"></i>
                    </button>
                </div>
            </div>
        `;
        content.insertAdjacentHTML('beforeend', html);
    });
}

/**
 * 移除音频文件
 */
function removeAudioFile(index) {
    audioFiles.splice(index, 1);
    updateAudioFileList();
    updateStartButton();
}

/**
 * 移除字幕文件
 */
function removeSubtitleFile(index) {
    subtitleFiles.splice(index, 1);
    updateSubtitleFileList();
}

/**
 * 更新开始按钮状态
 */
function updateStartButton() {
    document.getElementById('startTrainBtn').disabled = audioFiles.length === 0;
}

// ==================== 训练流程 ====================

/**
 * 开始训练
 */
async function startTraining() {
    if (audioFiles.length === 0) {
        showWarning('请先上传音频文件');
        return;
    }
    
    // 禁用按钮
    document.getElementById('startTrainBtn').disabled = true;
    
    // 显示进度卡片
    showTrainProgressCard();
    updateTrainProgress(0, 'uploading', '正在上传训练数据...');
    
    try {
        // 1. 上传音频文件
        const uploader = new FileUploader({ purpose: 'train' });
        const uploadedAudioPaths = [];
        
        for (let i = 0; i < audioFiles.length; i++) {
            const result = await uploader.upload(audioFiles[i]);
            uploadedAudioPaths.push(result.path);
            const progress = ((i + 1) / audioFiles.length) * 5;
            updateTrainProgress(progress, 'uploading', `上传音频: ${i + 1}/${audioFiles.length}`);
        }
        
        // 2. 上传字幕文件（如果有）
        const uploadedSubtitlePaths = [];
        for (let i = 0; i < subtitleFiles.length; i++) {
            const result = await uploader.upload(subtitleFiles[i]);
            uploadedSubtitlePaths.push(result.path);
        }
        
        updateTrainProgress(10, 'starting', '正在启动训练任务...');
        
        // 3. 启动训练任务
        const config = {
            audio_files: uploadedAudioPaths,
            subtitle_files: uploadedSubtitlePaths,
            speaker_name: document.getElementById('speakerName').value || 'custom_voice',
            epochs: parseInt(document.getElementById('epochsInput').value) || 100,
            batch_size: parseInt(document.getElementById('batchSizeInput').value) || 32,
            sample_rate: parseInt(document.getElementById('sampleRateSelect').value) || 22050
        };
        
        const taskResult = await apiPost('/api/train/start', config);
        currentTrainTaskId = taskResult.data.task_id;
        
        // 4. 开始轮询任务状态
        startTrainTaskPolling();
        
    } catch (error) {
        showError(error.message);
        hideTrainProgressCard();
        document.getElementById('startTrainBtn').disabled = false;
    }
}

/**
 * 停止训练
 */
async function stopTraining() {
    if (!currentTrainTaskId) return;
    
    try {
        await apiPost(`/api/train/${currentTrainTaskId}/stop`);
        showInfo('训练已停止');
        
        if (trainTaskPoller) {
            trainTaskPoller.stop();
        }
        
        hideTrainProgressCard();
        document.getElementById('startTrainBtn').disabled = false;
        
    } catch (error) {
        showError(error.message);
    }
}

/**
 * 开始训练任务状态轮询
 */
function startTrainTaskPolling() {
    if (trainTaskPoller) {
        trainTaskPoller.stop();
    }
    
    trainTaskPoller = new TaskPoller(
        currentTrainTaskId,
        onTrainTaskUpdate,
        onTrainTaskComplete,
        onTrainTaskError,
        2000 // 训练任务轮询间隔更长
    );
    
    trainTaskPoller.start();
}

/**
 * 训练任务更新回调
 */
function onTrainTaskUpdate(task) {
    const stageNames = {
        'queued': '等待处理',
        'initializing': '初始化',
        'preparing': '准备数据',
        'training': '训练中',
        'completed': '已完成'
    };
    
    const stageName = stageNames[task.stage] || task.stage;
    updateTrainProgress(task.progress, task.stage, task.message || stageName);
    
    // 更新训练状态面板
    document.getElementById('trainStatus').textContent = stageName;
    
    // 解析消息中的 epoch 和 loss 信息
    if (task.message && task.message.includes('Epoch')) {
        const epochMatch = task.message.match(/Epoch (\d+)\/(\d+)/);
        const lossMatch = task.message.match(/Loss: ([\d.]+)/);
        
        if (epochMatch) {
            document.getElementById('currentEpoch').textContent = `${epochMatch[1]}/${epochMatch[2]}`;
        }
        if (lossMatch) {
            document.getElementById('currentLoss').textContent = lossMatch[1];
        }
    }
}

/**
 * 训练任务完成回调
 */
function onTrainTaskComplete(task) {
    updateTrainProgress(100, 'completed', '训练完成！');
    
    // 显示结果
    showTrainResult(task);
    
    // 重新加载任务历史
    loadTrainTaskHistory();
    
    // 重置状态
    currentTrainTaskId = null;
    audioFiles = [];
    subtitleFiles = [];
    updateAudioFileList();
    updateSubtitleFileList();
    updateStartButton();
}

/**
 * 训练任务失败回调
 */
function onTrainTaskError(error) {
    showError(error);
    hideTrainProgressCard();
    document.getElementById('startTrainBtn').disabled = false;
}

// ==================== UI 更新 ====================

/**
 * 显示训练进度卡片
 */
function showTrainProgressCard() {
    document.getElementById('trainProgressCard').style.display = 'block';
    document.getElementById('trainResultCard').style.display = 'none';
}

/**
 * 隐藏训练进度卡片
 */
function hideTrainProgressCard() {
    document.getElementById('trainProgressCard').style.display = 'none';
}

/**
 * 更新训练进度
 */
function updateTrainProgress(percent, stage, message) {
    const progressBar = document.getElementById('trainProgressBar');
    const progressPercent = document.getElementById('trainPercent');
    const progressStage = document.getElementById('trainStage');
    const progressMessage = document.getElementById('trainMessage');
    
    progressBar.style.width = `${percent}%`;
    progressPercent.textContent = `${Math.round(percent)}%`;
    
    const stageNames = {
        'uploading': '上传中',
        'starting': '启动中',
        'queued': '排队中',
        'initializing': '初始化',
        'preparing': '准备数据',
        'training': '训练中',
        'completed': '已完成'
    };
    
    progressStage.textContent = stageNames[stage] || stage;
    progressMessage.textContent = message;
    
    if (percent >= 100) {
        progressBar.classList.remove('progress-bar-animated');
    }
}

/**
 * 显示训练结果
 */
function showTrainResult(task) {
    const resultCard = document.getElementById('trainResultCard');
    resultCard.style.display = 'block';
    
    // 填充结果数据
    if (task.result) {
        document.getElementById('resultSpeaker').textContent = 
            task.result.speaker_name || '-';
        document.getElementById('resultEpochs').textContent = 
            task.result.epochs || '-';
        document.getElementById('resultOutputDir').textContent = 
            task.result.output_dir || '-';
    }
    
    // 生成下载链接
    const downloadList = document.getElementById('trainDownloadList');
    downloadList.innerHTML = '';
    
    if (task.output_files && Object.keys(task.output_files).length > 0) {
        for (const [format, path] of Object.entries(task.output_files)) {
            const filename = path.split('/').pop();
            const fileId = filename.split('.')[0];
            const html = `
                <a href="/api/download/${fileId}?type=output" class="download-item" download>
                    <i class="bi bi-file-earmark-binary me-2 fs-4"></i>
                    <div>
                        <div class="fw-bold">${format.toUpperCase()}</div>
                        <div class="small text-muted">${filename}</div>
                    </div>
                </a>
            `;
            downloadList.insertAdjacentHTML('beforeend', html);
        }
    } else {
        downloadList.innerHTML = '<p class="text-muted mb-0">模型文件请在输出目录中查找</p>';
    }
}

// ==================== 任务历史 ====================

/**
 * 加载训练任务历史
 */
async function loadTrainTaskHistory() {
    try {
        const response = await apiGet('/api/tasks?type=train');
        renderTrainTaskHistory(response.data.tasks);
    } catch (error) {
        console.error('加载训练任务历史失败:', error);
    }
}

/**
 * 渲染训练任务历史列表
 */
function renderTrainTaskHistory(tasks) {
    const container = document.getElementById('trainTaskListContent');
    
    if (!tasks || tasks.length === 0) {
        container.innerHTML = '<p class="text-muted text-center mb-0">暂无训练任务</p>';
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
    
    for (const task of tasks.slice(0, 10)) {
        const statusClass = statusClasses[task.status] || 'text-secondary';
        const statusIcon = statusIcons[task.status] || 'bi-circle';
        
        html += `
            <div class="list-group-item d-flex justify-content-between align-items-center">
                <div>
                    <div class="fw-bold">${escapeHtml(task.filename || 'Training Task')}</div>
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
