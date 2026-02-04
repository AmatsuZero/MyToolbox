"""
Whisper API 路由

处理 Whisper 字幕提取相关的 API 请求
"""

import os
import threading
from pathlib import Path
from flask import request, current_app

from . import api_bp, success_response, error_response, handle_api_error
from ..task_manager import task_manager, TaskStatus, TaskStage


def get_whisper_integration():
    """延迟导入并获取 Whisper 集成"""
    try:
        from modules.whisper.whisper_integration import WhisperIntegration, create_model_config
        return WhisperIntegration, create_model_config
    except ImportError as e:
        raise ImportError(f"Whisper 模块未安装: {str(e)}")


def get_subtitle_generator():
    """延迟导入并获取字幕生成器"""
    try:
        from modules.whisper.subtitle_generator import SubtitleGenerator
        return SubtitleGenerator
    except ImportError as e:
        raise ImportError(f"字幕生成器模块未安装: {str(e)}")


def process_whisper_task(task_id: str, file_path: str, options: dict):
    """
    在后台线程中处理 Whisper 任务
    
    Args:
        task_id: 任务ID
        file_path: 音视频文件路径
        options: 处理选项
    """
    try:
        # 更新状态：开始处理
        task_manager.update_task(
            task_id,
            status=TaskStatus.PROCESSING,
            stage=TaskStage.EXTRACTING.value,
            progress=10,
            message='正在提取音频...'
        )
        
        # 获取组件
        WhisperIntegration, create_model_config = get_whisper_integration()
        SubtitleGenerator = get_subtitle_generator()
        
        # 创建模型配置
        model_size = options.get('model', 'base')
        language = options.get('language', 'auto')
        output_format = options.get('format', 'all')
        
        model_config = create_model_config(
            model_size=model_size,
            language=language if language != 'auto' else None,
            device='auto'
        )
        
        # 更新状态：加载模型
        task_manager.update_task(
            task_id,
            progress=20,
            message=f'正在加载 {model_size} 模型...'
        )
        
        # 初始化 Whisper
        whisper = WhisperIntegration(model_config)
        if not whisper.load_model(model_size):
            raise RuntimeError('模型加载失败')
        
        # 更新状态：语音识别
        task_manager.update_task(
            task_id,
            stage=TaskStage.TRANSCRIBING.value,
            progress=40,
            message='正在进行语音识别...'
        )
        
        # 执行语音识别
        file_path_obj = Path(file_path)
        result = whisper.transcribe_audio(file_path_obj, language=language if language != 'auto' else None)
        
        if not result.success:
            raise RuntimeError(f'语音识别失败: {result.error_message}')
        
        # 更新状态：生成字幕
        task_manager.update_task(
            task_id,
            stage=TaskStage.GENERATING.value,
            progress=70,
            message='正在生成字幕文件...'
        )
        
        # 确定输出目录
        output_dir = current_app.config.get('OUTPUT_FOLDER', 'output')
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化字幕生成器
        subtitle_gen = SubtitleGenerator(output_dir)
        
        # 创建字幕片段
        segments = subtitle_gen.create_segments_from_transcription(result)
        
        # 生成字幕文件
        output_files = {}
        base_name = file_path_obj.stem
        
        formats_to_generate = ['srt', 'vtt', 'txt', 'json', 'csv'] if output_format == 'all' else [output_format]
        
        for fmt in formats_to_generate:
            output_name = f"{base_name}.{fmt}"
            if fmt == 'srt':
                output_path = subtitle_gen.generate_srt(segments, output_name)
            elif fmt == 'vtt':
                output_path = subtitle_gen.generate_vtt(segments, output_name)
            elif fmt == 'txt':
                output_path = subtitle_gen.generate_txt(segments, output_name)
            elif fmt == 'json':
                output_path = subtitle_gen.generate_json(segments, output_name)
            elif fmt == 'csv':
                output_path = subtitle_gen.generate_csv(segments, output_name)
            else:
                continue
            
            output_files[fmt] = str(output_path)
        
        # 更新状态：完成
        task_manager.update_task(
            task_id,
            status=TaskStatus.COMPLETED,
            stage=TaskStage.DOWNLOAD_READY.value,
            progress=100,
            message='处理完成',
            result={
                'text': result.text,
                'language': result.language,
                'confidence': result.confidence,
                'duration': result.duration,
                'segments_count': len(segments)
            },
            output_files=output_files
        )
        
    except Exception as e:
        # 更新状态：失败
        task_manager.update_task(
            task_id,
            status=TaskStatus.FAILED,
            progress=0,
            message='处理失败',
            error=str(e)
        )


@api_bp.route('/whisper/start', methods=['POST'])
@handle_api_error
def start_whisper_task():
    """
    启动 Whisper 语音识别任务
    
    POST /api/whisper/start
    
    JSON 请求体:
        {
            "file_id": "上传文件的ID",
            "file_path": "文件路径（可选，优先使用file_id）",
            "model": "模型大小 (tiny/base/small/medium/large)",
            "language": "语言代码或 'auto'",
            "format": "输出格式 (srt/vtt/txt/json/csv/all)"
        }
        
    返回:
        {
            "success": true,
            "data": {
                "task_id": "任务ID"
            }
        }
    """
    data = request.get_json() or {}
    
    # 获取文件路径
    file_id = data.get('file_id')
    file_path = data.get('file_path')
    
    if not file_id and not file_path:
        return error_response('NO_FILE', '必须提供 file_id 或 file_path', status_code=400)
    
    # 如果提供了 file_id，查找文件
    if file_id:
        upload_folder = Path(current_app.config.get('UPLOAD_FOLDER', 'uploads'))
        found = False
        for f in upload_folder.iterdir():
            if f.stem == file_id:
                file_path = str(f)
                found = True
                break
        if not found:
            return error_response('FILE_NOT_FOUND', f'文件 {file_id} 不存在', status_code=404)
    
    # 验证文件存在
    if not os.path.exists(file_path):
        return error_response('FILE_NOT_FOUND', f'文件不存在: {file_path}', status_code=404)
    
    # 获取处理选项
    options = {
        'model': data.get('model', 'base'),
        'language': data.get('language', 'auto'),
        'format': data.get('format', 'all')
    }
    
    # 验证模型大小
    valid_models = ['tiny', 'base', 'small', 'medium', 'large']
    if options['model'] not in valid_models:
        return error_response(
            'INVALID_MODEL',
            f'无效的模型大小，可选值: {", ".join(valid_models)}',
            status_code=400
        )
    
    # 创建任务
    filename = os.path.basename(file_path)
    task = task_manager.create_task('whisper', filename)
    
    # 更新初始状态
    task_manager.update_task(
        task.task_id,
        status=TaskStatus.PENDING,
        stage='queued',
        progress=0,
        message='任务已创建，等待处理'
    )
    
    # 在后台线程中处理任务
    thread = threading.Thread(
        target=process_whisper_task,
        args=(task.task_id, file_path, options),
        daemon=True
    )
    thread.start()
    
    return success_response({
        'task_id': task.task_id,
        'file_id': task.file_id,
        'filename': filename,
        'options': options
    }, '任务已创建')


@api_bp.route('/whisper/models', methods=['GET'])
@handle_api_error
def get_whisper_models():
    """
    获取可用的 Whisper 模型列表
    
    GET /api/whisper/models
    """
    models = [
        {'id': 'tiny', 'name': 'Tiny', 'size': '~75MB', 'speed': '最快', 'accuracy': '较低'},
        {'id': 'base', 'name': 'Base', 'size': '~150MB', 'speed': '快', 'accuracy': '中等'},
        {'id': 'small', 'name': 'Small', 'size': '~500MB', 'speed': '中等', 'accuracy': '较好'},
        {'id': 'medium', 'name': 'Medium', 'size': '~1.5GB', 'speed': '较慢', 'accuracy': '很好'},
        {'id': 'large', 'name': 'Large', 'size': '~3GB', 'speed': '慢', 'accuracy': '最佳'},
    ]
    
    return success_response({'models': models})


@api_bp.route('/whisper/languages', methods=['GET'])
@handle_api_error
def get_whisper_languages():
    """
    获取支持的语言列表
    
    GET /api/whisper/languages
    """
    languages = [
        {'code': 'auto', 'name': '自动检测'},
        {'code': 'zh', 'name': '中文'},
        {'code': 'en', 'name': '英语'},
        {'code': 'ja', 'name': '日语'},
        {'code': 'ko', 'name': '韩语'},
        {'code': 'fr', 'name': '法语'},
        {'code': 'de', 'name': '德语'},
        {'code': 'es', 'name': '西班牙语'},
        {'code': 'ru', 'name': '俄语'},
        {'code': 'it', 'name': '意大利语'},
        {'code': 'pt', 'name': '葡萄牙语'},
    ]
    
    return success_response({'languages': languages})


@api_bp.route('/task/<task_id>/status', methods=['GET'])
@handle_api_error
def get_task_status(task_id: str):
    """
    获取任务状态
    
    GET /api/task/<task_id>/status
    """
    task = task_manager.get_task(task_id)
    
    if not task:
        return error_response('TASK_NOT_FOUND', f'任务 {task_id} 不存在', status_code=404)
    
    return success_response(task.to_dict())


@api_bp.route('/task/<task_id>/cancel', methods=['POST'])
@handle_api_error
def cancel_task(task_id: str):
    """
    取消任务
    
    POST /api/task/<task_id>/cancel
    """
    task = task_manager.get_task(task_id)
    
    if not task:
        return error_response('TASK_NOT_FOUND', f'任务 {task_id} 不存在', status_code=404)
    
    # 只能取消等待中或处理中的任务
    if task.status not in (TaskStatus.PENDING, TaskStatus.PROCESSING):
        return error_response(
            'CANNOT_CANCEL',
            f'无法取消状态为 {task.status.value} 的任务',
            status_code=400
        )
    
    task_manager.update_task(
        task_id,
        status=TaskStatus.CANCELLED,
        message='任务已取消'
    )
    
    return success_response(message='任务已取消')


@api_bp.route('/tasks', methods=['GET'])
@handle_api_error
def list_tasks():
    """
    获取任务列表
    
    GET /api/tasks
    
    查询参数:
        type: 任务类型过滤 ('whisper' 或 'train')
    """
    task_type = request.args.get('type')
    tasks = task_manager.list_tasks(task_type)
    
    return success_response({
        'tasks': [t.to_dict() for t in tasks],
        'total': len(tasks)
    })
