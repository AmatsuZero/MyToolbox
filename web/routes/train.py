"""
训练 API 路由

处理语音训练相关的 API 请求
"""

import os
import threading
from pathlib import Path
from flask import request, current_app

from . import api_bp, success_response, error_response, handle_api_error
from ..task_manager import task_manager, TaskStatus


def get_train_controller():
    """延迟导入并获取训练控制器"""
    try:
        from modules.training.train_controller import TrainController
        from modules.training.train_config import TrainConfig
        return TrainController, TrainConfig
    except ImportError as e:
        raise ImportError(f"训练模块未安装: {str(e)}")


def process_train_task(task_id: str, config_dict: dict):
    """
    在后台线程中处理训练任务
    
    Args:
        task_id: 任务ID
        config_dict: 训练配置字典
    """
    try:
        # 更新状态：开始处理
        task_manager.update_task(
            task_id,
            status=TaskStatus.PROCESSING,
            stage='initializing',
            progress=5,
            message='正在初始化训练环境...'
        )
        
        # 获取训练组件
        TrainController, TrainConfig = get_train_controller()
        
        # 创建训练配置
        train_config = TrainConfig(
            input_dir=Path(config_dict['input_dir']) if config_dict.get('input_dir') else None,
            output_path=Path(config_dict.get('output_dir', 'trained_models')),
            speaker_name=config_dict.get('speaker_name', 'custom_voice'),
            model_type=config_dict.get('model_type', 'vits'),
        language=config_dict.get('language', 'auto'),  # 默认自动检测
            epochs=config_dict.get('epochs', 100),
            batch_size=config_dict.get('batch_size', 32),
            sample_rate=config_dict.get('sample_rate', 22050),
            verbose=True
        )
        
        # 如果指定了单个文件
        if config_dict.get('audio_files'):
            train_config.audio_files = [Path(f) for f in config_dict['audio_files']]
        if config_dict.get('subtitle_files'):
            train_config.subtitle_files = [Path(f) for f in config_dict['subtitle_files']]
        
        # 更新状态：准备数据
        task_manager.update_task(
            task_id,
            stage='preparing',
            progress=10,
            message='正在准备训练数据...'
        )
        
        # 创建训练控制器
        controller = TrainController(train_config, verbose=True)
        
        # 定义进度回调
        def progress_callback(epoch: int, total_epochs: int, loss: float, stage: str):
            progress = 10 + (epoch / total_epochs) * 85  # 10% - 95%
            task_manager.update_task(
                task_id,
                stage=stage,
                progress=progress,
                message=f'训练中: Epoch {epoch}/{total_epochs}, Loss: {loss:.4f}'
            )
        
        # 如果控制器支持进度回调，设置它
        if hasattr(controller, 'set_progress_callback'):
            controller.set_progress_callback(progress_callback)
        
        # 更新状态：开始训练
        task_manager.update_task(
            task_id,
            stage='training',
            progress=15,
            message='正在训练模型...'
        )
        
        # 运行训练
        result = controller.run()
        
        if not result.success:
            raise RuntimeError(f'训练失败: {result.error_message if hasattr(result, "error_message") else "未知错误"}')
        
        # 获取输出文件
        output_files = {}
        output_dir = Path(config_dict.get('output_dir', 'trained_models'))
        if output_dir.exists():
            for f in output_dir.iterdir():
                if f.is_file():
                    output_files[f.suffix.lstrip('.')] = str(f)
        
        # 更新状态：完成
        task_manager.update_task(
            task_id,
            status=TaskStatus.COMPLETED,
            stage='completed',
            progress=100,
            message='训练完成',
            result={
                'speaker_name': config_dict.get('speaker_name', 'custom_voice'),
                'epochs': config_dict.get('epochs', 100),
                'output_dir': str(output_dir)
            },
            output_files=output_files
        )
        
    except Exception as e:
        # 更新状态：失败
        task_manager.update_task(
            task_id,
            status=TaskStatus.FAILED,
            progress=0,
            message='训练失败',
            error=str(e)
        )


@api_bp.route('/train/start', methods=['POST'])
@handle_api_error
def start_train_task():
    """
    启动语音训练任务
    
    POST /api/train/start
    
    JSON 请求体:
        {
            "input_dir": "训练数据目录",
            "audio_files": ["音频文件路径列表（可选）"],
            "subtitle_files": ["字幕文件路径列表（可选）"],
            "output_dir": "模型输出目录",
            "speaker_name": "说话人名称",
            "epochs": 训练轮数,
            "batch_size": 批处理大小,
            "sample_rate": 采样率
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
    
    # 验证必需参数
    input_dir = data.get('input_dir')
    audio_files = data.get('audio_files', [])
    
    if not input_dir and not audio_files:
        return error_response(
            'NO_INPUT',
            '必须提供 input_dir 或 audio_files',
            status_code=400
        )
    
    # 验证输入目录存在
    if input_dir and not os.path.exists(input_dir):
        return error_response('DIR_NOT_FOUND', f'目录不存在: {input_dir}', status_code=404)
    
    # 验证音频文件存在
    for audio_file in audio_files:
        if not os.path.exists(audio_file):
            return error_response('FILE_NOT_FOUND', f'音频文件不存在: {audio_file}', status_code=404)
    
    # 获取训练参数
    config_dict = {
        'input_dir': input_dir,
        'audio_files': audio_files,
        'subtitle_files': data.get('subtitle_files', []),
        'output_dir': data.get('output_dir', 'trained_models'),
        'speaker_name': data.get('speaker_name', 'custom_voice'),
        'model_type': data.get('model_type', 'vits'),
        'language': data.get('language', 'auto'),  # 默认自动检测
        'epochs': data.get('epochs', 100),
        'batch_size': data.get('batch_size', 32),
        'sample_rate': data.get('sample_rate', 22050)
    }
    
    # 验证参数范围
    if config_dict['epochs'] < 1:
        return error_response('INVALID_EPOCHS', '训练轮数必须大于0', status_code=400)
    if config_dict['batch_size'] < 1:
        return error_response('INVALID_BATCH_SIZE', '批处理大小必须大于0', status_code=400)
    if config_dict['sample_rate'] < 8000:
        return error_response('INVALID_SAMPLE_RATE', '采样率不能低于8000', status_code=400)
    
    # 创建任务
    task = task_manager.create_task('train', config_dict.get('speaker_name', 'custom_voice'))
    
    # 更新初始状态
    task_manager.update_task(
        task.task_id,
        status=TaskStatus.PENDING,
        stage='queued',
        progress=0,
        message='训练任务已创建，等待处理'
    )
    
    # 在后台线程中处理任务
    thread = threading.Thread(
        target=process_train_task,
        args=(task.task_id, config_dict),
        daemon=True
    )
    thread.start()
    
    return success_response({
        'task_id': task.task_id,
        'config': config_dict
    }, '训练任务已创建')


@api_bp.route('/train/config', methods=['GET'])
@handle_api_error
def get_train_config_options():
    """
    获取训练配置选项
    
    GET /api/train/config
    """
    options = {
        'model_types': [
            {'value': 'vits', 'label': 'VITS - 推荐（端到端，高质量）'},
            {'value': 'glow_tts', 'label': 'Glow-TTS - 流式模型'},
            {'value': 'fast_speech2', 'label': 'FastSpeech2 - 快速推理'},
            {'value': 'tacotron2', 'label': 'Tacotron2 - 经典模型'}
        ],
        'languages': [
            {'value': 'auto', 'label': '自动检测 (auto)'},
            {'value': 'en', 'label': '英语 (en)'},
            {'value': 'zh-cn', 'label': '中文 (zh-cn)'},
            {'value': 'ja', 'label': '日语 (ja)'},
            {'value': 'ko', 'label': '韩语 (ko)'}
        ],
        'epochs': {
            'default': 100,
            'min': 1,
            'max': 1000,
            'description': '训练轮数'
        },
        'batch_size': {
            'default': 32,
            'min': 1,
            'max': 256,
            'description': '批处理大小'
        },
        'sample_rate': {
            'default': 22050,
            'options': [8000, 16000, 22050, 44100, 48000],
            'description': '音频采样率'
        }
    }
    
    return success_response({'options': options})


@api_bp.route('/train/<task_id>/stop', methods=['POST'])
@handle_api_error
def stop_train_task(task_id: str):
    """
    停止训练任务
    
    POST /api/train/<task_id>/stop
    """
    task = task_manager.get_task(task_id)
    
    if not task:
        return error_response('TASK_NOT_FOUND', f'任务 {task_id} 不存在', status_code=404)
    
    if task.task_type != 'train':
        return error_response('INVALID_TASK_TYPE', '该任务不是训练任务', status_code=400)
    
    # 只能停止进行中的任务
    if task.status != TaskStatus.PROCESSING:
        return error_response(
            'CANNOT_STOP',
            f'无法停止状态为 {task.status.value} 的任务',
            status_code=400
        )
    
    # 注意：实际的训练停止逻辑需要在训练控制器中实现
    task_manager.update_task(
        task_id,
        status=TaskStatus.CANCELLED,
        message='训练已停止'
    )
    
    return success_response(message='训练已停止')
