"""
ONNX TTS Web API

提供Flask路由和API端点，支持模型上传、文本合成、进度查询和音频下载
"""

import os
import uuid
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from flask import Blueprint, request, jsonify, send_file
from werkzeug.utils import secure_filename

from .model_loader import ONNXModelLoader
from .inference_engine import ONNXInferenceEngine, InferenceProgress
from .text_processor import TextProcessor
from .audio_processor import AudioProcessor
from .exceptions import ONNXTTSError
from .logger import logger

# 创建Blueprint
onnx_tts_bp = Blueprint('onnx_tts', __name__, url_prefix='/api/onnx-tts')

# 全局状态管理
class SessionManager:
    """会话管理器"""
    
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.upload_dir = Path("uploads/onnx_models")
        self.output_dir = Path("output/onnx_tts")
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def create_session(self) -> str:
        """创建新会话"""
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "created_at": datetime.now().isoformat(),
            "model_path": None,
            "model_loader": None,
            "progress": None,
            "output_file": None,
            "error": None,
        }
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """获取会话"""
        return self.sessions.get(session_id)
    
    def update_session(self, session_id: str, **kwargs):
        """更新会话"""
        if session_id in self.sessions:
            self.sessions[session_id].update(kwargs)
    
    def cleanup_old_sessions(self, max_age_hours: int = 24):
        """清理旧会话"""
        # 简化实现：实际应用中应该定期清理
        pass

# 全局会话管理器
session_manager = SessionManager()


@onnx_tts_bp.route('/health', methods=['GET'])
def health_check():
    """健康检查"""
    try:
        from . import check_onnx_available
        available, error = check_onnx_available()
        
        return jsonify({
            "status": "ok" if available else "error",
            "onnx_available": available,
            "error": error if not available else None
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


@onnx_tts_bp.route('/session', methods=['POST'])
def create_session():
    """创建新会话"""
    try:
        session_id = session_manager.create_session()
        return jsonify({
            "success": True,
            "session_id": session_id
        })
    except Exception as e:
        logger.error(f"创建会话失败: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@onnx_tts_bp.route('/upload-model', methods=['POST'])
def upload_model():
    """上传ONNX模型文件"""
    try:
        # 获取会话ID
        session_id = request.form.get('session_id')
        if not session_id:
            return jsonify({
                "success": False,
                "error": "缺少session_id参数"
            }), 400
        
        session = session_manager.get_session(session_id)
        if not session:
            return jsonify({
                "success": False,
                "error": "无效的session_id"
            }), 404
        
        # 检查文件
        if 'model_file' not in request.files:
            return jsonify({
                "success": False,
                "error": "未找到模型文件"
            }), 400
        
        file = request.files['model_file']
        if file.filename == '':
            return jsonify({
                "success": False,
                "error": "未选择文件"
            }), 400
        
        # 验证文件扩展名
        if not file.filename.endswith('.onnx'):
            return jsonify({
                "success": False,
                "error": "只支持.onnx格式的模型文件"
            }), 400
        
        # 保存文件
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_filename = f"{timestamp}_{filename}"
        model_path = session_manager.upload_dir / unique_filename
        file.save(str(model_path))
        
        logger.info(f"模型文件已上传: {model_path}")
        
        # 加载模型
        device = request.form.get('device', 'auto')
        model_loader = ONNXModelLoader(
            model_path=str(model_path),
            device=device
        )
        
        # 更新会话
        session_manager.update_session(
            session_id,
            model_path=str(model_path),
            model_loader=model_loader
        )
        
        # 返回模型信息
        return jsonify({
            "success": True,
            "model_info": {
                "filename": filename,
                "size": model_path.stat().st_size,
                "device": model_loader.device,
                "inputs": list(model_loader.input_metadata.keys()),
                "outputs": list(model_loader.output_metadata.keys())
            }
        })
        
    except ONNXTTSError as e:
        logger.error(f"模型上传失败: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400
    except Exception as e:
        logger.error(f"模型上传失败: {str(e)}")
        return jsonify({
            "success": False,
            "error": f"服务器错误: {str(e)}"
        }), 500


@onnx_tts_bp.route('/synthesize', methods=['POST'])
def synthesize():
    """执行文本合成"""
    try:
        data = request.get_json()
        
        # 验证参数
        session_id = data.get('session_id')
        text = data.get('text')
        
        if not session_id:
            return jsonify({
                "success": False,
                "error": "缺少session_id参数"
            }), 400
        
        if not text:
            return jsonify({
                "success": False,
                "error": "缺少text参数"
            }), 400
        
        session = session_manager.get_session(session_id)
        if not session:
            return jsonify({
                "success": False,
                "error": "无效的session_id"
            }), 404
        
        if not session.get('model_loader'):
            return jsonify({
                "success": False,
                "error": "请先上传模型"
            }), 400
        
        # 获取参数
        input_format = data.get('input_format', 'text')
        output_type = data.get('output_type', 'waveform')
        output_format = data.get('output_format', 'wav')
        sample_rate = data.get('sample_rate', 22050)
        
        # 进度回调
        def progress_callback(progress: InferenceProgress):
            session_manager.update_session(
                session_id,
                progress={
                    "stage": progress.stage,
                    "progress": progress.progress,
                    "message": progress.message
                }
            )
        
        # 初始化组件
        text_processor = TextProcessor()
        inference_engine = ONNXInferenceEngine(
            model_loader=session['model_loader'],
            text_processor=text_processor,
            progress_callback=progress_callback
        )
        audio_processor = AudioProcessor(sample_rate=sample_rate)
        
        # 执行推理
        output_dict = inference_engine.infer_text(
            text=text,
            input_format=input_format
        )
        
        # 处理音频
        waveform = audio_processor.process_model_output(
            output_dict=output_dict,
            output_type=output_type
        )
        
        # 保存音频
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"tts_{timestamp}.{output_format}"
        output_path = session_manager.output_dir / output_filename
        
        saved_path = audio_processor.save_audio(
            waveform=waveform,
            output_path=output_path,
            format=output_format
        )
        
        # 更新会话
        session_manager.update_session(
            session_id,
            output_file=str(saved_path),
            progress={
                "stage": "complete",
                "progress": 100,
                "message": "合成完成"
            }
        )
        
        # 返回结果
        return jsonify({
            "success": True,
            "output_file": output_filename,
            "file_size": saved_path.stat().st_size,
            "duration": len(waveform) / sample_rate
        })
        
    except ONNXTTSError as e:
        logger.error(f"合成失败: {str(e)}")
        if session_id:
            session_manager.update_session(
                session_id,
                error=str(e)
            )
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400
    except Exception as e:
        logger.error(f"合成失败: {str(e)}")
        if session_id:
            session_manager.update_session(
                session_id,
                error=str(e)
            )
        return jsonify({
            "success": False,
            "error": f"服务器错误: {str(e)}"
        }), 500


@onnx_tts_bp.route('/progress/<session_id>', methods=['GET'])
def get_progress(session_id: str):
    """获取合成进度"""
    try:
        session = session_manager.get_session(session_id)
        if not session:
            return jsonify({
                "success": False,
                "error": "无效的session_id"
            }), 404
        
        return jsonify({
            "success": True,
            "progress": session.get('progress'),
            "error": session.get('error')
        })
        
    except Exception as e:
        logger.error(f"获取进度失败: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@onnx_tts_bp.route('/download/<filename>', methods=['GET'])
def download_audio(filename: str):
    """下载生成的音频文件"""
    try:
        # 安全检查
        filename = secure_filename(filename)
        file_path = session_manager.output_dir / filename
        
        if not file_path.exists():
            return jsonify({
                "success": False,
                "error": "文件不存在"
            }), 404
        
        return send_file(
            str(file_path),
            as_attachment=True,
            download_name=filename
        )
        
    except Exception as e:
        logger.error(f"下载失败: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


def register_onnx_tts_routes(app):
    """
    注册ONNX TTS路由到Flask应用
    
    Args:
        app: Flask应用实例
    """
    app.register_blueprint(onnx_tts_bp)
    logger.info("ONNX TTS API路由已注册")
