"""
ONNX Runtime TTS模块

该模块提供基于ONNX Runtime的语音合成推理功能。
作为可选依赖，需要安装onnxruntime才能使用。

安装方法：
    pip install mytoolbox[onnx]
"""

import sys
from typing import Optional

# 延迟导入检查
_ONNX_AVAILABLE = None
_IMPORT_ERROR = None


def check_onnx_available() -> tuple[bool, Optional[str]]:
    """
    检查ONNX Runtime是否可用
    
    Returns:
        tuple: (是否可用, 错误信息)
    """
    global _ONNX_AVAILABLE, _IMPORT_ERROR
    
    if _ONNX_AVAILABLE is not None:
        return _ONNX_AVAILABLE, _IMPORT_ERROR
    
    try:
        import onnxruntime
        import numpy
        import scipy
        _ONNX_AVAILABLE = True
        _IMPORT_ERROR = None
    except ImportError as e:
        _ONNX_AVAILABLE = False
        _IMPORT_ERROR = (
            f"ONNX Runtime依赖未安装: {str(e)}\n"
            "请运行以下命令安装:\n"
            "  pip install mytoolbox[onnx]\n"
            "或手动安装:\n"
            "  pip install onnxruntime>=1.16.0 numpy>=1.24.0 scipy>=1.11.0"
        )
    
    return _ONNX_AVAILABLE, _IMPORT_ERROR


def require_onnx():
    """
    确保ONNX Runtime可用，否则抛出友好的错误信息
    
    Raises:
        ImportError: 当ONNX Runtime不可用时
    """
    available, error = check_onnx_available()
    if not available:
        raise ImportError(error)


# 延迟导入模块
def __getattr__(name):
    """延迟导入模块成员"""
    require_onnx()
    
    # 导入实际模块
    if name == "ONNXModelLoader":
        from .model_loader import ONNXModelLoader
        return ONNXModelLoader
    elif name == "ONNXInferenceEngine":
        from .inference_engine import ONNXInferenceEngine
        return ONNXInferenceEngine
    elif name == "TextProcessor":
        from .text_processor import TextProcessor
        return TextProcessor
    elif name == "AudioProcessor":
        from .audio_processor import AudioProcessor
        return AudioProcessor
    
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "check_onnx_available",
    "require_onnx",
    "ONNXModelLoader",
    "ONNXInferenceEngine",
    "TextProcessor",
    "AudioProcessor",
]
