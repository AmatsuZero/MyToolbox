"""
ONNX TTS模块的自定义异常类
"""


class ONNXTTSError(Exception):
    """ONNX TTS模块的基础异常类"""
    pass


class ModelLoadError(ONNXTTSError):
    """模型加载失败异常"""
    pass


class ModelValidationError(ONNXTTSError):
    """模型验证失败异常"""
    pass


class TextProcessingError(ONNXTTSError):
    """文本处理失败异常"""
    pass


class InferenceError(ONNXTTSError):
    """推理执行失败异常"""
    pass


class AudioProcessingError(ONNXTTSError):
    """音频处理失败异常"""
    pass


class ConfigurationError(ONNXTTSError):
    """配置错误异常"""
    pass


class DependencyError(ONNXTTSError):
    """依赖缺失异常"""
    pass
