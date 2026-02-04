"""
ONNX TTS模块的日志和错误处理工具
"""

import logging
import functools
import traceback
from typing import Callable, Any
from .exceptions import ONNXTTSError

# 配置日志
logger = logging.getLogger("mytoolbox.onnx_tts")


def setup_logger(level=logging.INFO):
    """
    配置日志记录器
    
    Args:
        level: 日志级别
    """
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)


def handle_errors(error_type: type = ONNXTTSError, 
                  message: str = "操作失败",
                  log_traceback: bool = True):
    """
    错误处理装饰器
    
    Args:
        error_type: 要捕获并转换的异常类型
        message: 错误消息前缀
        log_traceback: 是否记录完整的堆栈跟踪
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except error_type:
                # 已经是我们的自定义异常，直接重新抛出
                raise
            except Exception as e:
                # 记录详细错误信息
                if log_traceback:
                    logger.error(f"{message}: {str(e)}")
                    logger.debug(traceback.format_exc())
                else:
                    logger.error(f"{message}: {str(e)}")
                
                # 转换为自定义异常
                raise error_type(f"{message}: {str(e)}") from e
        
        return wrapper
    return decorator


def log_operation(operation_name: str):
    """
    操作日志装饰器
    
    Args:
        operation_name: 操作名称
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            logger.info(f"开始{operation_name}...")
            try:
                result = func(*args, **kwargs)
                logger.info(f"{operation_name}完成")
                return result
            except Exception as e:
                logger.error(f"{operation_name}失败: {str(e)}")
                raise
        
        return wrapper
    return decorator


def format_error_message(error: Exception, context: str = "") -> str:
    """
    格式化错误消息，提供用户友好的提示
    
    Args:
        error: 异常对象
        context: 错误上下文信息
        
    Returns:
        格式化的错误消息
    """
    error_msg = str(error)
    
    # 根据错误类型提供具体建议
    suggestions = []
    
    if "No such file or directory" in error_msg or "FileNotFoundError" in str(type(error)):
        suggestions.append("请检查文件路径是否正确")
        suggestions.append("确保文件存在且有读取权限")
    
    elif "onnxruntime" in error_msg.lower() or "onnx" in error_msg.lower():
        suggestions.append("请确保已安装ONNX Runtime: pip install mytoolbox[onnx]")
        suggestions.append("检查ONNX模型文件格式是否正确")
    
    elif "memory" in error_msg.lower() or "MemoryError" in str(type(error)):
        suggestions.append("系统内存不足，请尝试使用更小的模型")
        suggestions.append("或减少批处理大小")
    
    elif "timeout" in error_msg.lower():
        suggestions.append("操作超时，请检查模型复杂度")
        suggestions.append("或考虑使用性能更好的硬件")
    
    elif "cuda" in error_msg.lower() or "gpu" in error_msg.lower():
        suggestions.append("GPU相关错误，请检查CUDA是否正确安装")
        suggestions.append("或尝试使用CPU模式")
    
    # 构建完整的错误消息
    full_message = f"错误: {error_msg}"
    if context:
        full_message = f"{context} - {full_message}"
    
    if suggestions:
        full_message += "\n\n建议:\n" + "\n".join(f"  • {s}" for s in suggestions)
    
    return full_message


# 初始化日志
setup_logger()
