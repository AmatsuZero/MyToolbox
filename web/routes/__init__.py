"""
路由模块

定义所有 Web 路由和 API 蓝图
"""

from flask import Blueprint, jsonify
from typing import Any, Optional
from functools import wraps

# 创建蓝图
main_bp = Blueprint('main', __name__)
api_bp = Blueprint('api', __name__)


def success_response(data: Any = None, message: str = '操作成功') -> dict:
    """
    创建成功响应
    
    Args:
        data: 响应数据
        message: 成功消息
        
    Returns:
        JSON 响应字典
    """
    response = {
        'success': True,
        'message': message,
        'data': data
    }
    return jsonify(response)


def error_response(code: str, message: str, details: Optional[dict] = None, status_code: int = 400) -> tuple:
    """
    创建错误响应
    
    Args:
        code: 错误代码
        message: 错误消息
        details: 额外的错误详情
        status_code: HTTP 状态码
        
    Returns:
        JSON 响应元组 (response, status_code)
    """
    response = {
        'success': False,
        'error': {
            'code': code,
            'message': message
        }
    }
    if details:
        response['error']['details'] = details
    return jsonify(response), status_code


def handle_api_error(f):
    """
    API 错误处理装饰器
    
    自动捕获异常并返回结构化错误响应
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except ValueError as e:
            return error_response('VALIDATION_ERROR', str(e), status_code=400)
        except FileNotFoundError as e:
            return error_response('NOT_FOUND', str(e), status_code=404)
        except PermissionError as e:
            return error_response('PERMISSION_DENIED', str(e), status_code=403)
        except Exception as e:
            return error_response('INTERNAL_ERROR', f'服务器内部错误: {str(e)}', status_code=500)
    return decorated_function


# 导入并注册子路由模块
from . import main
from . import upload
from . import whisper
from . import train


__all__ = [
    'main_bp',
    'api_bp',
    'success_response',
    'error_response',
    'handle_api_error'
]
