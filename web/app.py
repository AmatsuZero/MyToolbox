"""
Flask 应用工厂模块

实现 Flask 应用的创建和配置
"""

import os
import webbrowser
from threading import Timer
from flask import Flask
from flask_cors import CORS


def create_app(config: dict = None) -> Flask:
    """
    创建并配置 Flask 应用实例
    
    Args:
        config: 可选的配置字典，用于覆盖默认配置
        
    Returns:
        配置完成的 Flask 应用实例
    """
    # 获取 web 目录路径
    web_dir = os.path.dirname(os.path.abspath(__file__))
    
    app = Flask(
        __name__,
        static_folder=os.path.join(web_dir, 'static'),
        template_folder=os.path.join(web_dir, 'templates')
    )
    
    # 默认配置
    app.config.update({
        'SECRET_KEY': os.urandom(24).hex(),
        'MAX_CONTENT_LENGTH': 2 * 1024 * 1024 * 1024,  # 2GB 文件大小限制
        'UPLOAD_FOLDER': os.path.join(os.getcwd(), 'uploads'),
        'JSON_AS_ASCII': False,  # 支持中文 JSON 响应
    })
    
    # 应用自定义配置
    if config:
        app.config.update(config)
    
    # 确保上传目录存在
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # 配置 CORS
    CORS(app, resources={
        r"/api/*": {
            "origins": "*",
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"]
        }
    })
    
    # 注册蓝图
    _register_blueprints(app)
    
    # 注册错误处理器
    _register_error_handlers(app)
    
    return app


def _register_blueprints(app: Flask) -> None:
    """注册所有蓝图"""
    from .routes import main_bp, api_bp
    
    app.register_blueprint(main_bp)
    app.register_blueprint(api_bp, url_prefix='/api')
    
    # 注册 ONNX TTS 蓝图（可选模块）
    try:
        from modules.tts.onnx.web_api import onnx_tts_bp
        app.register_blueprint(onnx_tts_bp)
    except ImportError:
        pass  # ONNX TTS 模块未安装，跳过


def _register_error_handlers(app: Flask) -> None:
    """注册全局错误处理器"""
    from .routes import error_response
    
    @app.errorhandler(400)
    def bad_request(error):
        return error_response('BAD_REQUEST', str(error.description), status_code=400)
    
    @app.errorhandler(404)
    def not_found(error):
        return error_response('NOT_FOUND', '请求的资源不存在', status_code=404)
    
    @app.errorhandler(413)
    def request_entity_too_large(error):
        return error_response('FILE_TOO_LARGE', '上传的文件超过大小限制（最大 2GB）', status_code=413)
    
    @app.errorhandler(500)
    def internal_server_error(error):
        return error_response('INTERNAL_ERROR', '服务器内部错误', status_code=500)


def run_app(
    app: Flask,
    host: str = '127.0.0.1',
    port: int = 5000,
    open_browser: bool = True,
    initial_route: str = '/'
) -> None:
    """
    启动 Flask 应用
    
    Args:
        app: Flask 应用实例
        host: 绑定地址
        port: 绑定端口
        open_browser: 是否自动打开浏览器
        initial_route: 初始页面路由
    """
    url = f"http://{host if host != '0.0.0.0' else 'localhost'}:{port}{initial_route}"
    
    print(f"\n🚀 Web 服务已启动")
    print(f"📍 访问地址: {url}")
    print(f"💡 按 Ctrl+C 停止服务\n")
    
    if open_browser:
        # 延迟打开浏览器，确保服务已启动
        Timer(1.0, lambda: webbrowser.open(url)).start()
    
    app.run(host=host, port=port, debug=False, threaded=True)
