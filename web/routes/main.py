"""
主页面路由

定义主页面和功能页面的路由
"""

from flask import render_template
from . import main_bp


@main_bp.route('/')
def index():
    """主页面 - 功能选择入口"""
    return render_template('index.html')


@main_bp.route('/whisper')
def whisper_page():
    """Whisper 字幕提取页面"""
    return render_template('whisper.html')


@main_bp.route('/train')
def train_page():
    """语音训练页面"""
    return render_template('train.html')
