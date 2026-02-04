"""
文件上传 API 路由

处理文件上传相关的 API 请求
"""

import os
import uuid
import shutil
from pathlib import Path
from typing import Set
from flask import request, current_app, send_file

from . import api_bp, success_response, error_response, handle_api_error
from ..task_manager import task_manager


# 支持的音视频格式
SUPPORTED_AUDIO_FORMATS: Set[str] = {
    '.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a', '.wma'
}

SUPPORTED_VIDEO_FORMATS: Set[str] = {
    '.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm', '.m4v'
}

SUPPORTED_SUBTITLE_FORMATS: Set[str] = {
    '.srt', '.vtt', '.ass', '.txt'
}

ALL_SUPPORTED_FORMATS = SUPPORTED_AUDIO_FORMATS | SUPPORTED_VIDEO_FORMATS | SUPPORTED_SUBTITLE_FORMATS


def get_upload_folder() -> Path:
    """获取上传目录"""
    upload_folder = current_app.config.get('UPLOAD_FOLDER', 'uploads')
    path = Path(upload_folder)
    path.mkdir(parents=True, exist_ok=True)
    return path


def validate_file_extension(filename: str, allowed_formats: Set[str] = None) -> bool:
    """
    验证文件扩展名
    
    Args:
        filename: 文件名
        allowed_formats: 允许的格式集合，默认为所有支持的格式
        
    Returns:
        是否为支持的格式
    """
    if allowed_formats is None:
        allowed_formats = ALL_SUPPORTED_FORMATS
    
    ext = Path(filename).suffix.lower()
    return ext in allowed_formats


def get_file_type(filename: str) -> str:
    """
    获取文件类型
    
    Args:
        filename: 文件名
        
    Returns:
        文件类型 ('audio', 'video', 'subtitle', 'unknown')
    """
    ext = Path(filename).suffix.lower()
    
    if ext in SUPPORTED_AUDIO_FORMATS:
        return 'audio'
    elif ext in SUPPORTED_VIDEO_FORMATS:
        return 'video'
    elif ext in SUPPORTED_SUBTITLE_FORMATS:
        return 'subtitle'
    else:
        return 'unknown'


@api_bp.route('/upload', methods=['POST'])
@handle_api_error
def upload_file():
    """
    文件上传接口
    
    POST /api/upload
    
    表单数据:
        file: 上传的文件
        purpose: 用途 ('whisper' 或 'train')
        
    返回:
        {
            "success": true,
            "data": {
                "file_id": "uuid",
                "filename": "原始文件名",
                "size": 文件大小(字节),
                "type": "audio/video/subtitle",
                "path": "服务器保存路径"
            }
        }
    """
    # 检查是否有文件
    if 'file' not in request.files:
        return error_response('NO_FILE', '未提供文件', status_code=400)
    
    file = request.files['file']
    
    # 检查文件名
    if file.filename == '' or file.filename is None:
        return error_response('NO_FILENAME', '文件名为空', status_code=400)
    
    filename = file.filename
    purpose = request.form.get('purpose', 'whisper')
    
    # 验证文件格式
    if purpose == 'whisper':
        allowed_formats = SUPPORTED_AUDIO_FORMATS | SUPPORTED_VIDEO_FORMATS
    elif purpose == 'train':
        allowed_formats = SUPPORTED_AUDIO_FORMATS | SUPPORTED_SUBTITLE_FORMATS
    else:
        allowed_formats = ALL_SUPPORTED_FORMATS
    
    if not validate_file_extension(filename, allowed_formats):
        return error_response(
            'INVALID_FORMAT',
            f'不支持的文件格式，允许的格式: {", ".join(sorted(allowed_formats))}',
            status_code=400
        )
    
    # 生成唯一文件ID和保存路径
    file_id = str(uuid.uuid4())
    ext = Path(filename).suffix.lower()
    safe_filename = f"{file_id}{ext}"
    
    upload_folder = get_upload_folder()
    save_path = upload_folder / safe_filename
    
    # 保存文件
    try:
        file.save(str(save_path))
    except Exception as e:
        return error_response('SAVE_ERROR', f'文件保存失败: {str(e)}', status_code=500)
    
    # 获取文件信息
    file_size = save_path.stat().st_size
    file_type = get_file_type(filename)
    
    # 检查文件大小限制
    max_size = current_app.config.get('MAX_CONTENT_LENGTH', 2 * 1024 * 1024 * 1024)
    if file_size > max_size:
        # 删除超大文件
        save_path.unlink(missing_ok=True)
        return error_response(
            'FILE_TOO_LARGE',
            f'文件大小超过限制 ({max_size / (1024*1024*1024):.1f}GB)',
            status_code=413
        )
    
    return success_response({
        'file_id': file_id,
        'filename': filename,
        'size': file_size,
        'type': file_type,
        'path': str(save_path)
    }, '文件上传成功')


@api_bp.route('/upload/batch', methods=['POST'])
@handle_api_error
def upload_batch():
    """
    批量文件上传接口
    
    POST /api/upload/batch
    
    表单数据:
        files: 多个上传的文件
        purpose: 用途 ('whisper' 或 'train')
        
    返回:
        {
            "success": true,
            "data": {
                "uploaded": [文件信息列表],
                "failed": [失败的文件列表]
            }
        }
    """
    if 'files' not in request.files:
        return error_response('NO_FILES', '未提供文件', status_code=400)
    
    files = request.files.getlist('files')
    purpose = request.form.get('purpose', 'whisper')
    
    if not files:
        return error_response('NO_FILES', '文件列表为空', status_code=400)
    
    # 确定允许的格式
    if purpose == 'whisper':
        allowed_formats = SUPPORTED_AUDIO_FORMATS | SUPPORTED_VIDEO_FORMATS
    elif purpose == 'train':
        allowed_formats = SUPPORTED_AUDIO_FORMATS | SUPPORTED_SUBTITLE_FORMATS
    else:
        allowed_formats = ALL_SUPPORTED_FORMATS
    
    uploaded = []
    failed = []
    upload_folder = get_upload_folder()
    
    for file in files:
        if file.filename == '' or file.filename is None:
            continue
        
        filename = file.filename
        
        # 验证格式
        if not validate_file_extension(filename, allowed_formats):
            failed.append({
                'filename': filename,
                'error': '不支持的文件格式'
            })
            continue
        
        # 保存文件
        try:
            file_id = str(uuid.uuid4())
            ext = Path(filename).suffix.lower()
            safe_filename = f"{file_id}{ext}"
            save_path = upload_folder / safe_filename
            
            file.save(str(save_path))
            
            file_size = save_path.stat().st_size
            file_type = get_file_type(filename)
            
            uploaded.append({
                'file_id': file_id,
                'filename': filename,
                'size': file_size,
                'type': file_type,
                'path': str(save_path)
            })
        except Exception as e:
            failed.append({
                'filename': filename,
                'error': str(e)
            })
    
    return success_response({
        'uploaded': uploaded,
        'failed': failed,
        'total': len(files),
        'success_count': len(uploaded),
        'fail_count': len(failed)
    }, f'上传完成: {len(uploaded)} 成功, {len(failed)} 失败')


@api_bp.route('/upload/<file_id>', methods=['DELETE'])
@handle_api_error
def delete_upload(file_id: str):
    """
    删除已上传的文件
    
    DELETE /api/upload/<file_id>
    """
    upload_folder = get_upload_folder()
    
    # 查找文件
    for file_path in upload_folder.iterdir():
        if file_path.stem == file_id:
            file_path.unlink()
            return success_response(message='文件已删除')
    
    return error_response('NOT_FOUND', '文件不存在', status_code=404)


@api_bp.route('/download/<file_id>', methods=['GET'])
@handle_api_error
def download_file(file_id: str):
    """
    下载文件
    
    GET /api/download/<file_id>
    
    查询参数:
        type: 文件类型 ('output' 表示输出文件, 默认为上传文件)
    """
    file_type = request.args.get('type', 'upload')
    
    if file_type == 'upload':
        base_folder = get_upload_folder()
    else:
        # 输出文件目录
        base_folder = Path(current_app.config.get('OUTPUT_FOLDER', 'output'))
    
    # 查找文件
    for file_path in base_folder.iterdir():
        if file_path.stem == file_id or file_path.name.startswith(file_id):
            if file_path.is_file():
                return send_file(
                    str(file_path),
                    as_attachment=True,
                    download_name=file_path.name
                )
    
    return error_response('NOT_FOUND', '文件不存在', status_code=404)


@api_bp.route('/files', methods=['GET'])
@handle_api_error
def list_files():
    """
    列出已上传的文件
    
    GET /api/files
    """
    upload_folder = get_upload_folder()
    
    files = []
    for file_path in upload_folder.iterdir():
        if file_path.is_file():
            files.append({
                'file_id': file_path.stem,
                'filename': file_path.name,
                'size': file_path.stat().st_size,
                'type': get_file_type(file_path.name),
                'modified': file_path.stat().st_mtime
            })
    
    # 按修改时间排序
    files.sort(key=lambda x: x['modified'], reverse=True)
    
    return success_response({
        'files': files,
        'total': len(files)
    })
