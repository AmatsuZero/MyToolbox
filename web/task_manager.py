"""
任务管理器模块

管理异步任务的状态跟踪和进度更新
"""

import uuid
import time
import threading
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Callable
from enum import Enum


class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"         # 等待处理
    UPLOADING = "uploading"     # 上传中
    PROCESSING = "processing"   # 处理中
    COMPLETED = "completed"     # 已完成
    FAILED = "failed"          # 失败
    CANCELLED = "cancelled"     # 已取消


class TaskStage(Enum):
    """任务阶段枚举（用于 Whisper）"""
    UPLOAD = "upload"           # 文件上传
    EXTRACTING = "extracting"   # 音频提取
    TRANSCRIBING = "transcribing"  # 语音识别
    GENERATING = "generating"   # 生成字幕
    DOWNLOAD_READY = "download_ready"  # 下载就绪


@dataclass
class TaskInfo:
    """任务信息数据类"""
    task_id: str
    task_type: str  # 'whisper' or 'train'
    status: TaskStatus = TaskStatus.PENDING
    stage: Optional[str] = None
    progress: float = 0.0  # 0-100
    message: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    # 额外数据
    file_id: Optional[str] = None
    filename: Optional[str] = None
    output_files: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'task_id': self.task_id,
            'task_type': self.task_type,
            'status': self.status.value,
            'stage': self.stage,
            'progress': self.progress,
            'message': self.message,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'result': self.result,
            'error': self.error,
            'file_id': self.file_id,
            'filename': self.filename,
            'output_files': self.output_files
        }


class TaskManager:
    """
    任务管理器
    
    管理所有异步任务的状态，支持进度更新和状态查询
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._tasks: Dict[str, TaskInfo] = {}
        self._tasks_lock = threading.Lock()
        self._callbacks: Dict[str, Callable] = {}
        self._initialized = True
    
    def create_task(self, task_type: str, filename: Optional[str] = None) -> TaskInfo:
        """
        创建新任务
        
        Args:
            task_type: 任务类型 ('whisper' 或 'train')
            filename: 关联的文件名
            
        Returns:
            新创建的任务信息
        """
        task_id = str(uuid.uuid4())
        file_id = str(uuid.uuid4()) if filename else None
        
        task = TaskInfo(
            task_id=task_id,
            task_type=task_type,
            file_id=file_id,
            filename=filename
        )
        
        with self._tasks_lock:
            self._tasks[task_id] = task
        
        return task
    
    def get_task(self, task_id: str) -> Optional[TaskInfo]:
        """
        获取任务信息
        
        Args:
            task_id: 任务ID
            
        Returns:
            任务信息，不存在则返回 None
        """
        with self._tasks_lock:
            return self._tasks.get(task_id)
    
    def update_task(
        self,
        task_id: str,
        status: Optional[TaskStatus] = None,
        stage: Optional[str] = None,
        progress: Optional[float] = None,
        message: Optional[str] = None,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        output_files: Optional[Dict[str, str]] = None
    ) -> Optional[TaskInfo]:
        """
        更新任务状态
        
        Args:
            task_id: 任务ID
            status: 新状态
            stage: 新阶段
            progress: 进度 (0-100)
            message: 状态消息
            result: 结果数据
            error: 错误信息
            output_files: 输出文件映射
            
        Returns:
            更新后的任务信息
        """
        with self._tasks_lock:
            task = self._tasks.get(task_id)
            if not task:
                return None
            
            if status is not None:
                task.status = status
            if stage is not None:
                task.stage = stage
            if progress is not None:
                task.progress = min(100.0, max(0.0, progress))
            if message is not None:
                task.message = message
            if result is not None:
                task.result = result
            if error is not None:
                task.error = error
            if output_files is not None:
                task.output_files.update(output_files)
            
            task.updated_at = time.time()
            
            # 触发回调
            if task_id in self._callbacks:
                try:
                    self._callbacks[task_id](task)
                except Exception:
                    pass
            
            return task
    
    def delete_task(self, task_id: str) -> bool:
        """
        删除任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            是否成功删除
        """
        with self._tasks_lock:
            if task_id in self._tasks:
                del self._tasks[task_id]
                if task_id in self._callbacks:
                    del self._callbacks[task_id]
                return True
            return False
    
    def list_tasks(self, task_type: Optional[str] = None) -> list[TaskInfo]:
        """
        列出所有任务
        
        Args:
            task_type: 可选的任务类型过滤
            
        Returns:
            任务列表
        """
        with self._tasks_lock:
            tasks = list(self._tasks.values())
            if task_type:
                tasks = [t for t in tasks if t.task_type == task_type]
            return sorted(tasks, key=lambda t: t.created_at, reverse=True)
    
    def register_callback(self, task_id: str, callback: Callable[[TaskInfo], None]):
        """
        注册任务状态更新回调
        
        Args:
            task_id: 任务ID
            callback: 回调函数
        """
        with self._tasks_lock:
            self._callbacks[task_id] = callback
    
    def cleanup_old_tasks(self, max_age_seconds: int = 3600):
        """
        清理过期任务
        
        Args:
            max_age_seconds: 最大任务年龄（秒）
        """
        current_time = time.time()
        with self._tasks_lock:
            to_delete = []
            for task_id, task in self._tasks.items():
                # 只清理已完成或失败的任务
                if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
                    if current_time - task.updated_at > max_age_seconds:
                        to_delete.append(task_id)
            
            for task_id in to_delete:
                del self._tasks[task_id]
                if task_id in self._callbacks:
                    del self._callbacks[task_id]


# 全局任务管理器实例
task_manager = TaskManager()
