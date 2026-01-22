"""
错误处理模块 - 提供错误日志记录、异常处理和性能监控功能

该模块包含用于处理应用程序错误、记录异常信息和监控系统性能的类和函数。
主要功能包括：
- 错误日志记录和分类
- 异常捕获和处理
- 自动重试机制
- 系统性能监控
- 内存优化
"""

import gc
import json
import logging
import signal
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import psutil


class ErrorLevel(Enum):
    """错误级别枚举"""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ErrorContext:
    """错误上下文信息"""

    file_path: Optional[str] = None
    file_size: Optional[int] = None
    processing_time: Optional[float] = None
    retry_count: int = 0


@dataclass
class SystemState:
    """系统状态信息"""

    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None


@dataclass
class LocationInfo:
    """错误位置信息"""

    module: str
    function: str


@dataclass
class ExceptionInfo:
    """异常信息"""

    type: Optional[str] = None
    message: Optional[str] = None
    traceback: Optional[str] = None


@dataclass
class ErrorInfo:
    """错误信息类，将相关属性分组到嵌套数据类中，减少顶层实例属性数量"""

    timestamp: float
    level: ErrorLevel
    location: LocationInfo
    message: str
    exception: ExceptionInfo = field(default_factory=ExceptionInfo)
    context: ErrorContext = field(default_factory=ErrorContext)
    system: SystemState = field(default_factory=SystemState)

    def to_dict(self) -> Dict:
        """转换为字典"""
        result = {}

        # 添加基本属性
        result["timestamp"] = self.timestamp
        result["level"] = self.level.value

        # 展开location属性
        result["module"] = self.location.module
        result["function"] = self.location.function

        result["message"] = self.message

        # 展开exception属性
        result["exception_type"] = self.exception.type
        result["exception_message"] = self.exception.message
        result["traceback"] = self.exception.traceback

        # 展开context和system属性
        context_dict = asdict(self.context)
        system_dict = asdict(self.system)
        result.update(context_dict)
        result.update(system_dict)

        return result


@dataclass
class SystemResources:
    """系统资源使用情况"""

    memory_usage_mb: float
    cpu_usage_percent: float
    gpu_memory_mb: Optional[float] = None
    disk_usage_percent: float = 0.0
    network_usage_mb: float = 0.0
    active_threads: int = 0
    open_files: int = 0


@dataclass
class ProcessingStats:
    """处理统计信息"""

    processing_files: int = 0
    completed_files: int = 0
    failed_files: int = 0
    average_processing_time: float = 0.0
    total_processing_time: float = 0.0


@dataclass
class PerformanceMetrics:
    """性能指标类"""

    timestamp: float
    system_resources: SystemResources
    processing_stats: ProcessingStats

    def to_dict(self) -> Dict:
        """转换为字典"""
        result = {
            "timestamp": self.timestamp,
            "memory_usage_mb": self.system_resources.memory_usage_mb,
            "cpu_usage_percent": self.system_resources.cpu_usage_percent,
            "gpu_memory_mb": self.system_resources.gpu_memory_mb,
            "disk_usage_percent": self.system_resources.disk_usage_percent,
            "network_usage_mb": self.system_resources.network_usage_mb,
            "active_threads": self.system_resources.active_threads,
            "open_files": self.system_resources.open_files,
            "processing_files": self.processing_stats.processing_files,
            "completed_files": self.processing_stats.completed_files,
            "failed_files": self.processing_stats.failed_files,
            "average_processing_time": self.processing_stats.average_processing_time,
            "total_processing_time": self.processing_stats.total_processing_time,
        }
        return result


@dataclass
class RetryStrategyConfig:
    """重试策略配置"""

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_backoff: bool = True
    retry_on_errors: Optional[List[str]] = None

    def __post_init__(self):
        """初始化后处理"""
        if self.retry_on_errors is None:
            self.retry_on_errors = [
                "ConnectionError",
                "TimeoutError",
                "IOError",
                "MemoryError",
                "ResourceWarning",
            ]


class RetryStrategy:
    """重试策略类"""

    def __init__(self, config: Optional[RetryStrategyConfig] = None):
        """
        初始化重试策略

        Args:
            config: 重试策略配置，如果为None则使用默认配置
        """
        self.config = config or RetryStrategyConfig()
        self.max_retries = self.config.max_retries
        self.base_delay = self.config.base_delay
        self.max_delay = self.config.max_delay
        self.exponential_backoff = self.config.exponential_backoff
        self.retry_on_errors = self.config.retry_on_errors

    def should_retry(self, error: Exception) -> bool:
        """判断是否应该重试"""
        error_type = type(error).__name__
        # 确保 retry_on_errors 不为 None，虽然在 RetryStrategyConfig.__post_init__ 中已处理
        # 但为了类型安全，这里再次确认
        retry_on_errors = self.retry_on_errors or []
        return error_type in retry_on_errors

    def get_delay(self, attempt: int) -> float:
        """计算重试延迟时间"""
        if self.exponential_backoff:
            delay = self.base_delay * (2 ** (attempt - 1))
        else:
            delay = self.base_delay * attempt

        return min(delay, self.max_delay)


@dataclass
class ErrorHandlerConfig:
    """错误处理器配置"""

    log_file: str = "error_log.json"
    max_log_size_mb: int = 100
    enable_performance_monitoring: bool = True


class ErrorHandler:
    """错误处理器"""

    def __init__(
        self,
        log_file: str = "error_log.json",
        max_log_size_mb: int = 100,
        enable_performance_monitoring: bool = True,
    ):
        self.config = ErrorHandlerConfig(
            log_file=log_file,
            max_log_size_mb=max_log_size_mb,
            enable_performance_monitoring=enable_performance_monitoring,
        )
        self.error_log: List[ErrorInfo] = []
        self.performance_log: List[PerformanceMetrics] = []
        self.retry_strategy = RetryStrategy()
        self.logger = self._setup_logger()
        self.monitoring_thread = None
        self.monitoring_active = False
        self._setup_signal_handlers()

    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger("ErrorHandler")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            stream_handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)

        return logger

    def _setup_signal_handlers(self):
        """设置信号处理器"""

        def signal_handler(signum, _):
            self.logger.info("收到信号 %s，正在优雅关闭...", signum)
            self.stop_performance_monitoring()
            self.save_logs()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    @dataclass
    class LogErrorParams:
        """日志错误参数数据类，用于减少函数参数数量"""

        level: ErrorLevel
        module: str
        function: str
        message: str
        exception: Optional[Exception] = None
        # 将文件相关属性组合到一个上下文对象中，减少实例属性数量
        context: ErrorContext = field(default_factory=ErrorContext)

    def log_error(
        self,
        error_params: LogErrorParams,
    ) -> ErrorInfo:
        """记录错误信息

        Args:
            error_params: 包含所有错误信息的参数对象

        Returns:
            ErrorInfo: 创建的错误信息对象
        """

        # 使用参数中的上下文对象
        error_context = error_params.context

        system_state = SystemState(
            memory_usage_mb=psutil.virtual_memory().used / (1024 * 1024),
            cpu_usage_percent=psutil.cpu_percent(),
        )

        # 创建位置信息对象
        location_info = LocationInfo(
            module=error_params.module, function=error_params.function
        )

        # 创建异常信息对象
        exception_info = ExceptionInfo(
            type=(
                type(error_params.exception).__name__
                if error_params.exception
                else None
            ),
            message=str(error_params.exception) if error_params.exception else None,
            traceback=traceback.format_exc() if error_params.exception else None,
        )

        error_info = ErrorInfo(
            timestamp=time.time(),
            level=error_params.level,
            location=location_info,
            message=error_params.message,
            exception=exception_info,
            context=error_context,
            system=system_state,
        )

        self.error_log.append(error_info)

        # 根据级别记录日志
        if error_params.level == ErrorLevel.INFO:
            self.logger.info(
                "%s.%s: %s",
                error_params.module,
                error_params.function,
                error_params.message,
            )
        elif error_params.level == ErrorLevel.WARNING:
            self.logger.warning(
                "%s.%s: %s",
                error_params.module,
                error_params.function,
                error_params.message,
            )
        elif error_params.level == ErrorLevel.ERROR:
            self.logger.error(
                "%s.%s: %s",
                error_params.module,
                error_params.function,
                error_params.message,
            )
        elif error_params.level == ErrorLevel.CRITICAL:
            self.logger.critical(
                "%s.%s: %s",
                error_params.module,
                error_params.function,
                error_params.message,
            )

        # 如果日志文件过大，清理旧日志
        self._cleanup_old_logs()

        return error_info

    def handle_exception(
        self,
        module: str,
        function: str,
        exception: Exception,
        **kwargs
    ) -> ErrorInfo:
        """处理异常
        
        Args:
            module: 模块名称
            function: 函数名称
            exception: 异常对象
            **kwargs: 可选参数，可包含：
                file_path: 文件路径
                file_size: 文件大小
                processing_time: 处理时间
                retry_count: 重试次数
                
        Returns:
            ErrorInfo: 错误信息对象
        """

        # 判断异常级别
        if isinstance(exception, (MemoryError, OSError, IOError)):
            level = ErrorLevel.CRITICAL
        elif isinstance(exception, (TimeoutError, ConnectionError)):
            level = ErrorLevel.ERROR
        else:
            level = ErrorLevel.WARNING

        # 从kwargs中提取错误上下文参数
        error_context = ErrorContext(
            file_path=kwargs.get('file_path'),
            file_size=kwargs.get('file_size'),
            processing_time=kwargs.get('processing_time'),
            retry_count=kwargs.get('retry_count', 0),
        )

        # 创建日志错误参数对象
        error_params = self.LogErrorParams(
            level=level,
            module=module,
            function=function,
            message=f"处理过程中发生异常: {str(exception)}",
            exception=exception,
            context=error_context,
        )

        return self.log_error(error_params=error_params)

    def retry_with_backoff(
        self, func: Callable, *args, max_retries: Optional[int] = None, **kwargs
    ) -> Any:
        """使用指数退避策略重试函数"""

        max_retries = max_retries or self.retry_strategy.max_retries

        for attempt in range(1, max_retries + 1):
            try:
                result = func(*args, **kwargs)
                return result

            except (
                IOError,
                OSError,
                TimeoutError,
                ConnectionError,
                MemoryError,
                ValueError,
                RuntimeError,
            ) as e:
                if attempt == max_retries or not self.retry_strategy.should_retry(e):
                    # 最后一次尝试或不应该重试的错误
                    self.handle_exception(
                        module=func.__module__,
                        function=func.__name__,
                        exception=e,
                        retry_count=attempt,
                    )
                    raise

                # 创建错误上下文对象
                error_context = ErrorContext(
                    retry_count=attempt,
                )

                # 记录重试信息
                error_params = self.LogErrorParams(
                    level=ErrorLevel.WARNING,
                    module=func.__module__,
                    function=func.__name__,
                    message=f"第 {attempt} 次重试，错误: {str(e)}",
                    exception=e,
                    context=error_context,
                )
                self.log_error(error_params=error_params)

                # 计算延迟时间
                delay = self.retry_strategy.get_delay(attempt)
                self.logger.info("等待 %.2f 秒后重试...", delay)
                time.sleep(delay)

        # 如果所有重试都失败但没有抛出异常，返回 None
        return None

    def start_performance_monitoring(self, interval: float = 5.0):
        """开始性能监控"""

        if not self.config.enable_performance_monitoring:
            return

        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitor_performance, args=(interval,), daemon=True
        )
        self.monitoring_thread.start()
        self.logger.info("性能监控已启动，间隔: %s秒", interval)

    def stop_performance_monitoring(self):
        """停止性能监控"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        self.logger.info("性能监控已停止")

    def _monitor_performance(self, interval: float):
        """性能监控线程函数"""

        while self.monitoring_active:
            try:
                metrics = self._collect_performance_metrics()
                self.performance_log.append(metrics)

                # 检查资源使用情况
                self._check_resource_usage(metrics)

                time.sleep(interval)

            except (
                IOError,
                OSError,
                MemoryError,
                ValueError,
                AttributeError,
                RuntimeError,
            ) as e:
                self.logger.error("性能监控错误: %s", str(e))
                time.sleep(interval)

    def _collect_performance_metrics(self) -> PerformanceMetrics:
        """收集性能指标"""

        memory = psutil.virtual_memory()
        cpu = psutil.cpu_percent()
        disk = psutil.disk_usage("/")

        # 获取网络使用情况
        net_io = psutil.net_io_counters()
        network_usage = (net_io.bytes_sent + net_io.bytes_recv) / (1024 * 1024)

        # 获取GPU内存使用情况（如果可用）
        gpu_memory = None
        try:
            import torch  # pylint: disable=import-outside-toplevel

            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)
        except ImportError:
            pass

        # 创建系统资源对象
        system_resources = SystemResources(
            memory_usage_mb=memory.used / (1024 * 1024),
            cpu_usage_percent=cpu,
            gpu_memory_mb=gpu_memory,
            disk_usage_percent=disk.percent,
            network_usage_mb=network_usage,
            active_threads=threading.active_count(),
            open_files=len(psutil.Process().open_files()),
        )

        # 创建处理统计对象 - 这里我们保留空值，因为原代码中也没有设置这些值
        processing_stats = ProcessingStats()

        return PerformanceMetrics(
            timestamp=time.time(),
            system_resources=system_resources,
            processing_stats=processing_stats,
        )

    def _check_resource_usage(self, metrics: PerformanceMetrics):
        """检查资源使用情况并发出警告"""

        # 内存使用警告
        if metrics.system_resources.memory_usage_mb > 8192:  # 8GB
            self.logger.warning("内存使用过高，建议清理缓存")

        # CPU使用警告
        if metrics.system_resources.cpu_usage_percent > 90:
            self.logger.warning("CPU使用率过高，可能影响性能")

        # 磁盘空间警告
        if metrics.system_resources.disk_usage_percent > 90:
            self.logger.warning("磁盘空间不足，建议清理临时文件")

    def optimize_memory(self):
        """优化内存使用"""

        self.logger.info("正在优化内存使用...")

        # 清理Python垃圾回收
        gc.collect()

        # 清理GPU内存（如果可用）
        try:
            import torch  # pylint: disable=import-outside-toplevel

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        # 清理文件缓存
        try:
            # 清理文件系统缓存（需要root权限）
            subprocess.run(["sync"], capture_output=True, check=False)
            # 注意：直接使用 echo 重定向不会工作，需要使用 shell=True 或者不同的方法
            # 这个操作需要 root 权限，通常会失败，所以我们静默处理
            subprocess.run(
                ["sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"],
                capture_output=True,
                check=False,
            )
        except (IOError, OSError, PermissionError, subprocess.SubprocessError):
            pass

        self.logger.info("内存优化完成")

    def get_error_summary(self) -> Dict:
        """获取错误摘要"""

        total_errors = len(self.error_log)
        error_counts = {"info": 0, "warning": 0, "error": 0, "critical": 0}

        for error in self.error_log:
            error_counts[error.level.value] += 1

        return {
            "total_errors": total_errors,
            "error_counts": error_counts,
            "last_error_time": self.error_log[-1].timestamp if self.error_log else None,
            "most_common_module": (
                max(
                    set(e.location.module for e in self.error_log),
                    key=lambda x: sum(
                        1 for e in self.error_log if e.location.module == x
                    ),
                )
                if self.error_log
                else None
            ),
        }

    def get_performance_summary(self) -> Dict:
        """获取性能摘要"""

        if not self.performance_log:
            return {}

        recent_logs = self.performance_log[-100:]  # 最近100个记录

        return {
            "average_memory_usage_mb": (
                sum(m.system_resources.memory_usage_mb for m in recent_logs)
                / len(recent_logs)
            ),
            "average_cpu_usage_percent": (
                sum(m.system_resources.cpu_usage_percent for m in recent_logs)
                / len(recent_logs)
            ),
            "max_memory_usage_mb": max(
                m.system_resources.memory_usage_mb for m in recent_logs
            ),
            "max_cpu_usage_percent": max(
                m.system_resources.cpu_usage_percent for m in recent_logs
            ),
            "monitoring_duration_hours": (
                (recent_logs[-1].timestamp - recent_logs[0].timestamp) / 3600
                if len(recent_logs) > 1
                else 0
            ),
        }

    def save_logs(self):
        """保存日志到文件"""

        try:
            # 保存错误日志
            error_data = {
                "errors": [error.to_dict() for error in self.error_log],
                "summary": self.get_error_summary(),
            }

            with open(
                Path(self.config.log_file).with_suffix(".errors.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(error_data, f, indent=2, ensure_ascii=False)

            # 保存性能日志
            if self.performance_log:
                performance_data = {
                    "metrics": [metric.to_dict() for metric in self.performance_log],
                    "summary": self.get_performance_summary(),
                }

                with open(
                    Path(self.config.log_file).with_suffix(".performance.json"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(performance_data, f, indent=2, ensure_ascii=False)

            self.logger.info("日志已保存")

        except (IOError, OSError, TypeError, ValueError) as e:
            self.logger.error("保存日志失败: %s", str(e))

    def _cleanup_old_logs(self):
        """清理旧日志"""

        if Path(self.config.log_file).exists() and Path(
            self.config.log_file
        ).stat().st_size > (self.config.max_log_size_mb * 1024 * 1024):
            # 保留最近的错误日志
            keep_count = min(1000, len(self.error_log))
            self.error_log = self.error_log[-keep_count:]

            # 保留最近的性能日志
            keep_count = min(1000, len(self.performance_log))
            self.performance_log = self.performance_log[-keep_count:]

            self.logger.info("已清理旧日志")

    def __del__(self):
        """析构函数"""
        self.stop_performance_monitoring()
        self.save_logs()


def create_error_handler(log_file: str = "error_log.json") -> ErrorHandler:
    """创建错误处理器的便捷函数"""
    return ErrorHandler(log_file)


if __name__ == "__main__":
    # 测试代码
    handler = ErrorHandler()
    handler.start_performance_monitoring(interval=2.0)

    try:
        # 模拟一些操作
        time.sleep(5)

        # 模拟一个错误
        params = handler.LogErrorParams(
            level=ErrorLevel.WARNING,
            module="test",
            function="test_function",
            message="这是一个测试警告",
            context=ErrorContext(),
        )
        handler.log_error(error_params=params)

        # 停止监控
        handler.stop_performance_monitoring()

        # 保存日志
        handler.save_logs()

        print("错误处理测试完成")

    except KeyboardInterrupt:
        handler.stop_performance_monitoring()
        handler.save_logs()
        print("测试被中断")
