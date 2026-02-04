"""
错误处理器配置模块。
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ErrorHandlerConfig:
    """错误处理器配置

    使用分层结构组织配置参数，减少直接属性数量，解决"Too many instance attributes"问题。
    将相关配置分组到子配置类中，提高代码的可维护性和可读性。
    """

    @dataclass
    class LoggingConfig:
        """日志相关配置"""

        log_file: str = "error_log.json"
        max_log_size_mb: int = 100
        auto_save_logs: bool = True
        save_interval_minutes: int = 30

    @dataclass
    class MonitoringConfig:
        """监控相关配置"""

        enable_performance_monitoring: bool = True
        performance_monitoring_interval: float = 5.0
        enable_signal_handlers: bool = True
        critical_error_threshold: int = 10

    @dataclass
    class RetryConfig:
        """重试策略配置"""

        max_error_retries: int = 3
        retry_delay_base: float = 1.0
        retry_delay_max: float = 60.0
        exponential_backoff: bool = True

    @dataclass
    class MemoryConfig:
        """内存管理配置"""

        enable_memory_optimization: bool = True
        memory_cleanup_threshold_mb: int = 4096

    # 主要配置属性
    logging: Optional[LoggingConfig] = None
    monitoring: Optional[MonitoringConfig] = None
    retry: Optional[RetryConfig] = None
    memory: Optional[MemoryConfig] = None

    def __post_init__(self):
        """初始化后创建子配置对象"""
        if self.logging is None:
            self.logging = self.LoggingConfig()
        if self.monitoring is None:
            self.monitoring = self.MonitoringConfig()
        if self.retry is None:
            self.retry = self.RetryConfig()
        if self.memory is None:
            self.memory = self.MemoryConfig()

    # 为了保持向后兼容性，提供属性访问器
    @property
    def log_file(self) -> str:
        """获取日志文件路径"""
        if self.logging is None:
            return "error_log.json"  # 默认值
        return self.logging.log_file

    @property
    def max_log_size_mb(self) -> int:
        """获取最大日志大小(MB)"""
        if self.logging is None:
            return 100  # 默认值
        return self.logging.max_log_size_mb

    @property
    def auto_save_logs(self) -> bool:
        """获取是否自动保存日志"""
        if self.logging is None:
            return True  # 默认值
        return self.logging.auto_save_logs

    @property
    def save_interval_minutes(self) -> int:
        """获取日志保存间隔(分钟)"""
        if self.logging is None:
            return 30  # 默认值
        return self.logging.save_interval_minutes

    @property
    def enable_performance_monitoring(self) -> bool:
        """获取是否启用性能监控"""
        if self.monitoring is None:
            return True  # 默认值
        return self.monitoring.enable_performance_monitoring

    @property
    def performance_monitoring_interval(self) -> float:
        """获取性能监控间隔(秒)"""
        if self.monitoring is None:
            return 5.0  # 默认值
        return self.monitoring.performance_monitoring_interval

    @property
    def enable_signal_handlers(self) -> bool:
        """获取是否启用信号处理器"""
        if self.monitoring is None:
            return True  # 默认值
        return self.monitoring.enable_signal_handlers

    @property
    def critical_error_threshold(self) -> int:
        """获取严重错误阈值"""
        if self.monitoring is None:
            return 10  # 默认值
        return self.monitoring.critical_error_threshold

    @property
    def max_error_retries(self) -> int:
        """获取最大错误重试次数"""
        if self.retry is None:
            return 3  # 默认值
        return self.retry.max_error_retries

    @property
    def retry_delay_base(self) -> float:
        """获取重试延迟基准值(秒)"""
        if self.retry is None:
            return 1.0  # 默认值
        return self.retry.retry_delay_base

    @property
    def retry_delay_max(self) -> float:
        """获取最大重试延迟(秒)"""
        if self.retry is None:
            return 60.0  # 默认值
        return self.retry.retry_delay_max

    @property
    def exponential_backoff(self) -> bool:
        """获取是否使用指数退避策略"""
        if self.retry is None:
            return True  # 默认值
        return self.retry.exponential_backoff

    @property
    def enable_memory_optimization(self) -> bool:
        """获取是否启用内存优化"""
        if self.memory is None:
            return True  # 默认值
        return self.memory.enable_memory_optimization

    @property
    def memory_cleanup_threshold_mb(self) -> int:
        """获取内存清理阈值(MB)"""
        if self.memory is None:
            return 4096  # 默认值
        return self.memory.memory_cleanup_threshold_mb
