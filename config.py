"""
配置管理模块 - 处理项目的配置加载、保存和验证

本模块提供了一套完整的配置管理系统，包括：
- 默认配置的创建和加载
- 用户配置的保存和更新
- 配置验证和错误检查
- 预设配置的应用
- 配置摘要生成

主要组件:
- ProjectConfig: 项目整体配置数据类
- ConfigManager: 配置管理器类，处理配置文件的读写和验证
"""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from whisper_config import ModelConfig, BasicConfig
from config_subtitle import SubtitleGeneratorConfig
from config_error_handler import ErrorHandlerConfig
from config_cli import CLIConfig


@dataclass
class FileScannerConfig:
    """文件扫描器配置"""

    scan_directory: str = "downloads"
    recursive: bool = True
    file_patterns: Optional[List[str]] = None
    max_file_size_mb: int = 1024  # 最大文件大小（MB）
    min_file_size_mb: float = 0.1  # 最小文件大小（MB）
    supported_formats: Optional[List[str]] = None
    exclude_patterns: Optional[List[str]] = None

    def __post_init__(self):
        if self.file_patterns is None:
            self.file_patterns = [
                "*.mp3",
                "*.wav",
                "*.flac",
                "*.aac",
                "*.ogg",
                "*.mp4",
                "*.webm",
                "*.mkv",
                "*.avi",
                "*.mov",
            ]
        if self.supported_formats is None:
            self.supported_formats = [
                ".mp3",
                ".wav",
                ".flac",
                ".aac",
                ".ogg",
                ".mp4",
                ".webm",
                ".mkv",
                ".avi",
                ".mov",
            ]
        if self.exclude_patterns is None:
            self.exclude_patterns = ["*.tmp", "*.temp", "*.log", "*.bak"]


@dataclass
class OutputConfig:
    """输出配置"""

    output_directory: str = "converted_audio"
    target_format: str = "wav"


@dataclass
class AudioQualityConfig:
    """音频质量配置"""

    sample_rate: int = 16000
    channels: int = 1
    bit_depth: int = 16
    codec: str = "pcm_s16le"


@dataclass
class AudioProcessingConfig:
    """音频处理配置"""

    normalize_audio: bool = True
    target_level: float = -23.0  # LUFS
    remove_silence: bool = False
    silence_threshold: float = -40.0  # dB


@dataclass
class ChunkingConfig:
    """分块和并行处理配置"""

    chunk_length: int = 30  # 分块长度（秒）
    max_chunk_size_mb: int = 100  # 最大分块大小（MB）
    enable_parallel_processing: bool = True
    max_workers: int = 4


@dataclass
class FormatConverterConfig:
    """格式转换器配置"""

    output: Optional[OutputConfig] = None
    quality: Optional[AudioQualityConfig] = None
    processing: Optional[AudioProcessingConfig] = None
    chunking: Optional[ChunkingConfig] = None

    def __post_init__(self):
        if self.output is None:
            self.output = OutputConfig()
        if self.quality is None:
            self.quality = AudioQualityConfig()
        if self.processing is None:
            self.processing = AudioProcessingConfig()
        if self.chunking is None:
            self.chunking = ChunkingConfig()

    # 为了保持向后兼容性，提供属性访问器
    @property
    def output_directory(self) -> str:
        """获取输出目录路径

        Returns:
            str: 输出目录的路径
        """
        if self.output is None:
            return "converted_audio"  # 默认值
        return self.output.output_directory

    @property
    def target_format(self) -> str:
        """获取目标格式

        Returns:
            str: 音频转换的目标格式
        """
        if self.output is None:
            return "wav"  # 默认值
        return self.output.target_format

    @property
    def sample_rate(self) -> int:
        """获取音频采样率

        Returns:
            int: 音频采样率（Hz）
        """
        if self.quality is None:
            return 16000  # 默认值
        return self.quality.sample_rate

    @property
    def channels(self) -> int:
        """获取音频通道数

        Returns:
            int: 音频通道数（单声道=1，立体声=2）
        """
        if self.quality is None:
            return 1  # 默认值
        return self.quality.channels

    @property
    def bit_depth(self) -> int:
        """获取音频位深度

        Returns:
            int: 音频位深度（如16位、24位等）
        """
        if self.quality is None:
            return 16  # 默认值
        return self.quality.bit_depth

    @property
    def codec(self) -> str:
        """获取音频编解码器

        Returns:
            str: 音频编解码器名称
        """
        if self.quality is None:
            return "pcm_s16le"  # 默认值
        return self.quality.codec

    @property
    def normalize_audio(self) -> bool:
        """获取是否启用音频标准化

        Returns:
            bool: 如果启用音频标准化则为True，否则为False
        """
        if self.processing is None:
            return True  # 默认值
        return self.processing.normalize_audio

    @property
    def target_level(self) -> float:
        """获取音频标准化的目标电平

        Returns:
            float: 目标电平值（LUFS）
        """
        if self.processing is None:
            return -23.0  # 默认值
        return self.processing.target_level

    @property
    def remove_silence(self) -> bool:
        """获取是否移除静音部分

        Returns:
            bool: 如果启用静音移除则为True，否则为False
        """
        if self.processing is None:
            return False  # 默认值
        return self.processing.remove_silence

    @property
    def silence_threshold(self) -> float:
        """获取静音检测阈值

        Returns:
            float: 静音检测阈值（dB）
        """
        if self.processing is None:
            return -40.0  # 默认值
        return self.processing.silence_threshold

    @property
    def chunk_length(self) -> int:
        """获取音频分块长度

        Returns:
            int: 分块长度（秒）
        """
        if self.chunking is None:
            return 30  # 默认值
        return self.chunking.chunk_length

    @property
    def max_chunk_size_mb(self) -> int:
        """获取最大分块大小

        Returns:
            int: 最大分块大小（MB）
        """
        if self.chunking is None:
            return 100  # 默认值
        return self.chunking.max_chunk_size_mb

    @property
    def enable_parallel_processing(self) -> bool:
        """获取是否启用并行处理

        Returns:
            bool: 如果启用并行处理则为True，否则为False
        """
        if self.chunking is None:
            return True  # 默认值
        return self.chunking.enable_parallel_processing

    @property
    def max_workers(self) -> int:
        """获取最大工作线程数

        Returns:
            int: 并行处理的最大工作线程数
        """
        if self.chunking is None:
            return 4  # 默认值
        return self.chunking.max_workers


@dataclass
class WhisperConfig:
    """Whisper模型配置

    使用分层结构组织配置参数，减少直接属性数量，解决"Too many instance attributes"问题。
    实际配置参数存储在whisper_config.py的ModelConfig类中。
    """

    # 基本配置
    model_size: str = "base"  # tiny, base, small, medium, large
    language: str = "auto"
    device: str = "auto"  # cpu, cuda, mps

    # 模型文件配置
    download_root: str = "./models"
    cache_dir: Optional[str] = None
    local_files_only: bool = False

    # 内部配置对象
    _model_config: Optional[Any] = None

    def __post_init__(self):
        """初始化后创建内部ModelConfig对象"""
        # 创建基础配置
        basic = BasicConfig(
            size=self.model_size,
            language=self.language,
            device=self.device,
        )

        # 创建完整配置
        self._model_config = ModelConfig(basic=basic)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        if self._model_config:
            config_dict = self._model_config.to_dict()
            # 添加额外属性
            config_dict.update(
                {
                    "download_root": self.download_root,
                    "cache_dir": self.cache_dir,
                    "local_files_only": self.local_files_only,
                }
            )
            return config_dict

        # 如果内部配置未初始化，返回基本属性
        return {
            "model_size": self.model_size,
            "language": self.language,
            "device": self.device,
            "download_root": self.download_root,
            "cache_dir": self.cache_dir,
            "local_files_only": self.local_files_only,
        }

    def __getattr__(self, name):
        """支持访问内部ModelConfig的属性"""
        if self._model_config and hasattr(self._model_config, name):
            return getattr(self._model_config, name)
        raise AttributeError(f"'{self.__class__.__name__}' 对象没有属性 '{name}'")








@dataclass
class ResourceConfig:
    """资源使用配置"""

    enable_gpu_acceleration: bool = True
    max_memory_usage_mb: int = 8192
    max_cpu_usage_percent: int = 80
    enable_memory_optimization: bool = True


@dataclass
class ProcessingConfig:
    """处理配置"""

    batch_size: int = 1
    max_workers: int = 4
    chunk_overlap: float = 0.1
    enable_streaming: bool = False
    streaming_buffer_size_mb: int = 10


@dataclass
class CacheConfig:
    """缓存配置"""

    cache_transcriptions: bool = True
    cache_directory: str = "./cache"
    cache_expiry_days: int = 30


@dataclass
class PerformanceConfig:
    """性能配置"""

    resources: Optional[ResourceConfig] = None
    processing: Optional[ProcessingConfig] = None
    cache: Optional[CacheConfig] = None
    enable_profiling: bool = False
    profiling_output: str = "./profiling"

    def __post_init__(self):
        if self.resources is None:
            self.resources = ResourceConfig()
        if self.processing is None:
            self.processing = ProcessingConfig()
        if self.cache is None:
            self.cache = CacheConfig()





@dataclass
class ProjectConfig:
    """项目整体配置"""

    project_name: str = "语音转文字工具"
    version: str = "1.0.0"
    description: str = (
        "基于 OpenAI Whisper 的语音转文字工具，支持多种音视频格式和字幕生成"
    )
    author: str = "MyToolbox Team"
    license: str = "MIT"

    # 模块配置
    file_scanner: Optional[FileScannerConfig] = None
    format_converter: Optional[FormatConverterConfig] = None
    whisper: Optional[WhisperConfig] = None
    subtitle_generator: Optional[SubtitleGeneratorConfig] = None
    error_handler: Optional[ErrorHandlerConfig] = None
    performance: Optional[PerformanceConfig] = None
    cli: Optional[CLIConfig] = None

    def __post_init__(self):
        if self.file_scanner is None:
            self.file_scanner = FileScannerConfig()
        if self.format_converter is None:
            self.format_converter = FormatConverterConfig()
        if self.whisper is None:
            self.whisper = WhisperConfig()
        if self.subtitle_generator is None:
            self.subtitle_generator = SubtitleGeneratorConfig()
        if self.error_handler is None:
            self.error_handler = ErrorHandlerConfig()
        if self.performance is None:
            self.performance = PerformanceConfig()
        if self.cli is None:
            self.cli = CLIConfig()

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        # 为了保持与原始结构兼容，我们需要特殊处理 format_converter
        format_converter_dict = {}
        fc = self.format_converter

        # 将嵌套结构扁平化
        if fc is not None:
            if fc.output is not None:
                format_converter_dict.update(asdict(fc.output))
            if fc.quality is not None:
                format_converter_dict.update(asdict(fc.quality))
            if fc.processing is not None:
                format_converter_dict.update(asdict(fc.processing))
            if fc.chunking is not None:
                format_converter_dict.update(asdict(fc.chunking))

        # 处理错误处理器配置，将嵌套结构扁平化
        error_handler_dict = {}
        eh = self.error_handler

        # 将嵌套结构扁平化
        if eh is not None:
            if eh.logging is not None:
                error_handler_dict.update(asdict(eh.logging))
            if eh.monitoring is not None:
                error_handler_dict.update(asdict(eh.monitoring))
            if eh.retry is not None:
                error_handler_dict.update(asdict(eh.retry))
            if eh.memory is not None:
                error_handler_dict.update(asdict(eh.memory))

        # 安全地处理可能为None的字段
        result = {
            "project_name": self.project_name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "license": self.license,
            "format_converter": format_converter_dict,
            "error_handler": error_handler_dict,
        }

        # 安全地添加可能为None的字段
        if self.file_scanner is not None:
            result["file_scanner"] = asdict(self.file_scanner)
        if self.whisper is not None:
            result["whisper"] = asdict(self.whisper)
        if self.subtitle_generator is not None:
            result["subtitle_generator"] = asdict(self.subtitle_generator)
        if self.performance is not None:
            result["performance"] = asdict(self.performance)
        if self.cli is not None:
            result["cli"] = asdict(self.cli)

        return result

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ProjectConfig":
        """从字典创建配置"""

        # 提取基础配置
        base_config = {}
        for key in ["project_name", "version", "description", "author", "license"]:
            if key in config_dict:
                base_config[key] = config_dict[key]

        # 创建模块配置
        file_scanner = FileScannerConfig(**config_dict.get("file_scanner", {}))

        # 创建格式转换器配置
        format_converter = cls._create_format_converter_config(
            config_dict.get("format_converter", {})
        )

        # 创建其他模块配置
        module_configs = {
            "whisper": WhisperConfig(**config_dict.get("whisper", {})),
            "subtitle_generator": SubtitleGeneratorConfig(
                **config_dict.get("subtitle_generator", {})
            ),
            "error_handler": cls._create_error_handler_config(
                config_dict.get("error_handler", {})
            ),
            "performance": PerformanceConfig(**config_dict.get("performance", {})),
            "cli": CLIConfig(**config_dict.get("cli", {})),
        }

        return cls(
            **base_config,
            file_scanner=file_scanner,
            format_converter=format_converter,
            **module_configs,
        )

    @staticmethod
    def _create_format_converter_config(
        fc_dict: Dict[str, Any],
    ) -> FormatConverterConfig:
        """从字典创建格式转换器配置"""
        # 创建嵌套配置对象
        output_config = OutputConfig(
            output_directory=fc_dict.get("output_directory", "converted_audio"),
            target_format=fc_dict.get("target_format", "wav"),
        )
        quality_config = AudioQualityConfig(
            sample_rate=fc_dict.get("sample_rate", 16000),
            channels=fc_dict.get("channels", 1),
            bit_depth=fc_dict.get("bit_depth", 16),
            codec=fc_dict.get("codec", "pcm_s16le"),
        )
        processing_config = AudioProcessingConfig(
            normalize_audio=fc_dict.get("normalize_audio", True),
            target_level=fc_dict.get("target_level", -23.0),
            remove_silence=fc_dict.get("remove_silence", False),
            silence_threshold=fc_dict.get("silence_threshold", -40.0),
        )
        chunking_config = ChunkingConfig(
            chunk_length=fc_dict.get("chunk_length", 30),
            max_chunk_size_mb=fc_dict.get("max_chunk_size_mb", 100),
            enable_parallel_processing=fc_dict.get("enable_parallel_processing", True),
            max_workers=fc_dict.get("max_workers", 4),
        )

        return FormatConverterConfig(
            output=output_config,
            quality=quality_config,
            processing=processing_config,
            chunking=chunking_config,
        )

    @staticmethod
    def _create_error_handler_config(
        eh_dict: Dict[str, Any],
    ) -> ErrorHandlerConfig:
        """从字典创建错误处理器配置

        将扁平的字典结构转换为分层的ErrorHandlerConfig对象

        Args:
            eh_dict: 包含错误处理器配置的字典

        Returns:
            ErrorHandlerConfig: 创建的错误处理器配置对象
        """
        # 创建嵌套配置对象
        logging_config = ErrorHandlerConfig.LoggingConfig(
            log_file=eh_dict.get("log_file", "error_log.json"),
            max_log_size_mb=eh_dict.get("max_log_size_mb", 100),
            auto_save_logs=eh_dict.get("auto_save_logs", True),
            save_interval_minutes=eh_dict.get("save_interval_minutes", 30),
        )

        monitoring_config = ErrorHandlerConfig.MonitoringConfig(
            enable_performance_monitoring=eh_dict.get(
                "enable_performance_monitoring", True
            ),
            performance_monitoring_interval=eh_dict.get(
                "performance_monitoring_interval", 5.0
            ),
            enable_signal_handlers=eh_dict.get("enable_signal_handlers", True),
            critical_error_threshold=eh_dict.get("critical_error_threshold", 10),
        )

        retry_config = ErrorHandlerConfig.RetryConfig(
            max_error_retries=eh_dict.get("max_error_retries", 3),
            retry_delay_base=eh_dict.get("retry_delay_base", 1.0),
            retry_delay_max=eh_dict.get("retry_delay_max", 60.0),
            exponential_backoff=eh_dict.get("exponential_backoff", True),
        )

        memory_config = ErrorHandlerConfig.MemoryConfig(
            enable_memory_optimization=eh_dict.get("enable_memory_optimization", True),
            memory_cleanup_threshold_mb=eh_dict.get(
                "memory_cleanup_threshold_mb", 4096
            ),
        )

        return ErrorHandlerConfig(
            logging=logging_config,
            monitoring=monitoring_config,
            retry=retry_config,
            memory=memory_config,
        )


class ConfigManager:
    """配置管理器"""

    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self.default_config_file = self.config_dir / "default_config.json"
        self.user_config_file = self.config_dir / "user_config.json"
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger("ConfigManager")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def create_default_config(self) -> ProjectConfig:
        """创建默认配置"""
        return ProjectConfig()

    def save_default_config(self) -> bool:
        """保存默认配置"""

        try:
            project_config = self.create_default_config()

            with open(self.default_config_file, "w", encoding="utf-8") as f:
                json.dump(project_config.to_dict(), f, indent=2, ensure_ascii=False)

            self.logger.info("默认配置已保存: %s", self.default_config_file)
            return True

        except (IOError, PermissionError) as e:
            self.logger.error("保存默认配置失败: IO或权限错误: %s", str(e))
            return False
        except (json.JSONDecodeError, TypeError) as e:
            self.logger.error("保存默认配置失败: 数据序列化错误: %s", str(e))
            return False

    def load_config(self, config_file: Optional[Path] = None) -> ProjectConfig:
        """加载配置"""

        if config_file is None:
            # 优先加载用户配置，然后默认配置
            if self.user_config_file.exists():
                config_file = self.user_config_file
            elif self.default_config_file.exists():
                config_file = self.default_config_file
            else:
                self.logger.info("未找到配置文件，使用默认配置")
                return self.create_default_config()

        try:
            with open(config_file, "r", encoding="utf-8") as f:
                if config_file.suffix == ".json":
                    config_dict = json.load(f)
                elif config_file.suffix in [".yaml", ".yml"]:
                    config_dict = yaml.safe_load(f)
                else:
                    # 尝试 JSON
                    config_dict = json.load(f)

            loaded_config = ProjectConfig.from_dict(config_dict)
            self.logger.info("配置已加载: %s", config_file)
            return loaded_config

        except (json.JSONDecodeError, yaml.YAMLError) as e:
            self.logger.error("解析配置文件失败: %s", str(e))
            self.logger.info("使用默认配置")
            return self.create_default_config()
        except (IOError, PermissionError) as e:
            self.logger.error("读取配置文件失败: %s", str(e))
            self.logger.info("使用默认配置")
            return self.create_default_config()
        except (KeyError, TypeError, ValueError) as e:
            self.logger.error("配置文件格式错误: %s", str(e))
            self.logger.info("使用默认配置")
            return self.create_default_config()

    def save_config(
        self, project_config: ProjectConfig, config_file: Optional[Path] = None
    ) -> bool:
        """保存配置"""

        if config_file is None:
            config_file = self.user_config_file

        try:
            with open(config_file, "w", encoding="utf-8") as f:
                if config_file.suffix == ".json":
                    json.dump(project_config.to_dict(), f, indent=2, ensure_ascii=False)
                elif config_file.suffix in [".yaml", ".yml"]:
                    yaml.dump(
                        project_config.to_dict(),
                        f,
                        default_flow_style=False,
                        allow_unicode=True,
                    )
                else:
                    # 默认使用 JSON
                    json.dump(project_config.to_dict(), f, indent=2, ensure_ascii=False)

            self.logger.info("配置已保存: %s", config_file)
            return True

        except (IOError, PermissionError) as e:
            self.logger.error("保存配置文件时出现IO错误: %s", str(e))
            return False
        except (json.JSONDecodeError, yaml.YAMLError) as e:
            self.logger.error("序列化配置数据失败: %s", str(e))
            return False
        except TypeError as e:
            self.logger.error("配置数据类型错误: %s", str(e))
            return False
        except ValueError as e:
            self.logger.error("配置数据值错误: %s", str(e))
            return False

    def update_config(
        self, updates: Dict[str, Any], config_file: Optional[Path] = None
    ) -> bool:
        """更新配置"""

        loaded_config = self.load_config(config_file)

        # 递归更新配置字典
        def update_dict(d: Dict, u: Dict) -> Dict:
            for k, v in u.items():
                if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                    d[k] = update_dict(d[k], v)
                else:
                    d[k] = v
            return d

        config_dict = loaded_config.to_dict()
        updated_config_dict = update_dict(config_dict, updates)

        updated_config = ProjectConfig.from_dict(updated_config_dict)

        return self.save_config(updated_config, config_file)

    def validate_config(self, project_config: ProjectConfig) -> Dict[str, List[str]]:
        """验证配置有效性"""

        validation_errors = {}

        # 验证文件扫描器配置
        if project_config.file_scanner is not None:
            if (
                project_config.file_scanner.max_file_size_mb is not None
                and project_config.file_scanner.min_file_size_mb is not None
                and project_config.file_scanner.max_file_size_mb
                < project_config.file_scanner.min_file_size_mb
            ):
                validation_errors.setdefault("file_scanner", []).append(
                    "最大文件大小不能小于最小文件大小"
                )

        # 验证格式转换器配置
        if (
            project_config.format_converter is not None
            and project_config.format_converter.quality is not None
            and project_config.format_converter.quality.sample_rate is not None
            and project_config.format_converter.quality.sample_rate
            not in [
                8000,
                16000,
                22050,
                44100,
                48000,
            ]
        ):
            validation_errors.setdefault("format_converter", []).append(
                "不支持的采样率，支持: 8000, 16000, 22050, 44100, 48000"
            )

        # 验证Whisper配置
        if (
            project_config.whisper is not None
            and project_config.whisper.model_size is not None
            and project_config.whisper.model_size
            not in [
                "tiny",
                "base",
                "small",
                "medium",
                "large",
            ]
        ):
            validation_errors.setdefault("whisper", []).append(
                "不支持的模型大小，支持: tiny, base, small, medium, large"
            )

        # 验证性能配置
        if (
            project_config.performance is not None
            and project_config.performance.resources is not None
            and project_config.performance.resources.max_memory_usage_mb is not None
            and project_config.performance.resources.max_memory_usage_mb < 100
        ):
            validation_errors.setdefault("performance", []).append(
                "最大内存使用量不能小于100MB"
            )

        return validation_errors

    def get_config_summary(
        self, project_config: ProjectConfig
    ) -> Dict[str, Dict[str, Any]]:
        """获取配置摘要"""

        # 创建基本项目信息
        config_summary: Dict[str, Dict[str, Any]] = {
            "project": {
                "name": project_config.project_name,
                "version": project_config.version,
                "author": project_config.author,
            }
        }

        # 安全地添加文件扫描器信息
        if project_config.file_scanner is not None:
            config_summary["file_scanner"] = {
                "scan_directory": getattr(
                    project_config.file_scanner, "scan_directory", "未设置"
                ),
                "recursive": getattr(project_config.file_scanner, "recursive", False),
                "supported_formats": len(
                    getattr(project_config.file_scanner, "supported_formats", []) or []
                ),
            }

        # 安全地添加Whisper信息
        if project_config.whisper is not None:
            config_summary["whisper"] = {
                "model_size": getattr(project_config.whisper, "model_size", "未设置"),
                "language": getattr(project_config.whisper, "language", "未设置"),
                "device": getattr(project_config.whisper, "device", "未设置"),
            }

        # 安全地添加性能信息
        if project_config.performance is not None:
            perf_summary = {}

            # 检查resources是否存在
            if (
                hasattr(project_config.performance, "resources")
                and project_config.performance.resources is not None
            ):
                perf_summary["enable_gpu"] = getattr(
                    project_config.performance.resources,
                    "enable_gpu_acceleration",
                    False,
                )
                perf_summary["max_memory_mb"] = getattr(
                    project_config.performance.resources, "max_memory_usage_mb", 0
                )

            # 检查processing是否存在
            if (
                hasattr(project_config.performance, "processing")
                and project_config.performance.processing is not None
            ):
                perf_summary["max_workers"] = getattr(
                    project_config.performance.processing, "max_workers", 1
                )

            config_summary["performance"] = perf_summary

        return config_summary

    def create_preset_config(self, preset_name: str) -> ProjectConfig:
        """创建预设配置"""

        presets = {
            "fast": {
                "whisper": {"model_size": "tiny"},
                "performance": {"processing": {"max_workers": 8, "batch_size": 4}},
            },
            "accurate": {
                "whisper": {"model_size": "medium"},
                "performance": {"processing": {"max_workers": 2, "batch_size": 1}},
            },
            "balanced": {
                "whisper": {"model_size": "base"},
                "performance": {"processing": {"max_workers": 4, "batch_size": 2}},
            },
            "low_memory": {
                "whisper": {"model_size": "tiny"},
                "performance": {
                    "resources": {"max_memory_usage_mb": 2048},
                    "processing": {"max_workers": 2},
                },
            },
            "high_quality": {
                "whisper": {"model_size": "large"},
                "performance": {
                    "resources": {"max_memory_usage_mb": 16384},
                    "processing": {"batch_size": 1},
                },
            },
        }

        if preset_name not in presets:
            raise ValueError(f"不支持的预设: {preset_name}")

        project_config = self.create_default_config()
        preset_updates = presets[preset_name]

        return self._apply_preset(project_config, preset_updates)

    def _apply_preset(
        self, project_config: ProjectConfig, preset: Dict[str, Any]
    ) -> ProjectConfig:
        """应用预设配置"""

        config_dict = project_config.to_dict()

        def update_dict(d: Dict, u: Dict) -> Dict:
            for k, v in u.items():
                if isinstance(v, dict) and k in d:
                    d[k] = update_dict(d[k], v)
                else:
                    d[k] = v
            return d

        updated_config_dict = update_dict(config_dict, preset)

        return ProjectConfig.from_dict(updated_config_dict)


def create_config_manager(config_dir: str = "config") -> ConfigManager:
    """创建配置管理器的便捷函数"""
    return ConfigManager(config_dir)


if __name__ == "__main__":
    # 测试代码
    config_manager = ConfigManager()

    # 创建并保存默认配置
    config_manager.save_default_config()

    # 加载配置
    config = config_manager.load_config()

    # 显示配置摘要
    summary = config_manager.get_config_summary(config)
    print("配置摘要:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    # 验证配置
    validation_results = config_manager.validate_config(config)
    if validation_results:
        print("配置错误:")
        print(json.dumps(validation_results, indent=2, ensure_ascii=False))
    else:
        print("配置验证通过")
