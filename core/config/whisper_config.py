"""
Whisper 配置模块，提供 Whisper 语音识别模型的配置管理和优化功能。

该模块包含用于管理 Whisper 模型配置的类和函数，支持根据系统资源和音频特性自动优化配置参数。
主要功能包括：
- 模型配置的创建、保存和加载
- 根据系统资源和音频特性推荐合适的模型大小
- 针对不同语言的优化配置
- 性能指标收集和报告
- 系统资源监控和内存管理

Classes:
    ModelConfig: Whisper 模型配置数据类
    PerformanceMetrics: 性能指标数据类
    WhisperConfigManager: 配置管理器类
"""

import gc
import json
import logging
import os
import platform
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Optional

import psutil
import torch
import yaml


@dataclass
class BasicConfig:
    """基本模型配置"""

    name: str = "whisper_base"
    size: str = "base"  # tiny, base, small, medium, large
    language: str = "auto"
    device: str = "auto"
    compute_type: str = "default"


@dataclass
class CoreDecodingParams:
    """核心解码参数"""

    temperature: float = 0.0
    best_of: int = 5
    beam_size: int = 5
    patience: float = 1.0
    length_penalty: float = 1.0
    suppress_tokens: str = "-1"


@dataclass
class TextProcessingParams:
    """文本处理参数"""

    initial_prompt: Optional[str] = None
    condition_on_previous_text: bool = True


@dataclass
class PunctuationParams:
    """标点符号处理参数"""

    prepend_punctuations: str = "\"'([{-\n"
    append_punctuations: str = "\"'.。,，!！?？:：)]}、\n"


@dataclass
class DecodingConfig:
    """解码相关参数

    将解码参数按功能分组到三个子配置中：
    - core: 核心解码算法参数
    - text: 文本处理相关参数
    - punctuation: 标点符号处理参数

    通过 __getattr__ 和 __setattr__ 方法支持直接访问嵌套属性
    """

    core: CoreDecodingParams = field(default_factory=CoreDecodingParams)
    text: TextProcessingParams = field(default_factory=TextProcessingParams)
    punctuation: PunctuationParams = field(default_factory=PunctuationParams)

    def __getattr__(self, name):
        """支持直接访问嵌套配置的属性"""
        for nested_config in [self.core, self.text, self.punctuation]:
            if hasattr(nested_config, name):
                return getattr(nested_config, name)
        raise AttributeError(f"'{self.__class__.__name__}' 对象没有属性 '{name}'")

    def __setattr__(self, name, value):
        """支持直接设置嵌套配置的属性"""
        # 如果是自身的属性，直接设置
        if name in ["core", "text", "punctuation"]:
            super().__setattr__(name, value)
            return

        # 否则尝试在嵌套配置中设置
        for nested_config in [self.core, self.text, self.punctuation]:
            if hasattr(nested_config, name):
                setattr(nested_config, name, value)
                return

        # 如果属性不存在于任何配置中，则设置为自身的属性
        super().__setattr__(name, value)


@dataclass
class PerformanceConfig:
    """性能优化参数"""

    chunk_length: int = 30  # 音频分块长度（秒）
    max_memory_usage: int = 4096  # 最大内存使用量（MB）
    batch_size: int = 1  # 批处理大小
    num_workers: int = 0  # 工作进程数


@dataclass
class QualityConfig:
    """质量参数"""

    temperature_increment_on_fallback: float = 0.2
    compression_ratio_threshold: float = 2.4
    logprob_threshold: float = -1.0
    no_speech_threshold: float = 0.6
    word_timestamps: bool = True


@dataclass
class ModelConfig:
    """Whisper模型配置类"""

    basic: BasicConfig = field(default_factory=BasicConfig)
    decoding: DecodingConfig = field(default_factory=DecodingConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)

    def __getattr__(self, name):
        """支持直接访问嵌套配置的属性"""
        for nested_config in [self.basic, self.decoding, self.performance, self.quality]:
            if hasattr(nested_config, name):
                return getattr(nested_config, name)
        raise AttributeError(f"'{self.__class__.__name__}' 对象没有属性 '{name}'")

    def __setattr__(self, name, value):
        """支持直接设置嵌套配置的属性"""
        # 如果是自身的属性，直接设置
        if name in ["basic", "decoding", "performance", "quality"]:
            super().__setattr__(name, value)
            return

        # 否则尝试在嵌套配置中设置
        for nested_config in [self.basic, self.decoding, self.performance, self.quality]:
            if hasattr(nested_config, name):
                setattr(nested_config, name, value)
                return

        # 如果属性不存在于任何配置中，则设置为自身的属性
        super().__setattr__(name, value)

    def to_dict(self) -> Dict:
        """转换为字典"""
        result = {}
        for config_name in ["basic", "decoding", "performance", "quality"]:
            config_obj = getattr(self, config_name)
            result.update(asdict(config_obj))
        return result

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "ModelConfig":
        """从字典创建配置"""
        # 创建一个新的 ModelConfig 实例
        model_config = cls()

        # 将字典中的值分配到相应的嵌套配置中
        for key, value in config_dict.items():
            for config_name in ["basic", "decoding", "performance", "quality"]:
                config_obj = getattr(model_config, config_name)
                if hasattr(config_obj, key):
                    setattr(config_obj, key, value)
                    break

        return model_config


@dataclass
class TimeMetrics:
    """时间相关性能指标"""
    model_load_time: float = 0.0  # 模型加载时间（秒）
    transcription_time: float = 0.0  # 转录处理时间（秒）
    audio_duration: float = 0.0  # 音频时长（秒）
    processing_speed: float = 0.0  # 处理速度比率（秒/秒）


@dataclass
class ResourceMetrics:
    """资源使用性能指标"""
    memory_usage_mb: float = 0.0  # 内存使用量（MB）
    cpu_usage_percent: float = 0.0  # CPU使用率（%）
    gpu_memory_mb: float = 0.0  # GPU内存使用量（MB）
    model_size_mb: float = 0.0  # 模型大小（MB）


@dataclass
class PerformanceMetrics:
    """性能指标类

    存储和跟踪 Whisper 模型性能相关的指标，包括处理时间、资源使用情况和效率指标。
    """

    time: TimeMetrics = field(default_factory=TimeMetrics)
    resource: ResourceMetrics = field(default_factory=ResourceMetrics)

    def __getattr__(self, name):
        """支持直接访问嵌套配置的属性"""
        for nested_metrics in [self.time, self.resource]:
            if hasattr(nested_metrics, name):
                return getattr(nested_metrics, name)
        raise AttributeError(f"'{self.__class__.__name__}' 对象没有属性 '{name}'")

    def __setattr__(self, name, value):
        """支持直接设置嵌套配置的属性"""
        # 如果是自身的属性，直接设置
        if name in ["time", "resource"]:
            super().__setattr__(name, value)
            return

        # 否则尝试在嵌套配置中设置
        for nested_metrics in [self.time, self.resource]:
            if hasattr(nested_metrics, name):
                setattr(nested_metrics, name, value)
                return

        # 如果属性不存在于任何配置中，则设置为自身的属性
        super().__setattr__(name, value)


class WhisperConfigManager:
    """Whisper配置管理器"""

    # 支持的模型大小及其内存需求
    MODEL_SIZES = {
        "tiny": {
            "params": "39M",
            "memory_mb": 1000,
            "speed": "最快",
            "accuracy": "最低",
            "recommended_for": "实时转录、低资源环境",
        },
        "base": {
            "params": "74M",
            "memory_mb": 1000,
            "speed": "快",
            "accuracy": "低",
            "recommended_for": "一般用途、平衡性能",
        },
        "small": {
            "params": "244M",
            "memory_mb": 2000,
            "speed": "中等",
            "accuracy": "中等",
            "recommended_for": "高质量转录、标准应用",
        },
        "medium": {
            "params": "769M",
            "memory_mb": 5000,
            "speed": "慢",
            "accuracy": "高",
            "recommended_for": "高精度转录、专业用途",
        },
        "large": {
            "params": "1550M",
            "memory_mb": 10000,
            "speed": "最慢",
            "accuracy": "最高",
            "recommended_for": "研究用途、最高精度",
        },
    }

    # 语言优化配置
    LANGUAGE_OPTIMIZATIONS = {
        "zh": {
            "temperature": 0.2,
            "best_of": 3,
            "beam_size": 3,
            "initial_prompt": "以下是普通话语音转录",
            "language": "zh",
            "chunk_length": 20,
        },
        "en": {
            "temperature": 0.0,
            "best_of": 5,
            "beam_size": 5,
            "initial_prompt": "以下是英语语音转录",
            "language": "en",
            "chunk_length": 30,
        },
        "ja": {
            "temperature": 0.1,
            "best_of": 4,
            "beam_size": 4,
            "initial_prompt": "以下是日语语音转录",
            "language": "ja",
            "chunk_length": 25,
        },
        "ko": {
            "temperature": 0.1,
            "best_of": 4,
            "beam_size": 4,
            "initial_prompt": "以下是韩语语音转录",
            "language": "ko",
            "chunk_length": 25,
        },
        "fr": {
            "temperature": 0.0,
            "best_of": 4,
            "beam_size": 4,
            "initial_prompt": "以下是法语语音转录",
            "language": "fr",
            "chunk_length": 30,
        },
        "de": {
            "temperature": 0.0,
            "best_of": 4,
            "beam_size": 4,
            "initial_prompt": "以下是德语语音转录",
            "language": "de",
            "chunk_length": 30,
        },
        "es": {
            "temperature": 0.0,
            "best_of": 4,
            "beam_size": 4,
            "initial_prompt": "以下是西班牙语语音转录",
            "language": "es",
            "chunk_length": 30,
        },
    }

    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self.logger = self._setup_logger()
        self.performance_metrics = PerformanceMetrics()

    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger("WhisperConfigManager")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def get_system_info(self) -> Dict:
        """获取系统信息"""
        info = {
            "cpu_count": psutil.cpu_count(),
            "memory_total_mb": psutil.virtual_memory().total // (1024 * 1024),
            "memory_available_mb": psutil.virtual_memory().available // (1024 * 1024),
            "disk_usage": psutil.disk_usage("/")._asdict(),
            "platform": {
                "system": os.name,
                "release": platform.uname().release,
                "version": platform.uname().version,
            },
        }

        # GPU 信息
        if torch.cuda.is_available():
            info["gpu"] = {
                "available": True,
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "device_name": torch.cuda.get_device_name(),
                "memory_allocated_mb": torch.cuda.memory_allocated() // (1024 * 1024),
                "memory_reserved_mb": torch.cuda.memory_reserved() // (1024 * 1024),
            }
        else:
            info["gpu"] = {"available": False}

        return info

    def recommend_model_size(
        self, audio_duration: float = 0.0, target_language: str = "auto"
    ) -> str:
        """根据系统资源和需求推荐模型大小"""

        system_info = self.get_system_info()
        available_memory = system_info["memory_available_mb"]

        # 根据可用内存推荐
        if available_memory >= 10000:
            recommended = "large"
        elif available_memory >= 5000:
            recommended = "medium"
        elif available_memory >= 2000:
            recommended = "small"
        elif available_memory >= 1000:
            recommended = "base"
        else:
            recommended = "tiny"

        # 根据音频时长调整推荐
        if audio_duration > 3600:  # 超过1小时
            # 长音频推荐使用较小模型以提高处理速度
            if recommended in ["large", "medium"]:
                recommended = "small"
        elif audio_duration < 300:  # 短音频
            # 短音频可以使用较大模型提高精度
            if recommended == "tiny" and available_memory >= 2000:
                recommended = "base"

        # 语言特定优化
        if target_language != "auto" and target_language in self.LANGUAGE_OPTIMIZATIONS:
            # 获取语言配置但不直接使用，因为在这里我们只需要根据语言类型做决策
            # 实际的语言配置应用在 create_optimized_config 方法中
            _ = self.LANGUAGE_OPTIMIZATIONS[target_language]
            # 某些语言可能需要更大模型
            if target_language in ["zh", "ja", "ko"]:
                if recommended == "tiny" and available_memory >= 2000:
                    recommended = "base"

        self.logger.info(
            "推荐模型大小: %s (可用内存: %sMB)", recommended, available_memory
        )
        return recommended

    def _configure_device(self, model_config: ModelConfig, use_gpu: bool) -> None:
        """配置设备设置"""
        if not use_gpu:
            model_config.device = "cpu"
            return

        if torch.cuda.is_available():
            model_config.device = "cuda"
        elif torch.backends.mps.is_available():
            model_config.device = "mps"
        else:
            model_config.device = "cpu"

    def _apply_language_optimizations(
        self, model_config: ModelConfig, target_language: str
    ) -> None:
        """应用语言特定优化"""
        if (
            target_language == "auto"
            or target_language not in self.LANGUAGE_OPTIMIZATIONS
        ):
            return

        lang_config = self.LANGUAGE_OPTIMIZATIONS[target_language]
        for key, value in lang_config.items():
            if hasattr(model_config, key):
                setattr(model_config, key, value)

    def _optimize_performance(
        self, model_config: ModelConfig, system_info: Dict, audio_duration: float
    ) -> None:
        """优化性能相关配置"""
        # 根据内存调整批处理大小
        if system_info["memory_available_mb"] >= 8000:
            model_config.batch_size = 4
        elif system_info["memory_available_mb"] >= 4000:
            model_config.batch_size = 2
        else:
            model_config.batch_size = 1

        # 根据CPU核心数调整工作进程数
        model_config.num_workers = min(4, system_info["cpu_count"] // 2)

        # 根据音频时长调整分块长度
        if audio_duration > 1800:  # 超过30分钟
            model_config.chunk_length = 60  # 使用更大的分块
        elif audio_duration < 300:  # 短音频
            model_config.chunk_length = 15  # 使用较小的分块

    def create_optimized_config(
        self,
        audio_duration: float = 0.0,
        target_language: str = "auto",
        use_gpu: bool = True,
    ) -> ModelConfig:
        """创建优化配置"""
        # 推荐模型大小
        model_size = self.recommend_model_size(audio_duration, target_language)

        # 基础配置
        model_config = ModelConfig()
        model_config.basic.size = model_size

        # 设备配置
        self._configure_device(model_config, use_gpu)

        # 语言特定优化
        self._apply_language_optimizations(model_config, target_language)

        # 性能优化
        system_info = self.get_system_info()
        self._optimize_performance(model_config, system_info, audio_duration)

        self.logger.info(
            "创建优化配置: 模型=%s, 设备=%s, 语言=%s",
            model_size,
            model_config.device,
            target_language,
        )
        return model_config

    def save_config(self, model_config: ModelConfig, filename: str) -> bool:
        """保存配置到文件"""

        config_path = self.config_dir / filename

        try:
            with open(config_path, "w", encoding="utf-8") as f:
                if config_path.suffix == ".json":
                    json.dump(model_config.to_dict(), f, indent=2, ensure_ascii=False)
                elif config_path.suffix in [".yaml", ".yml"]:
                    yaml.dump(
                        model_config.to_dict(),
                        f,
                        default_flow_style=False,
                        allow_unicode=True,
                    )
                else:
                    # 默认使用 JSON
                    json.dump(model_config.to_dict(), f, indent=2, ensure_ascii=False)

            self.logger.info("配置已保存: %s", config_path)
            return True

        except (IOError, OSError, TypeError, ValueError) as e:
            self.logger.error("保存配置失败: %s", str(e))
            return False

    def load_config(self, filename: str) -> Optional[ModelConfig]:
        """从文件加载配置"""

        config_path = self.config_dir / filename

        if not config_path.exists():
            self.logger.error("配置文件不存在: %s", config_path)
            return None

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                if config_path.suffix == ".json":
                    config_dict = json.load(f)
                elif config_path.suffix in [".yaml", ".yml"]:
                    config_dict = yaml.safe_load(f)
                else:
                    # 尝试 JSON
                    config_dict = json.load(f)

            loaded_config = ModelConfig.from_dict(config_dict)
            self.logger.info("配置已加载: %s", config_path)
            return loaded_config

        except (
            IOError,
            OSError,
            json.JSONDecodeError,
            yaml.YAMLError,
            TypeError,
            ValueError,
        ) as e:
            self.logger.error("加载配置失败: %s", str(e))
            return None

    def get_model_info(self, model_size: str) -> Dict:
        """获取模型信息"""

        if model_size not in self.MODEL_SIZES:
            return {"error": f"不支持的模型大小: {model_size}"}

        return self.MODEL_SIZES[model_size]

    def get_all_model_info(self) -> Dict:
        """获取所有模型信息"""
        return self.MODEL_SIZES.copy()

    def update_performance_metrics(self, metrics: PerformanceMetrics):
        """更新性能指标"""
        self.performance_metrics = metrics

    def get_performance_report(self) -> Dict:
        """获取性能报告"""

        system_info = self.get_system_info()
        metrics_dict = asdict(self.performance_metrics)

        # 获取嵌套结构中的值
        processing_speed = self.performance_metrics.processing_speed
        memory_usage_mb = self.performance_metrics.memory_usage_mb
        model_size_mb = self.performance_metrics.model_size_mb
        cpu_usage_percent = self.performance_metrics.cpu_usage_percent

        report = {
            "system_info": system_info,
            "performance_metrics": metrics_dict,
            "efficiency_analysis": {
                "real_time_factor": processing_speed,
                "memory_efficiency": (
                    memory_usage_mb / model_size_mb
                    if model_size_mb > 0
                    else 0
                ),
                "cpu_efficiency": (
                    cpu_usage_percent / processing_speed
                    if processing_speed > 0
                    else 0
                ),
            },
        }

        return report

    def cleanup_memory(self):
        """清理内存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        self.logger.info("内存已清理")


def create_config_manager(config_dir: str = "config") -> WhisperConfigManager:
    """创建配置管理器的便捷函数"""
    return WhisperConfigManager(config_dir)


if __name__ == "__main__":
    # 测试代码
    config_manager = WhisperConfigManager()

    # 获取系统信息
    sys_info = config_manager.get_system_info()
    print("系统信息:", json.dumps(sys_info, indent=2, ensure_ascii=False))

    # 推荐模型大小
    RECOMMENDED_SIZE = config_manager.recommend_model_size(audio_duration=600)
    print(f"推荐模型大小: {RECOMMENDED_SIZE}")

    # 创建优化配置
    config = config_manager.create_optimized_config(
        audio_duration=600, target_language="zh", use_gpu=True
    )
    print("优化配置:", config.to_dict())
