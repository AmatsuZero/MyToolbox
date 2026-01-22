"""
Whisper语音识别模型集成模块，提供音频转录和处理功能。

该模块封装了OpenAI的Whisper模型，提供了易用的接口进行音频转录。
主要功能包括：
- 支持多种模型大小（tiny、base、small、medium、large）
- 自动设备选择（CPU、CUDA、MPS）
- 批量音频文件转录
- 多语言支持和语言特定优化
- 配置保存和加载
- 针对Apple Silicon的MPS后端优化

Classes:
    TranscriptionResult: 存储转录结果的数据类
    ModelConfig: 模型配置数据类
    WhisperIntegration: 主要的Whisper集成类
"""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast

import torch
import whisper


def _patch_mps_float64():
    """
    修补 MPS 后端的 float64 问题。
    MPS 不支持 float64，通过 monkey patch 将相关操作重定向到 float32。
    """
    if not torch.backends.mps.is_available():
        return

    # 保存原始的 torch.Tensor.to 方法
    _original_to = torch.Tensor.to

    def _patched_to(self, *args, **kwargs):
        # 如果目标是 MPS 设备且 tensor 是 float64，先转为 float32
        device = None
        dtype = kwargs.get("dtype", None)

        if args:
            if isinstance(args[0], torch.device):
                device = args[0]
            elif isinstance(args[0], str):
                device = torch.device(args[0])
            elif isinstance(args[0], torch.dtype):
                dtype = args[0]

        # 检查是否转移到 MPS
        is_to_mps = (device is not None and device.type == "mps") or (
            device is None and self.device.type == "mps"
        )

        # 如果是 float64 且目标是 MPS，先转为 float32
        if is_to_mps and torch.float64 in (self.dtype, dtype):
            self = self.float()  # 转为 float32
            if dtype == torch.float64:
                kwargs["dtype"] = torch.float32

        return _original_to(self, *args, **kwargs)

    torch.Tensor.to = _patched_to

    # 同时修补 numpy 转 tensor 的默认行为
    _original_from_numpy = torch.from_numpy

    def _patched_from_numpy(ndarray):
        tensor = _original_from_numpy(ndarray)
        # 如果是 float64，转为 float32
        if tensor.dtype == torch.float64:
            tensor = tensor.float()
        return tensor

    torch.from_numpy = _patched_from_numpy


# 在导入时就执行修补
_patch_mps_float64()


@dataclass
class TranscriptionResult:  # pylint: disable=too-many-instance-attributes
    """语音识别结果类

    包含音频转录的完整结果信息，包括文本、分段、语言、置信度等。
    由于需要完整记录转录结果的各个方面，属性数量超过默认限制是合理的。
    """

    audio_file: Path
    text: str
    segments: List[Dict[str, Union[str, float, int]]]
    language: str
    confidence: float
    processing_time: float
    model_used: str
    error_message: Optional[str] = None
    success: bool = True


@dataclass
class ModelConfig:  # pylint: disable=too-many-instance-attributes
    """模型配置类

    包含Whisper模型的所有配置参数，用于控制模型加载和转录行为。
    由于需要完整支持Whisper的各种参数选项，属性数量超过默认限制是必要的。
    """

    name: str
    size: str  # tiny, base, small, medium, large
    language: str = "auto"
    device: str = "auto"
    compute_type: str = "default"
    temperature: float = 0.0
    best_of: int = 5
    beam_size: int = 5
    patience: float = 1.0
    length_penalty: float = 1.0
    suppress_tokens: str = "-1"
    initial_prompt: Optional[str] = None
    condition_on_previous_text: bool = True
    compression_ratio_threshold: float = 2.4
    logprob_threshold: float = -1.0
    no_speech_threshold: float = 0.6
    word_timestamps: bool = True
    prepend_punctuations: str = "\"'“¿([{-\n"
    append_punctuations: str = "\"'.。,，!！?？:：”)]}、\n"


class WhisperIntegration:
    """Whisper模型集成类"""

    # 支持的模型大小及其内存需求（近似值）
    MODEL_SIZES = {
        "tiny": {
            "params": "39M",
            "memory": "~1GB",
            "speed": "最快",
            "accuracy": "最低",
        },
        "base": {"params": "74M", "memory": "~1GB", "speed": "快", "accuracy": "低"},
        "small": {
            "params": "244M",
            "memory": "~2GB",
            "speed": "中等",
            "accuracy": "中等",
        },
        "medium": {"params": "769M", "memory": "~5GB", "speed": "慢", "accuracy": "高"},
        "large": {
            "params": "1550M",
            "memory": "~10GB",
            "speed": "最慢",
            "accuracy": "最高",
        },
    }

    def __init__(self, model_config: Optional[ModelConfig] = None):
        self.config = model_config or ModelConfig("default", "base")
        self.model = None
        self.logger = self._setup_logger()
        self._setup_device()

    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger("WhisperIntegration")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _setup_device(self):
        """设置计算设备"""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                self.config.device = "cuda"
                self.logger.info("使用 CUDA GPU 加速")
            elif torch.backends.mps.is_available():
                self.config.device = "mps"
                self.logger.info("使用 Apple Silicon GPU 加速")
                # MPS 不支持 float64，提前设置默认 dtype 为 float32
                torch.set_default_dtype(torch.float32)
            else:
                self.config.device = "cpu"
                self.logger.info("使用 CPU 进行计算")
        elif self.config.device == "mps":
            # 显式指定 MPS 时也需要设置 float32
            torch.set_default_dtype(torch.float32)

    def load_model(self, model_size: Optional[str] = None) -> bool:
        """加载Whisper模型"""
        if model_size:
            self.config.size = model_size

        if self.config.size not in self.MODEL_SIZES:
            self.logger.error("不支持的模型大小: %s", self.config.size)
            return False

        try:
            self.logger.info("正在加载 Whisper %s 模型...", self.config.size)
            start_time = time.time()

            # 对于 MPS 设备，先在 CPU 上加载模型再转移，避免 float64 问题
            if self.config.device == "mps":
                self.model = whisper.load_model(
                    self.config.size, device="cpu", download_root="./models"
                )
                # 将模型转换为 float32 后再移至 MPS
                self.model = self.model.float().to("mps")
            else:
                self.model = whisper.load_model(
                    self.config.size,
                    device=self.config.device,
                    download_root="./models",
                )

            load_time = time.time() - start_time
            self.logger.info("模型加载完成，耗时: %.2f秒", load_time)

            return True

        except (
            torch.cuda.CudaError,
            torch.cuda.OutOfMemoryError,
            RuntimeError,
            OSError,
            ValueError,
            ImportError,
        ) as e:
            self.logger.error("模型加载失败: %s", str(e))
            return False

    def transcribe_audio(
        self, audio_file: Path, language: Optional[str] = None
    ) -> TranscriptionResult:
        """转录音频文件"""

        if not self.model:
            if not self.load_model():
                return TranscriptionResult(
                    audio_file=audio_file,
                    text="",
                    segments=[],
                    language="",
                    confidence=0.0,
                    processing_time=0.0,
                    model_used=self.config.size,
                    error_message="模型加载失败",
                    success=False,
                )

        if not audio_file.exists():
            return TranscriptionResult(
                audio_file=audio_file,
                text="",
                segments=[],
                language="",
                confidence=0.0,
                processing_time=0.0,
                model_used=self.config.size,
                error_message=f"音频文件不存在: {audio_file}",
                success=False,
            )

        try:
            self.logger.info("开始转录: %s", audio_file.name)
            start_time = time.time()

            # 构建转录选项
            # 处理语言参数："auto" 或空值表示自动检测，需要设为 None
            effective_language = language or self.config.language
            if effective_language in ("auto", "", None):
                effective_language = None  # Whisper 使用 None 表示自动语言检测

            options = {
                "language": effective_language,
                "temperature": self.config.temperature,
                "best_of": self.config.best_of,
                "beam_size": self.config.beam_size,
                "patience": self.config.patience,
                "length_penalty": self.config.length_penalty,
                "suppress_tokens": self.config.suppress_tokens,
                "initial_prompt": self.config.initial_prompt,
                "condition_on_previous_text": self.config.condition_on_previous_text,
                "compression_ratio_threshold": self.config.compression_ratio_threshold,
                "logprob_threshold": self.config.logprob_threshold,
                "no_speech_threshold": self.config.no_speech_threshold,
                "word_timestamps": self.config.word_timestamps,
                "prepend_punctuations": self.config.prepend_punctuations,
                "append_punctuations": self.config.append_punctuations,
            }

            # MPS 后端不支持 float64；强制关闭 fp16，避免内部走到不兼容的 dtype 分支
            if self.config.device == "mps":
                options["fp16"] = False
                # 强制设置默认类型为 float32
                torch.set_default_dtype(torch.float32)

            # 执行转录
            # 对于 MPS 设备，使用 CPU 进行转录以避免 float64 问题，然后将结果用于后续处理
            if self.config.device == "mps" and self.model is not None:
                # 临时将模型移到 CPU 执行转录，避免 MPS 的 float64 限制
                self.model = self.model.to("cpu")
                result: Any = self.model.transcribe(str(audio_file), **options)
                # 转录完成后将模型移回 MPS
                self.model = self.model.float().to("mps")
            elif self.model is not None:
                result: Any = self.model.transcribe(str(audio_file), **options)
            else:
                # 这个分支理论上不会到达，因为前面已经检查过了
                raise RuntimeError("模型未正确加载")

            processing_time = time.time() - start_time

            # 计算平均置信度（兼容缺失/None 的 avg_logprob）
            segments_list = result.get("segments", [])
            if isinstance(segments_list, list):
                avg_confidence = (
                    sum(
                        float(segment.get("avg_logprob", 0.0) or 0.0)
                        for segment in segments_list
                    )
                    / len(segments_list)
                    if segments_list
                    else 0.0
                )
            else:
                segments_list = []
                avg_confidence = 0.0

            transcription_result = TranscriptionResult(
                audio_file=audio_file,
                text=str(result.get("text", "")),
                segments=cast(List[Dict[str, Union[str, float, int]]], segments_list),
                language=str(result.get("language", "")),
                confidence=avg_confidence,
                processing_time=processing_time,
                model_used=self.config.size,
            )

            self.logger.info(
                "转录完成: %s, 耗时: %.2f秒, 置信度: %.2f",
                audio_file.name,
                processing_time,
                avg_confidence,
            )

            return transcription_result

        except (
            RuntimeError,
            OSError,
            ValueError,
            torch.cuda.CudaError,
            torch.cuda.OutOfMemoryError,
            AttributeError,
            TypeError,
            KeyError,
        ) as e:
            error_msg = f"转录过程中发生错误: {str(e)}"
            self.logger.error(error_msg)
            return TranscriptionResult(
                audio_file=audio_file,
                text="",
                segments=[],
                language="",
                confidence=0.0,
                processing_time=0.0,
                model_used=self.config.size,
                error_message=error_msg,
                success=False,
            )

    def batch_transcribe(
        self, audio_files: List[Path], language: Optional[str] = None
    ) -> List[TranscriptionResult]:
        """批量转录音频文件"""

        if not self.model:
            if not self.load_model():
                return []

        results = []

        for i, audio_file in enumerate(audio_files, 1):
            self.logger.info("处理文件 %d/%d: %s", i, len(audio_files), audio_file.name)

            result = self.transcribe_audio(audio_file, language)
            results.append(result)

            # 添加短暂延迟避免资源竞争
            time.sleep(0.1)

        return results

    def get_model_info(self) -> Dict:
        """获取模型信息"""
        if not self.model:
            return {"error": "模型未加载"}

        return {
            "model_size": self.config.size,
            "device": self.config.device,
            "model_info": self.MODEL_SIZES.get(self.config.size, {}),
            "is_multilingual": hasattr(self.model, "is_multilingual")
            and self.model.is_multilingual,
        }

    def optimize_for_language(self, language: str) -> bool:
        """为特定语言优化配置"""

        # 语言特定的优化设置
        language_optimizations = {
            "zh": {
                "temperature": 0.2,
                "best_of": 3,
                "beam_size": 3,
                "initial_prompt": "以下是普通话语音转录",
            },
            "en": {
                "temperature": 0.0,
                "best_of": 5,
                "beam_size": 5,
                "initial_prompt": "以下是英语语音转录",
            },
            "ja": {
                "temperature": 0.1,
                "best_of": 4,
                "beam_size": 4,
                "initial_prompt": "以下是日语语音转录",
            },
            "ko": {
                "temperature": 0.1,
                "best_of": 4,
                "beam_size": 4,
                "initial_prompt": "以下是韩语语音转录",
            },
        }

        if language in language_optimizations:
            for key, value in language_optimizations[language].items():
                setattr(self.config, key, value)

            self.logger.info("已为语言 '%s' 优化配置", language)
            return True

        self.logger.warning("未找到语言 '%s' 的优化配置", language)
        return False

    def save_config(self, config_path: Path) -> bool:
        """保存配置到文件"""
        try:
            with open(config_path, "w", encoding="utf-8") as f:
                config_dict = {
                    "name": self.config.name,
                    "size": self.config.size,
                    "language": self.config.language,
                    "device": self.config.device,
                    "compute_type": self.config.compute_type,
                    "temperature": self.config.temperature,
                    "best_of": self.config.best_of,
                    "beam_size": self.config.beam_size,
                    "patience": self.config.patience,
                    "length_penalty": self.config.length_penalty,
                    "suppress_tokens": self.config.suppress_tokens,
                    "initial_prompt": self.config.initial_prompt,
                    "condition_on_previous_text": self.config.condition_on_previous_text,
                    "compression_ratio_threshold": self.config.compression_ratio_threshold,
                    "logprob_threshold": self.config.logprob_threshold,
                    "no_speech_threshold": self.config.no_speech_threshold,
                    "word_timestamps": self.config.word_timestamps,
                    "prepend_punctuations": self.config.prepend_punctuations,
                    "append_punctuations": self.config.append_punctuations,
                }
                json.dump(config_dict, f, indent=2, ensure_ascii=False)

            self.logger.info("配置已保存到: %s", config_path)
            return True

        except (IOError, OSError, TypeError, ValueError) as e:
            self.logger.error("保存配置失败: %s", str(e))
            return False

    def load_config(self, config_path: Path) -> bool:
        """从文件加载配置"""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_dict = json.load(f)

            # 更新配置
            for key, value in config_dict.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)

            self.logger.info("配置已从文件加载: %s", config_path)
            return True

        except (
            IOError,
            OSError,
            json.JSONDecodeError,
            TypeError,
            ValueError,
            KeyError,
        ) as e:
            self.logger.error("加载配置失败: %s", str(e))
            return False


def create_model_config(
    model_size: str = "base", language: str = "auto", device: str = "auto"
) -> ModelConfig:
    """创建模型配置的便捷函数"""
    return ModelConfig(
        name=f"whisper_{model_size}", size=model_size, language=language, device=device
    )


if __name__ == "__main__":
    # 测试代码
    config = create_model_config("base")
    whisper_integration = WhisperIntegration(config)

    if whisper_integration.load_model():
        print("模型加载成功")
        print(f"模型信息: {whisper_integration.get_model_info()}")
    else:
        print("模型加载失败")
