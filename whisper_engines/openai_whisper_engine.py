from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import whisper

from .base_engine import EngineOptions, EngineTranscriptionResult, ProgressCallback
from .engine_registry import EngineRegistry


def _patch_mps_float64() -> None:
    if not torch.backends.mps.is_available():
        return

    _original_to = torch.Tensor.to

    def _patched_to(self, *args, **kwargs):
        device = None
        dtype = kwargs.get("dtype", None)

        if args:
            if isinstance(args[0], torch.device):
                device = args[0]
            elif isinstance(args[0], str):
                device = torch.device(args[0])
            elif isinstance(args[0], torch.dtype):
                dtype = args[0]

        is_to_mps = (device is not None and device.type == "mps") or (
            device is None and self.device.type == "mps"
        )

        if is_to_mps and torch.float64 in (self.dtype, dtype):
            self = self.float()
            if dtype == torch.float64:
                kwargs["dtype"] = torch.float32

        return _original_to(self, *args, **kwargs)

    torch.Tensor.to = _patched_to

    _original_from_numpy = torch.from_numpy

    def _patched_from_numpy(ndarray):
        tensor = _original_from_numpy(ndarray)
        if tensor.dtype == torch.float64:
            tensor = tensor.float()
        return tensor

    torch.from_numpy = _patched_from_numpy


class OpenAIWhisperEngine:
    """OpenAI Whisper 引擎实现。"""

    name = "openai-whisper"

    def __init__(self) -> None:
        self._logger = logging.getLogger("OpenAIWhisperEngine")
        self._model: Optional[Any] = None
        self._model_name: Optional[str] = None
        self._options = EngineOptions()
        self._device = "cpu"
        _patch_mps_float64()

    def configure(self, options: EngineOptions) -> None:
        self._options = options

    def is_available(self) -> bool:
        try:
            import whisper  # pylint: disable=unused-import

            return True
        except ImportError:
            return False

    def get_supported_models(self) -> List[str]:
        return ["tiny", "base", "small", "medium", "large"]

    def load_model(self, model_size: str) -> bool:
        if not self.is_available():
            return False

        try:
            self._model_name = model_size
            self._device = self._resolve_device()

            self._logger.info(
                "初始化 OpenAI Whisper 模型: %s, device=%s",
                model_size,
                self._device,
            )

            if self._device == "mps":
                self._model = whisper.load_model(
                    model_size, device="cpu", download_root="./models"
                )
                self._model = self._model.float().to("mps")
            else:
                self._model = whisper.load_model(
                    model_size, device=self._device, download_root="./models"
                )

            return True
        except (RuntimeError, OSError, ValueError, ImportError, TypeError) as exc:
            self._logger.error("OpenAI Whisper 模型加载失败: %s", exc)
            self._model = None
            return False

    def transcribe(
        self,
        audio_file: Path,
        language: Optional[str],
        progress_callback: Optional[ProgressCallback] = None,
    ) -> EngineTranscriptionResult:
        if not self._model:
            if not self.load_model(self._model_name or "base"):
                return EngineTranscriptionResult(
                    audio_file=audio_file,
                    text="",
                    segments=[],
                    language="",
                    confidence=0.0,
                    processing_time=0.0,
                    model_used=self._model_name or "base",
                    engine_name=self.name,
                    error_message="模型加载失败",
                    success=False,
                )

        if not audio_file.exists():
            return EngineTranscriptionResult(
                audio_file=audio_file,
                text="",
                segments=[],
                language="",
                confidence=0.0,
                processing_time=0.0,
                model_used=self._model_name or "base",
                engine_name=self.name,
                error_message=f"音频文件不存在: {audio_file}",
                success=False,
            )

        try:
            self._logger.info("使用 OpenAI Whisper 转录: %s", audio_file.name)
            start_time = time.time()

            effective_language = None if language in (None, "", "auto") else language
            options = {
                "language": effective_language,
                "word_timestamps": True,
            }

            if self._device == "mps":
                options["fp16"] = False
                torch.set_default_dtype(torch.float32)

            if self._device == "mps":
                self._model = self._model.to("cpu")
                result: Any = self._model.transcribe(str(audio_file), **options)
                self._model = self._model.float().to("mps")
            else:
                result = self._model.transcribe(str(audio_file), **options)

            processing_time = time.time() - start_time

            segments_list = result.get("segments", []) if isinstance(result, dict) else []
            confidence = self._calculate_confidence(segments_list)

            if progress_callback:
                progress_callback(1.0, "OpenAI Whisper 转录完成")

            return EngineTranscriptionResult(
                audio_file=audio_file,
                text=str(result.get("text", "")),
                segments=segments_list,
                language=str(result.get("language", "")),
                confidence=confidence,
                processing_time=processing_time,
                model_used=self._model_name or "base",
                engine_name=self.name,
            )

        except (RuntimeError, OSError, ValueError, AttributeError, TypeError) as exc:
            error_msg = f"OpenAI Whisper 转录失败: {exc}"
            self._logger.error(error_msg)
            return EngineTranscriptionResult(
                audio_file=audio_file,
                text="",
                segments=[],
                language="",
                confidence=0.0,
                processing_time=0.0,
                model_used=self._model_name or "base",
                engine_name=self.name,
                error_message=error_msg,
                success=False,
            )

    def get_device_info(self) -> Dict[str, Any]:
        return {
            "engine": self.name,
            "device": self._device,
            "model": self._model_name,
        }

    def _resolve_device(self) -> str:
        if self._options.device != "auto":
            return self._options.device

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _calculate_confidence(self, segments: List[Dict[str, Any]]) -> float:
        if not segments:
            return 0.0
        total = 0.0
        count = 0
        for segment in segments:
            value = segment.get("avg_logprob")
            if value is None:
                continue
            total += float(value)
            count += 1
        return total / count if count else 0.0


EngineRegistry.register_engine("openai-whisper", OpenAIWhisperEngine, priority=30)
