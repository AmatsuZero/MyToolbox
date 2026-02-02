from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil

from .base_engine import EngineOptions, EngineTranscriptionResult, ProgressCallback
from .engine_registry import EngineRegistry


class FasterWhisperEngine:
    """faster-whisper 引擎实现（跨平台后备）。"""

    name = "faster-whisper"

    def __init__(self) -> None:
        self._logger = logging.getLogger("FasterWhisperEngine")
        self._model: Optional[Any] = None
        self._model_name: Optional[str] = None
        self._options = EngineOptions()
        self._device = "cpu"
        self._compute_type = "int8"

    def configure(self, options: EngineOptions) -> None:
        self._options = options

    def is_available(self) -> bool:
        try:
            import faster_whisper  # pylint: disable=unused-import

            return True
        except ImportError:
            return False

    def get_supported_models(self) -> List[str]:
        return ["tiny", "base", "small", "medium", "large"]

    def load_model(self, model_size: str) -> bool:
        if not self.is_available():
            return False

        try:
            from faster_whisper import WhisperModel

            self._model_name = model_size
            self._device = self._resolve_device()
            self._compute_type = self._resolve_compute_type(self._device)

            kwargs = {
                "device": self._device,
                "compute_type": self._compute_type,
            }

            if self._device == "cpu":
                kwargs["cpu_threads"] = self._optimal_cpu_threads()
                kwargs["num_workers"] = max(1, self._optimal_cpu_threads() // 2)

            self._logger.info(
                "初始化 faster-whisper 模型: %s, device=%s, compute_type=%s",
                model_size,
                self._device,
                self._compute_type,
            )
            self._model = WhisperModel(model_size, **kwargs)
            return True
        except (RuntimeError, OSError, ValueError, ImportError, TypeError) as exc:
            self._logger.error("faster-whisper 模型加载失败: %s", exc)
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
            self._logger.info("使用 faster-whisper 转录: %s", audio_file.name)
            start_time = time.time()

            segments_iter, info = self._model.transcribe(
                str(audio_file),
                language=None if language in (None, "", "auto") else language,
                beam_size=5,
                word_timestamps=True,
            )

            segments = []
            text_fragments = []
            duration = getattr(info, "duration", 0.0) or 0.0

            for segment in segments_iter:
                segment_dict = {
                    "start": float(segment.start),
                    "end": float(segment.end),
                    "text": segment.text,
                }
                if hasattr(segment, "avg_logprob"):
                    segment_dict["avg_logprob"] = float(segment.avg_logprob)
                if hasattr(segment, "no_speech_prob"):
                    segment_dict["no_speech_prob"] = float(segment.no_speech_prob)

                segments.append(segment_dict)
                text_fragments.append(segment.text)

                if progress_callback and duration > 0:
                    progress = min(float(segment.end) / duration, 1.0)
                    progress_callback(progress, "faster-whisper 转录中")

            processing_time = time.time() - start_time
            confidence = self._calculate_confidence(segments)

            if progress_callback:
                progress_callback(1.0, "faster-whisper 转录完成")

            return EngineTranscriptionResult(
                audio_file=audio_file,
                text="".join(text_fragments).strip(),
                segments=segments,
                language=str(getattr(info, "language", "")),
                confidence=confidence,
                processing_time=processing_time,
                model_used=self._model_name or "base",
                engine_name=self.name,
            )

        except (RuntimeError, OSError, ValueError, AttributeError, TypeError) as exc:
            error_msg = f"faster-whisper 转录失败: {exc}"
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
            "compute_type": self._compute_type,
            "model": self._model_name,
        }

    def _resolve_device(self) -> str:
        if self._options.device != "auto":
            return self._options.device

        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except (ImportError, AttributeError):
            pass

        return "cpu"

    def _resolve_compute_type(self, device: str) -> str:
        if self._options.compute_type != "default":
            return self._sanitize_compute_type(self._options.compute_type, device)

        return "float16" if device == "cuda" else "int8"

    def _sanitize_compute_type(self, compute_type: str, device: str) -> str:
        cpu_allowed = {"int8", "float32"}
        cuda_allowed = {"int8", "int8_float16", "float16", "float32"}

        if device == "cuda":
            if compute_type in cuda_allowed:
                return compute_type
            return "float16"

        if compute_type in cpu_allowed:
            return compute_type
        return "int8"

    def _optimal_cpu_threads(self) -> int:
        physical = psutil.cpu_count(logical=False) or os.cpu_count() or 4
        return max(4, physical - 2)

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


EngineRegistry.register_engine("faster-whisper", FasterWhisperEngine, priority=20)
