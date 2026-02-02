from __future__ import annotations

import logging
import platform
from typing import List, Optional

from .base_engine import EngineOptions, WhisperEngine
from .engine_registry import EngineRegistry


class EngineSelector:
    """智能引擎选择器。"""

    ENGINE_PRIORITY = {
        "darwin_arm64": ["whisperkit", "faster-whisper", "openai-whisper"],
        "darwin_x86_64": ["faster-whisper", "openai-whisper"],
        "linux_cuda": ["faster-whisper", "openai-whisper"],
        "linux_cpu": ["faster-whisper", "openai-whisper"],
        "windows_cuda": ["faster-whisper", "openai-whisper"],
        "windows_cpu": ["faster-whisper", "openai-whisper"],
    }

    def __init__(self) -> None:
        self._logger = logging.getLogger("EngineSelector")
        self._registry = EngineRegistry()

    def select_engine(
        self, options: EngineOptions, preferred_engine: Optional[str] = None
    ) -> WhisperEngine:
        order = self._resolve_order(preferred_engine)
        for name in order:
            engine = self._registry.get_engine(name)
            if not engine:
                continue
            engine.configure(options)
            if engine.is_available():
                self._logger.info("选择引擎: %s", name)
                return engine

        available = self._registry.get_registered_names()
        raise RuntimeError(f"未找到可用引擎，可用引擎: {available}")

    def get_fallback_order(self, preferred_engine: Optional[str] = None) -> List[str]:
        return self._resolve_order(preferred_engine)

    def _resolve_order(self, preferred_engine: Optional[str]) -> List[str]:
        if preferred_engine:
            if preferred_engine not in self._registry.get_registered_names():
                raise ValueError(f"未知引擎: {preferred_engine}")
            return [preferred_engine]

        platform_key = self._get_platform_key()
        return self.ENGINE_PRIORITY.get(platform_key, ["openai-whisper"])

    def _get_platform_key(self) -> str:
        system = platform.system().lower()
        machine = platform.machine().lower()

        if system == "darwin":
            return "darwin_arm64" if machine == "arm64" else "darwin_x86_64"

        if system == "linux":
            return "linux_cuda" if self._has_cuda() else "linux_cpu"

        if system == "windows":
            return "windows_cuda" if self._has_cuda() else "windows_cpu"

        return "unknown"

    def _has_cuda(self) -> bool:
        try:
            import torch

            return torch.cuda.is_available()
        except (ImportError, AttributeError):
            return False
