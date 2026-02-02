"""Whisper 引擎注册与查询机制。"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Type

from .base_engine import WhisperEngine


@dataclass
class EngineInfo:
    """引擎描述信息。"""

    name: str
    engine_class: Type[WhisperEngine]
    priority: int


class EngineRegistry:
    """引擎注册表，支持按优先级获取可用引擎。"""

    _registry: Dict[str, EngineInfo] = {}

    def __init__(self) -> None:
        self._logger = logging.getLogger("WhisperEngineRegistry")

    @classmethod
    def register_engine(
        cls, name: str, engine_class: Type[WhisperEngine], priority: int = 100
    ) -> None:
        cls._registry[name] = EngineInfo(
            name=name, engine_class=engine_class, priority=priority
        )

    @classmethod
    def get_registered_names(cls) -> List[str]:
        return list(cls._registry.keys())

    def get_engine(self, name: str) -> Optional[WhisperEngine]:
        info = self._registry.get(name)
        if not info:
            return None
        try:
            return info.engine_class()
        except (RuntimeError, OSError, ValueError, ImportError) as exc:
            self._logger.error("初始化引擎失败: %s, %s", name, exc)
            return None

    def get_available_engines(self) -> List[WhisperEngine]:
        engines: List[WhisperEngine] = []
        for info in sorted(self._registry.values(), key=lambda item: item.priority):
            engine = self.get_engine(info.name)
            if engine and engine.is_available():
                engines.append(engine)
        return engines

    def resolve_first_available(
        self, preferred_order: Iterable[str]
    ) -> Optional[WhisperEngine]:
        for name in preferred_order:
            engine = self.get_engine(name)
            if engine and engine.is_available():
                return engine
        return None
