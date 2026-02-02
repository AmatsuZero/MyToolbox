"""Whisper 引擎抽象层定义。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Union

ProgressCallback = Callable[[float, str], None]


@dataclass
class EngineOptions:
    """引擎配置选项。"""

    device: str = "auto"
    compute_type: str = "default"
    preferred_engine: Optional[str] = None


class WhisperEngine(Protocol):
    """Whisper 引擎接口协议。"""

    name: str

    def configure(self, options: EngineOptions) -> None:
        """配置引擎选项。"""

    def is_available(self) -> bool:
        """检查引擎是否可用。"""

    def load_model(self, model_size: str) -> bool:
        """加载指定大小模型。"""

    def get_supported_models(self) -> List[str]:
        """返回支持的模型列表。"""

    def transcribe(
        self,
        audio_file: Path,
        language: Optional[str],
        progress_callback: Optional[ProgressCallback] = None,
    ) -> "EngineTranscriptionResult":
        """转录音频文件。"""

    def get_device_info(self) -> Dict[str, Any]:
        """返回设备信息。"""


@dataclass
class EngineTranscriptionResult:  # pylint: disable=too-many-instance-attributes
    """统一引擎转录结果。"""

    audio_file: Path
    text: str
    segments: List[Dict[str, Union[str, float, int]]]
    language: str
    confidence: float
    processing_time: float
    model_used: str
    engine_name: str
    error_message: Optional[str] = None
    success: bool = True
