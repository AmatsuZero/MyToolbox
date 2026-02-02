from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import subprocess
import time
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base_engine import EngineOptions, EngineTranscriptionResult, ProgressCallback
from .engine_registry import EngineRegistry


class WhisperKitEngine:
    """WhisperKit 引擎实现（Apple Silicon 优先）。"""

    name = "whisperkit"

    def __init__(self) -> None:
        self._logger = logging.getLogger("WhisperKitEngine")
        self._model_name: Optional[str] = None
        self._options = EngineOptions()
        self._cli_path: Optional[str] = None

    def configure(self, options: EngineOptions) -> None:
        self._options = options
        if options.device not in ("auto", ""):
            self._logger.warning(
                "WhisperKit CLI 当前不支持 device 选项，已忽略: %s", options.device
            )
        if options.compute_type not in ("default", ""):
            self._logger.warning(
                "WhisperKit CLI 当前不支持 compute_type 选项，已忽略: %s",
                options.compute_type,
            )

    def is_available(self) -> bool:
        if platform.system().lower() != "darwin":
            return False
        if platform.machine().lower() != "arm64":
            return False

        mac_ver = platform.mac_ver()[0]
        if mac_ver:
            major_version = int(mac_ver.split(".")[0])
            if major_version < 14:
                self._logger.warning("WhisperKit 需要 macOS 14+，当前: %s", mac_ver)
                return False

        self._cli_path = self._find_cli()
        if not self._cli_path:
            self._logger.warning("未找到 whisperkit-cli，请先安装 WhisperKit CLI")
            return False

        return True

    def get_supported_models(self) -> List[str]:
        return ["tiny", "base", "small", "medium", "large"]

    def load_model(self, model_size: str) -> bool:
        if not self.is_available():
            return False

        try:
            resolved_name = self._resolve_model_name(model_size)
            self._logger.info("准备 WhisperKit 模型: %s", resolved_name)
            self._model_name = model_size
            return True
        except (RuntimeError, OSError, ValueError, TypeError) as exc:
            self._logger.error("WhisperKit 模型准备失败: %s", exc)
            self._model_name = None
            return False

    def transcribe(
        self,
        audio_file: Path,
        language: Optional[str],
        progress_callback: Optional[ProgressCallback] = None,
    ) -> EngineTranscriptionResult:
        if not self._model_name:
            if not self.load_model("base"):
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
            self._logger.info("使用 WhisperKit CLI 转录: %s", audio_file.name)
            start_time = time.time()

            if progress_callback:
                progress_callback(0.0, "WhisperKit 开始转录")

            resolved_name = self._resolve_model_name(self._model_name or "base")
            result = self._run_cli_transcribe(
                audio_file,
                resolved_name,
                None if language in (None, "", "auto") else language,
            )

            processing_time = time.time() - start_time
            segments = result.get("segments", []) if isinstance(result, dict) else []
            confidence = self._calculate_confidence(segments)
            text = result.get("text", "") if isinstance(result, dict) else ""
            detected_language = (
                result.get("language", "") if isinstance(result, dict) else ""
            )
            if not detected_language and language not in (None, "", "auto"):
                detected_language = language or ""

            if progress_callback:
                progress_callback(1.0, "WhisperKit 转录完成")

            return EngineTranscriptionResult(
                audio_file=audio_file,
                text=str(text),
                segments=segments,
                language=str(detected_language),
                confidence=confidence,
                processing_time=processing_time,
                model_used=self._model_name or "base",
                engine_name=self.name,
            )

        except (RuntimeError, OSError, ValueError, KeyError, TypeError) as exc:
            error_msg = f"WhisperKit 转录失败: {exc}"
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
            "platform": platform.system(),
            "machine": platform.machine(),
            "model": self._model_name,
        }

    def _resolve_model_name(self, model_size: str) -> str:
        if model_size == "large":
            return "large-v3"
        return model_size

    def _find_cli(self) -> Optional[str]:
        env_path = os.environ.get("WHISPERKIT_CLI_PATH") or os.environ.get(
            "WHISPERKIT_CLI"
        )
        if env_path:
            candidate = Path(env_path).expanduser()
            if candidate.exists():
                return str(candidate)
        return shutil.which("whisperkit-cli")

    def _run_cli_transcribe(
        self,
        audio_file: Path,
        model_name: str,
        language: Optional[str],
    ) -> Dict[str, Any]:
        cli_path = self._cli_path or self._find_cli() or "whisperkit-cli"
        with tempfile.TemporaryDirectory(prefix="whisperkit_report_") as report_dir:
            command = [
                cli_path,
                "transcribe",
                "--model",
                model_name,
                "--audio-path",
                str(audio_file),
                "--report",
                "--report-path",
                report_dir,
            ]
            if language:
                command.extend(["--language", language])

            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
            )

            stdout = (completed.stdout or "").strip()
            stderr = (completed.stderr or "").strip()

            if completed.returncode != 0:
                raise RuntimeError(stderr or stdout or "WhisperKit CLI 执行失败")

            report_data = self._load_report_data(Path(report_dir), audio_file)
            if report_data:
                return report_data

            if not stdout:
                if stderr:
                    raise RuntimeError(stderr)
                return {"text": "", "segments": [], "language": ""}

            if stdout.lstrip().startswith("{") or stdout.lstrip().startswith("["):
                try:
                    data = json.loads(stdout)
                except json.JSONDecodeError:
                    return {"text": stdout, "segments": [], "language": ""}
                if isinstance(data, list):
                    return {"text": "".join(str(item) for item in data), "segments": data}
                if isinstance(data, dict):
                    return data

            return {"text": stdout, "segments": [], "language": ""}

    def _load_report_data(
        self, report_dir: Path, audio_file: Path
    ) -> Optional[Dict[str, Any]]:
        if not report_dir.exists():
            return None
        report_files = sorted(
            report_dir.glob("*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not report_files:
            return None
        candidates = [path for path in report_files if audio_file.stem in path.name]
        report_path = candidates[0] if candidates else report_files[0]
        try:
            with open(report_path, "r", encoding="utf-8") as handle:
                report_json = json.load(handle)
        except (OSError, json.JSONDecodeError, ValueError):
            return None
        return self._normalize_report_data(report_json)

    def _normalize_report_data(self, report_json: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(report_json, dict):
            return None
        segments = self._extract_segments(report_json)
        text = self._extract_text(report_json, segments)
        language = self._extract_language(report_json)
        if not segments and not text:
            return None
        return {
            "text": text,
            "segments": segments,
            "language": language,
        }

    def _extract_segments(self, report_json: Dict[str, Any]) -> List[Dict[str, Any]]:
        segment_candidates: List[Dict[str, Any]] = []
        for key in ("segments", "result", "transcription", "data", "output"):
            value = report_json.get(key)
            if isinstance(value, dict) and isinstance(value.get("segments"), list):
                segment_candidates = value.get("segments", [])
                break
            if isinstance(value, list):
                segment_candidates = value
                break
        if not segment_candidates and isinstance(report_json.get("chunks"), list):
            segment_candidates = report_json.get("chunks", [])
        return self._normalize_segments(segment_candidates)

    def _normalize_segments(
        self, segments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            start = segment.get("start")
            end = segment.get("end")
            if start is None and "start_time" in segment:
                start = segment.get("start_time")
            if end is None and "end_time" in segment:
                end = segment.get("end_time")
            if start is None or end is None:
                continue
            text = (
                segment.get("text")
                or segment.get("transcript")
                or segment.get("sentence")
                or ""
            )
            normalized_segment: Dict[str, Any] = {
                "start": float(start),
                "end": float(end),
                "text": str(text),
            }
            if "avg_logprob" in segment:
                normalized_segment["avg_logprob"] = segment.get("avg_logprob")
            if "words" in segment:
                normalized_segment["words"] = segment.get("words")
            normalized.append(normalized_segment)
        return normalized

    def _extract_text(
        self, report_json: Dict[str, Any], segments: List[Dict[str, Any]]
    ) -> str:
        for key in ("text", "transcript", "transcription"):
            value = report_json.get(key)
            if isinstance(value, str) and value.strip():
                return value
            if isinstance(value, dict) and isinstance(value.get("text"), str):
                return str(value.get("text"))
        if segments:
            return "".join(segment.get("text", "") for segment in segments).strip()
        return ""

    def _extract_language(self, report_json: Dict[str, Any]) -> str:
        for key in ("language", "lang"):
            value = report_json.get(key)
            if isinstance(value, str):
                return value
        if isinstance(report_json.get("result"), dict):
            value = report_json.get("result", {}).get("language")
            if isinstance(value, str):
                return value
        return ""

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


EngineRegistry.register_engine("whisperkit", WhisperKitEngine, priority=10)