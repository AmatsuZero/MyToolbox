"""
字幕解析器模块。
"""

import json
import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple

from models.subtitle_models import SubtitleSegment


class SubtitleParser:
    """字幕解析辅助类"""

    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger

    def _parse_srt_time(self, time_str: str) -> float:
        """
        解析 SRT 时间格式 (HH:MM:SS,mmm) 为秒数

        Args:
            time_str: SRT 格式的时间字符串，如 "00:01:23,456"

        Returns:
            转换后的秒数（浮点数）

        Raises:
            ValueError: 时间格式无效时抛出
        """
        time_str = time_str.strip()

        # 支持逗号和点号作为毫秒分隔符
        time_str = time_str.replace(",", ".")

        parts = time_str.split(":")
        if len(parts) != 3:
            raise ValueError(f"无效的 SRT 时间格式: {time_str}")

        try:
            hours = int(parts[0])
            minutes = int(parts[1])
            # 秒和毫秒部分
            sec_parts = parts[2].split(".")
            seconds = int(sec_parts[0])
            milliseconds = int(sec_parts[1]) if len(sec_parts) > 1 else 0

            return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000
        except (ValueError, IndexError) as e:
            raise ValueError(f"无效的 SRT 时间格式: {time_str}") from e

    def _parse_vtt_time(self, time_str: str) -> float:
        """
        解析 VTT 时间格式 (HH:MM:SS.mmm 或 MM:SS.mmm) 为秒数

        Args:
            time_str: VTT 格式的时间字符串，如 "00:01:23.456" 或 "01:23.456"

        Returns:
            转换后的秒数（浮点数）

        Raises:
            ValueError: 时间格式无效时抛出
        """
        time_str = time_str.strip()

        parts = time_str.split(":")

        try:
            if len(parts) == 3:
                # HH:MM:SS.mmm 格式
                hours = int(parts[0])
                minutes = int(parts[1])
                sec_parts = parts[2].split(".")
                seconds = int(sec_parts[0])
                milliseconds = int(sec_parts[1]) if len(sec_parts) > 1 else 0
                return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000
            if len(parts) == 2:
                # MM:SS.mmm 格式
                minutes = int(parts[0])
                sec_parts = parts[1].split(".")
                seconds = int(sec_parts[0])
                milliseconds = int(sec_parts[1]) if len(sec_parts) > 1 else 0
                return minutes * 60 + seconds + milliseconds / 1000
            raise ValueError(f"无效的 VTT 时间格式: {time_str}")
        except (ValueError, IndexError) as e:
            raise ValueError(f"无效的 VTT 时间格式: {time_str}") from e

    def load_srt_subtitles(self, srt_file: Path) -> List[SubtitleSegment]:
        """
        加载并解析 SRT 字幕文件

        SRT 格式示例:
        1
        00:00:01,000 --> 00:00:04,000
        第一段字幕文本

        2
        00:00:05,000 --> 00:00:08,000
        第二段字幕文本
        可以有多行

        Args:
            srt_file: SRT 文件路径

        Returns:
            解析后的字幕片段列表
        """
        try:
            # 读取文件内容
            content = self._read_file_with_multiple_encodings(srt_file)
            self.logger.info("正在解析SRT文件: %s", srt_file)

            # 处理内容并分割成块
            content = self._normalize_line_endings(content)
            blocks = content.strip().split("\n\n")

            # 解析所有字幕块
            segments = self._parse_srt_blocks(blocks)

            self.logger.info("成功解析 %d 个字幕片段", len(segments))
            return segments

        except FileNotFoundError:
            self.logger.error("SRT 文件不存在: %s", srt_file)
            raise
        except PermissionError:
            self.logger.error("没有权限读取 SRT 文件: %s", srt_file)
            raise
        except (IOError, OSError, ValueError, UnicodeError) as e:
            self.logger.error("解析 SRT 文件失败: %s", str(e))
            raise

    def _parse_srt_blocks(self, blocks: List[str]) -> List[SubtitleSegment]:
        """解析SRT字幕块列表"""
        segments = []

        for block in blocks:
            segment = self._parse_srt_block(block)
            if segment:
                segments.append(segment)

        return segments

    def _parse_srt_block(self, block: str) -> Optional[SubtitleSegment]:
        """解析单个SRT字幕块"""
        block = block.strip()
        if not block:
            return None

        lines = block.split("\n")
        if len(lines) < 2:
            self.logger.warning("跳过无效的字幕块: %s...", block[:50])
            return None

        try:
            # 解析字幕ID
            segment_id = self._extract_srt_id(lines[0])
            if segment_id is None:
                return None

            # 解析时间戳
            start_time, end_time = self._extract_srt_timestamps(lines[1])
            if start_time is None or end_time is None:
                return None

            # 提取文本
            text = self._extract_srt_text(lines[2:])
            if not text:
                self.logger.warning("跳过空文本的字幕块，序号: %s", segment_id)
                return None

            return SubtitleSegment(
                id=segment_id,
                start_time=start_time,
                end_time=end_time,
                text=text,
            )

        except ValueError as e:
            self.logger.warning(
                "解析字幕块时出错: %s, 块内容: %s...", str(e), block[:100]
            )
            return None

    def _extract_srt_id(self, id_line: str) -> Optional[int]:
        """从字幕块第一行提取ID"""
        id_line = id_line.strip()
        id_str = "".join(c for c in id_line if c.isdigit())

        if not id_str:
            self.logger.warning("跳过无效序号的字幕块: %s", id_line)
            return None

        return int(id_str)

    def _extract_srt_timestamps(
        self, time_line: str
    ) -> Tuple[Optional[float], Optional[float]]:
        """从字幕块第二行提取开始和结束时间"""
        time_line = time_line.strip()

        if "-->" not in time_line:
            self.logger.warning("跳过无效时间戳的字幕块: %s", time_line)
            return None, None

        time_parts = time_line.split("-->")
        if len(time_parts) != 2:
            self.logger.warning("跳过无效时间戳格式: %s", time_line)
            return None, None

        try:
            start_time = self._parse_srt_time(time_parts[0])
            end_time = self._parse_srt_time(time_parts[1])
            return start_time, end_time
        except ValueError:
            self.logger.warning("解析时间戳出错: %s", time_line)
            return None, None

    def _extract_srt_text(self, text_lines: List[str]) -> str:
        """从字幕块剩余行提取文本"""
        return "\n".join(line.strip() for line in text_lines if line.strip())

    def _read_file_with_multiple_encodings(self, file_path: Path) -> str:
        """尝试使用多种编码读取文件内容"""
        encodings = ["utf-8", "utf-8-sig", "gbk", "gb2312", "latin-1"]

        for encoding in encodings:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue

        raise ValueError(f"无法使用支持的编码格式读取文件: {file_path}")

    def _normalize_line_endings(self, content: str) -> str:
        """标准化文本的换行符"""
        return content.replace("\r\n", "\n").replace("\r", "\n")

    def _find_content_start(self, lines: List[str]) -> int:
        """找到VTT内容的实际起始行"""
        for i, line in enumerate(lines):
            line = line.strip()
            if line.upper().startswith("WEBVTT"):
                continue
            if line.upper().startswith("STYLE") or line.upper().startswith("NOTE"):
                continue
            if "-->" in line or (line and not line.startswith("::cue")):
                return i
        return 0

    def _parse_vtt_block(
        self, block: str, segment_id: int
    ) -> Optional[SubtitleSegment]:
        """解析单个VTT字幕块"""
        # 提前检查无效条件
        block = block.strip()
        if not block or block.upper().startswith(("STYLE", "NOTE")):
            return None

        lines = block.split("\n")

        # 找到时间戳行
        time_line_index = -1
        for idx, line in enumerate(lines):
            if "-->" in line:
                time_line_index = idx
                break

        # 检查时间戳行是否有效
        if time_line_index < 0 or time_line_index >= len(lines):
            self.logger.warning("跳过无效的字幕块（无时间戳）: %s...", block[:50])
            return None

        # 解析时间戳
        time_parts = lines[time_line_index].strip().split("-->")
        if len(time_parts) != 2:
            self.logger.warning(
                "跳过无效时间戳格式: %s", lines[time_line_index].strip()
            )
            return None

        try:
            # 直接解析时间，减少中间变量
            start_time = self._parse_vtt_time(time_parts[0].strip())
            end_time = self._parse_vtt_time(
                time_parts[1].strip().split()[0] if time_parts[1].strip() else ""
            )

            # 提取并清理文本
            text = self._extract_and_clean_text(lines[time_line_index + 1 :])
            if not text:
                self.logger.warning("跳过空文本的字幕块")
                return None

            return SubtitleSegment(
                id=segment_id,
                start_time=start_time,
                end_time=end_time,
                text=text,
            )
        except ValueError as e:
            self.logger.warning("解析时间戳出错: %s", str(e))
            return None

    def _extract_and_clean_text(self, text_lines: List[str]) -> str:
        """提取并清理VTT文本行"""
        cleaned_lines = []
        for line in text_lines:
            # 移除VTT格式标签
            cleaned_line = re.sub(r"<[^>]+>", "", line)
            if cleaned_line.strip():
                cleaned_lines.append(cleaned_line.strip())
        return "\n".join(cleaned_lines)

    def load_vtt_subtitles(self, vtt_file: Path) -> List[SubtitleSegment]:
        """
        加载并解析 VTT 字幕文件

        VTT 格式示例:
        WEBVTT

        00:00:01.000 --> 00:00:04.000
        第一段字幕文本

        00:00:05.000 --> 00:00:08.000
        第二段字幕文本

        Args:
            vtt_file: VTT 文件路径

        Returns:
            解析后的字幕片段列表
        """
        segments: List[SubtitleSegment] = []

        try:
            # 读取并预处理文件内容
            content = self._read_file_with_multiple_encodings(vtt_file)
            self.logger.info("正在解析VTT文件: %s", vtt_file)

            content = self._normalize_line_endings(content)
            lines = content.strip().split("\n")

            # 验证VTT头部
            if not lines or not lines[0].strip().upper().startswith("WEBVTT"):
                self.logger.warning(
                    "文件可能不是有效的 VTT 格式（缺少 WEBVTT 头）: %s", vtt_file
                )

            # 找到内容起始位置并分割字幕块
            content_start = self._find_content_start(lines)
            remaining_content = "\n".join(lines[content_start:])
            blocks = remaining_content.strip().split("\n\n")

            # 解析每个字幕块
            segment_id = 1
            for block in blocks:
                try:
                    segment = self._parse_vtt_block(block, segment_id)
                    if segment:
                        segments.append(segment)
                        segment_id += 1
                except ValueError as e:
                    self.logger.warning(
                        "解析字幕块时出错: %s, 块内容: %s...", str(e), block[:100]
                    )

            self.logger.info("成功解析 %d 个字幕片段", len(segments))

        except FileNotFoundError:
            self.logger.error("VTT 文件不存在: %s", vtt_file)
            raise
        except PermissionError:
            self.logger.error("没有权限读取 VTT 文件: %s", vtt_file)
            raise
        except (IOError, OSError, ValueError, UnicodeError) as e:
            self.logger.error("解析 VTT 文件失败: %s", str(e))
            raise

        return segments

    def load_json_subtitles(self, json_file: Path) -> List[SubtitleSegment]:
        """加载 JSON 字幕文件"""
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            segments = []
            for segment_data in data.get("segments", []):
                segment = SubtitleSegment(
                    id=segment_data.get("id", 0),
                    start_time=segment_data.get("start_time", 0),
                    end_time=segment_data.get("end_time", 0),
                    text=segment_data.get("text", ""),
                    speaker=segment_data.get("speaker"),
                    confidence=segment_data.get("confidence"),
                    words=segment_data.get("words"),
                )
                segments.append(segment)

            return segments

        except FileNotFoundError:
            self.logger.error("JSON 字幕文件不存在: %s", json_file)
            return []
        except PermissionError:
            self.logger.error("没有权限读取 JSON 字幕文件: %s", json_file)
            return []
        except json.JSONDecodeError as e:
            self.logger.error("JSON 字幕文件格式无效: %s, 错误: %s", json_file, str(e))
            return []
        except (KeyError, TypeError) as e:
            self.logger.error("JSON 字幕文件结构无效: %s, 错误: %s", json_file, str(e))
            return []
        except IOError as e:
            self.logger.error(
                "读取 JSON 字幕文件时发生 IO 错误: %s, 错误: %s", json_file, str(e)
            )
            return []
