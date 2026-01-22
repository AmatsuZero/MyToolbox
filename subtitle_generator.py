"""
字幕生成模块，提供多种格式字幕的生成、解析和处理功能。

该模块包含用于处理字幕的类和函数，支持SRT、VTT、ASS、TXT、JSON和CSV等多种格式。
主要功能包括：
- 从转录结果创建字幕片段
- 生成不同格式的字幕文件
- 解析和加载现有字幕文件
- 合并多个字幕文件
- 自定义字幕样式

Classes:
    SubtitleSegment: 表示单个字幕片段的数据类
    SubtitleStyle: 定义字幕的样式属性
    SubtitleGenerator: 主要的字幕生成和处理类
"""

import json
import csv
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO, Tuple, Union
from dataclasses import dataclass, field
import logging


@dataclass
class SubtitleSegment:
    """字幕片段类"""

    id: int
    start_time: float  # 秒
    end_time: float  # 秒
    text: str
    speaker: Optional[str] = None
    confidence: Optional[float] = None
    words: Optional[List[Dict]] = None

    def duration(self) -> float:
        """计算片段时长"""
        return self.end_time - self.start_time

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "id": self.id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "text": self.text,
            "speaker": self.speaker,
            "confidence": self.confidence,
            "duration": self.duration(),
            "words": self.words or [],
        }


@dataclass
class FontStyle:
    """字体样式类"""

    font_family: str = "Arial"
    font_size: int = 20
    font_color: str = "#FFFFFF"
    bold: bool = False
    italic: bool = False
    underline: bool = False


@dataclass
class LayoutStyle:
    """布局样式类"""

    alignment: str = "center"  # left, center, right
    position: str = "bottom"  # top, middle, bottom
    margin_vertical: int = 20
    margin_horizontal: int = 20


@dataclass
class BackgroundStyle:
    """背景样式类"""

    color: str = "#00000080"


@dataclass
class SubtitleStyle:
    """字幕样式类"""

    font: FontStyle = field(default_factory=FontStyle)
    layout: LayoutStyle = field(default_factory=LayoutStyle)
    background: BackgroundStyle = field(default_factory=BackgroundStyle)

    def __post_init__(self):
        """初始化后处理"""
        # 已经使用 default_factory 初始化，不需要额外处理
        pass

    # 属性访问器，保持向后兼容性
    @property
    def font_family(self) -> str:
        """获取字体名称"""
        return self.font.font_family

    @font_family.setter
    def font_family(self, value: str):
        """设置字体名称"""
        self.font.font_family = value

    @property
    def font_size(self) -> int:
        """获取字体大小"""
        return self.font.font_size

    @font_size.setter
    def font_size(self, value: int):
        """设置字体大小"""
        self.font.font_size = value

    @property
    def font_color(self) -> str:
        """获取字体颜色"""
        return self.font.font_color

    @font_color.setter
    def font_color(self, value: str):
        """设置字体颜色"""
        self.font.font_color = value

    @property
    def background_color(self) -> str:
        """获取背景颜色"""
        return self.background.color

    @background_color.setter
    def background_color(self, value: str):
        """设置背景颜色"""
        self.background.color = value

    @property
    def alignment(self) -> str:
        """获取文本对齐方式"""
        return self.layout.alignment

    @alignment.setter
    def alignment(self, value: str):
        """设置文本对齐方式"""
        self.layout.alignment = value

    @property
    def position(self) -> str:
        """获取字幕位置"""
        return self.layout.position

    @position.setter
    def position(self, value: str):
        """设置字幕位置"""
        self.layout.position = value

    @property
    def margin_vertical(self) -> int:
        """获取垂直边距"""
        return self.layout.margin_vertical

    @margin_vertical.setter
    def margin_vertical(self, value: int):
        """设置垂直边距"""
        self.layout.margin_vertical = value

    @property
    def margin_horizontal(self) -> int:
        """获取水平边距"""
        return self.layout.margin_horizontal

    @margin_horizontal.setter
    def margin_horizontal(self, value: int):
        """设置水平边距"""
        self.layout.margin_horizontal = value

    @property
    def bold(self) -> bool:
        """获取粗体状态"""
        return self.font.bold

    @bold.setter
    def bold(self, value: bool):
        """设置粗体状态"""
        self.font.bold = value

    @property
    def italic(self) -> bool:
        """获取斜体状态"""
        return self.font.italic

    @italic.setter
    def italic(self, value: bool):
        """设置斜体状态"""
        self.font.italic = value

    @property
    def underline(self) -> bool:
        """获取下划线状态"""
        return self.font.underline

    @underline.setter
    def underline(self, value: bool):
        """设置下划线状态"""
        self.font.underline = value


class SubtitleGenerator:
    """字幕生成器，支持多种字幕格式和高级功能"""

    def __init__(self, output_dir: str = "subtitles"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = self._setup_logger()
        self.default_style = SubtitleStyle()

    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger("SubtitleGenerator")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def format_time_srt(self, seconds: float) -> str:
        """格式化时间为 SRT 格式 (HH:MM:SS,mmm)"""
        # 修复：正确处理超过24小时的时间
        total_seconds = int(seconds)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        millis = int((seconds - total_seconds) * 1000)

        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def format_time_vtt(self, seconds: float) -> str:
        """格式化时间为 VTT 格式 (HH:MM:SS.mmm)"""
        # 修复：正确处理超过24小时的时间
        total_seconds = int(seconds)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        millis = int((seconds - total_seconds) * 1000)

        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

    def format_time_ass(self, seconds: float) -> str:
        """格式化时间为 ASS 格式 (H:MM:SS.xx)"""
        # 修复：使用 total_seconds() 来正确处理超过24小时的时间
        total_seconds = int(seconds)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        centiseconds = int((seconds - total_seconds) * 100)

        return f"{hours}:{minutes:02d}:{secs:02d}.{centiseconds:02d}"

    def create_segments_from_transcription(
        self, transcription_result
    ) -> List[SubtitleSegment]:
        """从转录结果创建字幕片段"""

        segments = []

        # 确保segments不为None，避免迭代None对象
        transcription_segments = transcription_result.segments or []

        for i, segment in enumerate(transcription_segments, 1):
            subtitle_segment = SubtitleSegment(
                id=i,
                start_time=segment["start"],
                end_time=segment["end"],
                text=segment["text"].strip(),
                confidence=segment.get("avg_logprob", None),
                words=segment.get("words", []),
            )
            segments.append(subtitle_segment)

        return segments

    def generate_srt(
        self,
        segments: List[SubtitleSegment],
        output_filename: str,
        encoding: str = "utf-8",
    ) -> Path:
        """生成 SRT 字幕文件"""

        output_path = self.output_dir / output_filename

        try:
            with open(output_path, "w", encoding=encoding) as f:
                for segment in segments:
                    start_time = self.format_time_srt(segment.start_time)
                    end_time = self.format_time_srt(segment.end_time)

                    f.write(f"{segment.id}\n")
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"{segment.text}\n\n")

            self.logger.info("SRT 字幕文件已生成: %s", output_filename)
            return output_path

        except (IOError, OSError) as e:
            self.logger.error("生成 SRT 文件失败: %s", str(e))
            raise

    def generate_vtt(
        self,
        segments: List[SubtitleSegment],
        output_filename: str,
        style: Optional[SubtitleStyle] = None,
        encoding: str = "utf-8",
    ) -> Path:
        """生成 VTT 字幕文件"""

        output_path = self.output_dir / output_filename

        try:
            with open(output_path, "w", encoding=encoding) as f:
                f.write("WEBVTT\n\n")

                # 添加样式
                if style:
                    f.write("STYLE\n")
                    f.write("::cue {\n")
                    f.write(f"  font-family: {style.font_family};\n")
                    f.write(f"  font-size: {style.font_size}px;\n")
                    f.write(f"  color: {style.font_color};\n")
                    f.write(f"  background-color: {style.background_color};\n")
                    f.write(f"  text-align: {style.alignment};\n")
                    f.write("}\n\n")

                for segment in segments:
                    start_time = self.format_time_vtt(segment.start_time)
                    end_time = self.format_time_vtt(segment.end_time)

                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"{segment.text}\n\n")

            self.logger.info("VTT 字幕文件已生成: %s", output_filename)
            return output_path

        except (IOError, OSError) as e:
            self.logger.error("生成 VTT 文件失败: %s", str(e))
            raise

    def _prepare_ass_style(
        self, style: Optional[SubtitleStyle] = None
    ) -> Dict[str, Any]:
        """准备 ASS 样式参数"""
        style = style or self.default_style

        style_params = {
            "style_name": "Default",
            "font_name": style.font_family,
            "font_size": style.font_size,
            "primary_color": style.font_color if style else "&H00FFFFFF",
            "background_color": style.background_color if style else "&H80000000",
            "bold": "-1" if style and style.bold else "0",
            "italic": "-1" if style and style.italic else "0",
            "underline": "-1" if style and style.underline else "0",
            "alignment": self._get_ass_alignment(
                style.alignment if style else "center"
            ),
            "margin_vertical": style.margin_vertical if style else 20,
        }

        # 构建样式行
        style_params["style_line"] = (
            f"Style: {style_params['style_name']},{style_params['font_name']},"
            f"{style_params['font_size']},{style_params['primary_color']},&H000000FF,"
            f"&H00000000,{style_params['background_color']},{style_params['bold']},"
            f"{style_params['italic']},{style_params['underline']},"
            f"0,100,100,0,0,1,2,0,{style_params['alignment']},"
            f"10,10,{style_params['margin_vertical']},1\n"
        )

        return style_params

    def _write_ass_header(self, file_handle: TextIO) -> None:
        """写入 ASS 文件头部和格式定义"""
        # ASS 文件头部
        file_handle.write("[Script Info]\n")
        file_handle.write("ScriptType: v4.00+\n")
        file_handle.write("Collisions: Normal\n")
        file_handle.write("PlayResX: 1920\n")
        file_handle.write("PlayResY: 1080\n")
        file_handle.write("Timer: 100.0000\n")
        file_handle.write("WrapStyle: 0\n")
        file_handle.write("ScaledBorderAndShadow: yes\n")
        file_handle.write("\n")

        # 样式定义
        file_handle.write("[V4+ Styles]\n")
        file_handle.write(
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
            "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
            "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
            "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        )

    def generate_ass(
        self,
        segments: List[SubtitleSegment],
        output_filename: str,
        style: Optional[SubtitleStyle] = None,
        encoding: str = "utf-8",
    ) -> Path:
        """生成 ASS 字幕文件（高级字幕格式）"""

        output_path = self.output_dir / output_filename

        try:
            with open(output_path, "w", encoding=encoding) as f:
                # 写入文件头部
                self._write_ass_header(f)

                # 准备样式参数
                style_params = self._prepare_ass_style(style)

                # 写入样式行
                f.write(style_params["style_line"])
                f.write("\n")

                # 事件
                f.write("[Events]\n")
                f.write(
                    "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, "
                    "Effect, Text\n"
                )

                for segment in segments:
                    start_time = self.format_time_ass(segment.start_time)
                    end_time = self.format_time_ass(segment.end_time)
                    # 转义 ASS 格式中的特殊字符
                    text = self._escape_ass_text(segment.text)

                    f.write(
                        f"Dialogue: 0,{start_time},{end_time},"
                        f"{style_params['style_name']},,0,0,0,,{text}\n"
                    )

            self.logger.info("ASS 字幕文件已生成: %s", output_filename)
            return output_path

        except (IOError, OSError) as e:
            self.logger.error("生成 ASS 文件失败: %s", str(e))
            raise

    def _escape_ass_text(self, text: str) -> str:
        """转义 ASS 格式中的特殊字符"""
        # ASS 格式需要转义的字符
        text = text.replace("\\", "\\\\")  # 反斜杠必须先转义
        text = text.replace("\n", "\\N")  # 换行符
        text = text.replace("{", "\\{")  # 左花括号
        text = text.replace("}", "\\}")  # 右花括号
        return text

    def _get_ass_alignment(self, alignment: str) -> str:
        """获取 ASS 对齐方式"""
        alignment_map = {
            "left": "1",
            "center": "2",
            "right": "3",
            "bottom": "8",
            "middle": "5",
            "top": "2",
        }
        return alignment_map.get(alignment, "2")

    def generate_txt(
        self,
        segments: List[SubtitleSegment],
        output_filename: str,
        include_timestamps: bool = False,
        encoding: str = "utf-8",
    ) -> Path:
        """生成纯文本文件"""

        output_path = self.output_dir / output_filename

        try:
            with open(output_path, "w", encoding=encoding) as f:
                if include_timestamps:
                    for segment in segments:
                        f.write(
                            f"[{segment.start_time:.2f}-{segment.end_time:.2f}] {segment.text}\n"
                        )
                else:
                    for segment in segments:
                        f.write(f"{segment.text}\n")

            self.logger.info("文本文件已生成: %s", output_filename)
            return output_path

        except (IOError, OSError) as e:
            self.logger.error("生成文本文件失败: %s", str(e))
            raise

    def generate_json(
        self,
        segments: List[SubtitleSegment],
        output_filename: str,
        include_metadata: bool = True,
        encoding: str = "utf-8",
    ) -> Path:
        """生成 JSON 格式文件"""

        output_path = self.output_dir / output_filename

        try:
            data: Dict[str, Any] = {
                "segments": [segment.to_dict() for segment in segments]
            }

            if include_metadata:
                data["metadata"] = self._build_metadata(segments)

            with open(output_path, "w", encoding=encoding) as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            self.logger.info("JSON 文件已生成: %s", output_filename)
            return output_path

        except (IOError, OSError, TypeError) as e:
            self.logger.error("生成 JSON 文件失败: %s", str(e))
            raise

    def _build_metadata(
        self, segments: List[SubtitleSegment]
    ) -> Dict[str, Union[int, float]]:
        """构建元数据字典"""
        total_segments = len(segments)
        total_duration = sum(segment.duration() for segment in segments)
        average_confidence = (
            sum(segment.confidence or 0 for segment in segments) / total_segments
            if total_segments
            else 0.0
        )
        average_segment_duration = (
            total_duration / total_segments if total_segments else 0.0
        )
        return {
            "total_segments": total_segments,
            "total_duration": total_duration,
            "average_confidence": average_confidence,
            "average_segment_duration": average_segment_duration,
        }

    def generate_csv(
        self,
        segments: List[SubtitleSegment],
        output_filename: str,
        encoding: str = "utf-8",
    ) -> Path:
        """生成 CSV 格式文件"""

        output_path = self.output_dir / output_filename

        try:
            with open(output_path, "w", encoding=encoding, newline="") as f:
                writer = csv.writer(f)

                # 写入头部
                writer.writerow(
                    [
                        "segment_id",
                        "start_time",
                        "end_time",
                        "duration",
                        "text",
                        "confidence",
                        "speaker",
                    ]
                )

                # 写入数据
                for segment in segments:
                    writer.writerow(
                        [
                            segment.id,
                            segment.start_time,
                            segment.end_time,
                            segment.duration(),
                            segment.text,
                            segment.confidence or "",
                            segment.speaker or "",
                        ]
                    )

            self.logger.info("CSV 文件已生成: %s", output_filename)
            return output_path

        except (IOError, OSError, csv.Error) as e:
            self.logger.error("生成 CSV 文件失败: %s", str(e))
            raise

    def merge_subtitles(
        self,
        subtitle_files: List[Path],
        output_filename: str,
        output_format: str = "srt",
    ) -> Path:
        """合并多个字幕文件"""

        all_segments = []
        current_id = 1

        for subtitle_file in subtitle_files:
            if subtitle_file.suffix == ".json":
                segments = self._load_json_subtitles(subtitle_file)
            elif subtitle_file.suffix == ".srt":
                segments = self._load_srt_subtitles(subtitle_file)
            elif subtitle_file.suffix == ".vtt":
                segments = self._load_vtt_subtitles(subtitle_file)
            else:
                self.logger.warning("不支持的格式: %s", subtitle_file.suffix)
                continue

            for segment in segments:
                segment.id = current_id
                all_segments.append(segment)
                current_id += 1

        # 按时间排序
        all_segments.sort(key=lambda x: x.start_time)

        # 重新编号
        for i, segment in enumerate(all_segments, 1):
            segment.id = i

        # 生成合并后的文件
        if output_format == "srt":
            return self.generate_srt(all_segments, output_filename)
        if output_format == "vtt":
            return self.generate_vtt(all_segments, output_filename)
        if output_format == "json":
            return self.generate_json(all_segments, output_filename)
        return self.generate_txt(all_segments, output_filename)

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
            elif len(parts) == 2:
                # MM:SS.mmm 格式
                minutes = int(parts[0])
                sec_parts = parts[1].split(".")
                seconds = int(sec_parts[0])
                milliseconds = int(sec_parts[1]) if len(sec_parts) > 1 else 0
                return minutes * 60 + seconds + milliseconds / 1000
            else:
                raise ValueError(f"无效的 VTT 时间格式: {time_str}")
        except (ValueError, IndexError) as e:
            raise ValueError(f"无效的 VTT 时间格式: {time_str}") from e

    def _load_srt_subtitles(self, srt_file: Path) -> List[SubtitleSegment]:
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

    def _load_vtt_subtitles(self, vtt_file: Path) -> List[SubtitleSegment]:
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

    def _load_json_subtitles(self, json_file: Path) -> List[SubtitleSegment]:
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

    def create_custom_style(self, **kwargs) -> SubtitleStyle:
        """创建自定义样式"""
        # 将关键字参数分类
        font_attrs = [
            "font_family",
            "font_size",
            "font_color",
            "bold",
            "italic",
            "underline",
        ]
        layout_attrs = ["alignment", "position", "margin_vertical", "margin_horizontal"]
        # 背景属性列表，用于分类处理样式属性
        background_attrs = ["background_color"]

        # 创建样式实例
        style = SubtitleStyle()

        # 分配属性
        for key, value in kwargs.items():
            if key in font_attrs:
                setattr(style, key, value)
            elif key in layout_attrs:
                setattr(style, key, value)
            elif key in background_attrs:
                style.background_color = value
            else:
                # 未知属性，直接设置到 SubtitleStyle 实例
                setattr(style, key, value)

        return style


def create_subtitle_generator(output_dir: str = "subtitles") -> SubtitleGenerator:
    """创建字幕生成器的便捷函数"""
    return SubtitleGenerator(output_dir)


if __name__ == "__main__":
    # 测试代码
    generator = SubtitleGenerator()
    print("字幕生成器已初始化")
