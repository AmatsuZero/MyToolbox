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

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO, Union

from models.subtitle_models import SubtitleSegment, SubtitleStyle
from modules.subtitle_parsers import SubtitleParser





class SubtitleGenerator:
    """字幕生成器，支持多种字幕格式和高级功能"""

    def __init__(self, output_dir: str = "subtitles"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = self._setup_logger()
        self.parser = SubtitleParser(self.logger)
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

    def _load_srt_subtitles(self, srt_file: Path) -> List[SubtitleSegment]:
        """加载并解析 SRT 字幕文件"""
        return self.parser.load_srt_subtitles(srt_file)

    def _load_vtt_subtitles(self, vtt_file: Path) -> List[SubtitleSegment]:
        """加载并解析 VTT 字幕文件"""
        return self.parser.load_vtt_subtitles(vtt_file)

    def _load_json_subtitles(self, json_file: Path) -> List[SubtitleSegment]:
        """加载 JSON 字幕文件"""
        return self.parser.load_json_subtitles(json_file)


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
