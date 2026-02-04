"""
输出生成器模块，用于将语音转录结果转换为多种格式的字幕和文本文件。

支持的输出格式包括：
- SRT 字幕文件
- VTT 网页字幕文件
- 纯文本文件
- JSON 结构化数据
- CSV 表格数据

此模块提供了灵活的输出选项，可以单独生成特定格式或一次性生成所有支持的格式。
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

from modules.whisper.whisper_integration import TranscriptionResult


@dataclass
class SubtitleSegment:
    """字幕片段类"""

    start: float
    end: float
    text: str
    speaker: Optional[str] = None
    confidence: Optional[float] = None


class OutputGenerator:
    """输出生成器，支持多种格式"""

    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger("OutputGenerator")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def format_time(self, seconds: float) -> str:
        """格式化时间戳为 SRT/VTT 格式"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        milliseconds = int((secs - int(secs)) * 1000)

        return f"{hours:02d}:{minutes:02d}:{int(secs):02d}.{milliseconds:03d}"

    def generate_srt(
        self,
        transcription_result: TranscriptionResult,
        output_filename: Optional[str] = None,
    ) -> Path:
        """生成 SRT 字幕文件"""

        if output_filename is None:
            output_filename = f"{transcription_result.audio_file.stem}.srt"

        output_path = self.output_dir / output_filename

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                # 确保segments不为None，避免迭代None对象
                segments = transcription_result.segments or []
                for i, segment in enumerate(segments, 1):
                    start_time = self.format_time(float(segment["start"]))
                    end_time = self.format_time(float(segment["end"]))

                    f.write(f"{i}\n")
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"{str(segment['text']).strip()}\n\n")

            self.logger.info("SRT 字幕文件已生成: %s", output_filename)
            return output_path

        except Exception as e:
            self.logger.error("生成 SRT 文件失败: %s", str(e))
            raise

    def generate_vtt(
        self,
        transcription_result: TranscriptionResult,
        output_filename: Optional[str] = None,
    ) -> Path:
        """生成 VTT 字幕文件"""

        if output_filename is None:
            output_filename = f"{transcription_result.audio_file.stem}.vtt"

        output_path = self.output_dir / output_filename

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("WEBVTT\n\n")

                # 确保segments不为None，避免迭代None对象
                segments = transcription_result.segments or []
                for _, segment in enumerate(segments, 1):
                    start_time = self.format_time(float(segment["start"]))
                    end_time = self.format_time(float(segment["end"]))

                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"{str(segment['text']).strip()}\n\n")

            self.logger.info("VTT 字幕文件已生成: %s", output_filename)
            return output_path

        except Exception as e:
            self.logger.error("生成 VTT 文件失败: %s", str(e))
            raise

    def generate_txt(
        self,
        transcription_result: TranscriptionResult,
        output_filename: Optional[str] = None,
        include_timestamps: bool = False,
    ) -> Path:
        """生成纯文本文件"""

        if output_filename is None:
            output_filename = f"{transcription_result.audio_file.stem}.txt"

        output_path = self.output_dir / output_filename

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                if include_timestamps:
                    # 确保segments不为None，避免迭代None对象
                    segments = transcription_result.segments or []
                    for segment in segments:
                        start = segment["start"]
                        end = segment["end"]
                        text = str(segment["text"]).strip()
                        f.write(f"[{start:.2f}-{end:.2f}] {text}\n")
                else:
                    f.write(transcription_result.text)

            self.logger.info("文本文件已生成: %s", output_filename)
            return output_path

        except Exception as e:
            self.logger.error("生成文本文件失败: %s", str(e))
            raise

    def generate_json(
        self,
        transcription_result: TranscriptionResult,
        output_filename: Optional[str] = None,
    ) -> Path:
        """生成 JSON 格式文件"""

        if output_filename is None:
            output_filename = f"{transcription_result.audio_file.stem}.json"

        output_path = self.output_dir / output_filename

        try:
            # 确保segments不为None，避免迭代None对象
            segments = transcription_result.segments or []

            result_data = {
                "audio_file": str(transcription_result.audio_file),
                "text": transcription_result.text,
                "language": transcription_result.language,
                "confidence": transcription_result.confidence,
                "processing_time": transcription_result.processing_time,
                "model_used": transcription_result.model_used,
                "segments": segments,
                "metadata": {
                    "total_segments": len(segments),
                    "total_duration": (
                        sum(float(seg["end"]) - float(seg["start"]) for seg in segments)
                        if segments
                        else 0.0
                    ),
                    "average_confidence": transcription_result.confidence,
                },
            }

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)

            self.logger.info("JSON 文件已生成: %s", output_filename)
            return output_path

        except Exception as e:
            self.logger.error("生成 JSON 文件失败: %s", str(e))
            raise

    def generate_csv(
        self,
        transcription_result: TranscriptionResult,
        output_filename: Optional[str] = None,
    ) -> Path:
        """生成 CSV 格式文件"""

        if output_filename is None:
            output_filename = f"{transcription_result.audio_file.stem}.csv"

        output_path = self.output_dir / output_filename

        try:
            with open(output_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)

                # 写入头部
                writer.writerow(
                    ["segment_id", "start_time", "end_time", "text", "confidence"]
                )

                # 确保segments不为None，避免迭代None对象
                segments = transcription_result.segments or []

                # 写入数据
                for i, segment in enumerate(segments, 1):
                    writer.writerow(
                        [
                            i,
                            segment["start"],
                            segment["end"],
                            str(segment["text"]).strip(),
                            segment.get("avg_logprob", 0),
                        ]
                    )

            self.logger.info("CSV 文件已生成: %s", output_filename)
            return output_path

        except Exception as e:
            self.logger.error("生成 CSV 文件失败: %s", str(e))
            raise

    def generate_all_formats(
        self, transcription_result: TranscriptionResult, prefix: Optional[str] = None
    ) -> Dict[str, Path]:
        """生成所有支持的格式"""

        if prefix is None:
            prefix = transcription_result.audio_file.stem

        generated_files = {}

        try:
            # 生成 SRT
            srt_file = self.generate_srt(transcription_result, f"{prefix}.srt")
            generated_files["srt"] = srt_file

            # 生成 VTT
            vtt_file = self.generate_vtt(transcription_result, f"{prefix}.vtt")
            generated_files["vtt"] = vtt_file

            # 生成 TXT
            txt_file = self.generate_txt(transcription_result, f"{prefix}.txt")
            generated_files["txt"] = txt_file

            # 生成 JSON
            json_file = self.generate_json(transcription_result, f"{prefix}.json")
            generated_files["json"] = json_file

            # 生成 CSV
            csv_file = self.generate_csv(transcription_result, f"{prefix}.csv")
            generated_files["csv"] = csv_file

            self.logger.info("所有格式文件已生成，前缀: %s", prefix)
            return generated_files

        except (IOError, ValueError, KeyError, TypeError) as e:
            self.logger.error("生成所有格式文件失败: %s", str(e))
            return generated_files

    def create_summary_report(
        self,
        transcription_results: List[TranscriptionResult],
        output_filename: str = "transcription_summary.json",
    ) -> Path:
        """创建批量处理摘要报告"""

        output_path = self.output_dir / output_filename

        try:
            summary = {
                "total_files": len(transcription_results),
                "successful_files": sum(1 for r in transcription_results if r.success),
                "failed_files": sum(1 for r in transcription_results if not r.success),
                "total_processing_time": sum(
                    r.processing_time for r in transcription_results if r.success
                ),
                "average_confidence": sum(
                    r.confidence for r in transcription_results if r.success
                )
                / max(1, sum(1 for r in transcription_results if r.success)),
                "files": [],
            }

            for result in transcription_results:
                file_info = {
                    "filename": result.audio_file.name,
                    "success": result.success,
                    "processing_time": result.processing_time,
                    "confidence": result.confidence,
                    "language": result.language,
                    "model_used": result.model_used,
                    "text_length": len(result.text),
                    "segments_count": len(result.segments),
                }

                if not result.success:
                    file_info["error_message"] = result.error_message

                summary["files"].append(file_info)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            self.logger.info("摘要报告已生成: %s", output_filename)
            return output_path

        except Exception as e:
            self.logger.error("生成摘要报告失败: %s", str(e))
            raise

    def merge_transcriptions(
        self,
        transcription_results: List[TranscriptionResult],
        output_filename: str = "merged_transcription.txt",
    ) -> Path:
        """合并多个转录结果"""

        output_path = self.output_dir / output_filename

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("# 合并转录结果\n\n")

                for i, result in enumerate(transcription_results, 1):
                    f.write(f"## 文件 {i}: {result.audio_file.name}\n")
                    f.write(f"语言: {result.language}\n")
                    f.write(f"置信度: {result.confidence:.2f}\n")
                    f.write(f"处理时间: {result.processing_time:.2f}秒\n\n")
                    f.write(f"{result.text}\n\n")
                    f.write("---\n\n")

            self.logger.info("合并转录结果已生成: %s", output_filename)
            return output_path

        except Exception as e:
            self.logger.error("合并转录结果失败: %s", str(e))
            raise


def create_output_generator(output_dir: str = "output") -> OutputGenerator:
    """创建输出生成器的便捷函数"""
    return OutputGenerator(output_dir)


if __name__ == "__main__":
    # 测试代码
    generator = OutputGenerator()
    print("输出生成器已初始化")
