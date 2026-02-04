"""
音频处理模块，提供音频文件的转换、提取和预处理功能。

该模块使用ffmpeg作为后端，提供了一系列音频处理功能，包括：
- 将各种媒体格式转换为Whisper模型推荐的WAV格式
- 从视频文件中提取音频
- 音频标准化（音量均衡）
- 分割大型音频文件以避免内存溢出
- 临时文件管理

Classes:
    ProcessingResult: 存储处理结果的数据类
    AudioProcessor: 主要的音频处理类，提供各种音频处理功能
"""

import subprocess
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass
import logging

from core.utils.file_scanner import MediaFile


@dataclass
class ProcessingResult:
    """处理结果类"""

    input_file: Path
    output_file: Path
    success: bool
    error_message: Optional[str] = None
    processing_time: Optional[float] = None
    output_size: Optional[int] = None


class AudioProcessor:
    """音频处理器，负责格式转换和预处理"""

    def __init__(self, output_dir: str = "processed_audio"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger("AudioProcessor")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def check_ffmpeg_available(self) -> bool:
        """检查系统是否安装了 ffmpeg"""
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"], capture_output=True, text=True, check=False
            )
            return result.returncode == 0
        except FileNotFoundError:
            self.logger.warning("ffmpeg 未安装，请先安装 ffmpeg")
            return False

    def convert_to_wav(
        self, media_file: MediaFile, output_filename: Optional[str] = None
    ) -> ProcessingResult:
        """将媒体文件转换为 WAV 格式（Whisper 推荐格式）"""

        if not self.check_ffmpeg_available():
            return ProcessingResult(
                input_file=media_file.path,
                output_file=Path(),
                success=False,
                error_message="ffmpeg 未安装",
            )

        # 生成输出文件名
        if output_filename is None:
            output_filename = f"{media_file.path.stem}_converted.wav"

        output_path = self.output_dir / output_filename

        try:
            # 构建 ffmpeg 命令
            cmd = [
                "ffmpeg",
                "-i",
                str(media_file.path),
                "-ac",
                "1",  # 单声道（Whisper 推荐）
                "-ar",
                "16000",  # 16kHz 采样率（Whisper 推荐）
                "-acodec",
                "pcm_s16le",  # 16位 PCM
                "-y",  # 覆盖输出文件
                str(output_path),
            ]

            self.logger.info("正在转换: %s -> %s", media_file.filename, output_filename)

            # 执行转换
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)

            if result.returncode == 0:
                processing_result = ProcessingResult(
                    input_file=media_file.path,
                    output_file=output_path,
                    success=True,
                    output_size=output_path.stat().st_size,
                )
                self.logger.info("转换成功: %s", output_filename)
                return processing_result

            error_msg = f"ffmpeg 转换失败: {result.stderr}"  # 这里保留f-string因为它不是直接用于日志记录
            self.logger.error(error_msg)
            return ProcessingResult(
                input_file=media_file.path,
                output_file=output_path,
                success=False,
                error_message=error_msg,
            )

        except (
            subprocess.SubprocessError,
            FileNotFoundError,
            PermissionError,
            OSError,
            ValueError,
            RuntimeError,
        ) as e:
            error_msg = f"转换过程中发生错误: {str(e)}"
            self.logger.error(error_msg)
            return ProcessingResult(
                input_file=media_file.path,
                output_file=output_path,
                success=False,
                error_message=error_msg,
            )

    def extract_audio_from_video(
        self, media_file: MediaFile, output_filename: Optional[str] = None
    ) -> ProcessingResult:
        """从视频文件中提取音频"""

        if media_file.file_type != "video":
            return ProcessingResult(
                input_file=media_file.path,
                output_file=Path(),
                success=False,
                error_message="输入文件不是视频文件",
            )

        return self.convert_to_wav(media_file, output_filename)

    def normalize_audio(
        self, input_file: Path, target_level: float = -23.0
    ) -> ProcessingResult:
        """音频标准化（音量均衡）"""

        if not self.check_ffmpeg_available():
            return ProcessingResult(
                input_file=input_file,
                output_file=Path(),
                success=False,
                error_message="ffmpeg 未安装",
            )

        output_filename = f"{input_file.stem}_normalized.wav"
        output_path = self.output_dir / output_filename

        try:
            cmd = [
                "ffmpeg",
                "-i",
                str(input_file),
                "-af",
                f"loudnorm=I={target_level}:TP=-1.5:LRA=11:print_format=summary",
                "-y",
                str(output_path),
            ]

            self.logger.info("正在标准化音频: %s", input_file.name)

            result = subprocess.run(cmd, capture_output=True, text=True, check=False)

            if result.returncode == 0:
                processing_result = ProcessingResult(
                    input_file=input_file,
                    output_file=output_path,
                    success=True,
                    output_size=output_path.stat().st_size,
                )
                self.logger.info("音频标准化成功: %s", output_filename)
                return processing_result

            error_msg = f"音频标准化失败: {result.stderr}"
            self.logger.error(error_msg)
            return ProcessingResult(
                input_file=input_file,
                output_file=output_path,
                success=False,
                error_message=error_msg,
            )

        except (
            subprocess.SubprocessError,
            FileNotFoundError,
            PermissionError,
            OSError,
            ValueError,
            RuntimeError,
        ) as e:
            error_msg = f"音频标准化过程中发生错误: {str(e)}"
            self.logger.error(error_msg)
            return ProcessingResult(
                input_file=input_file,
                output_file=output_path,
                success=False,
                error_message=error_msg,
            )

    def split_large_file(
        self, input_file: Path, segment_duration: int = 600
    ) -> List[ProcessingResult]:
        """分割大文件以避免内存溢出"""

        if not self.check_ffmpeg_available():
            return [self._create_error_result(input_file, "ffmpeg 未安装")]

        results = []

        try:
            # 获取文件时长
            duration = self._get_file_duration(input_file)
            if duration == 0:
                return [self._create_error_result(input_file, "无法获取文件时长")]

            # 计算分段数量
            segments = int(duration // segment_duration) + 1
            self.logger.info(
                "将文件分割为 %d 个片段，每段 %d 秒", segments, segment_duration
            )

            # 处理每个分段
            for i in range(segments):
                result = self._process_segment(input_file, i, segment_duration)
                results.append(result)

            return results

        except (
            subprocess.SubprocessError,
            FileNotFoundError,
            PermissionError,
            OSError,
            ValueError,
            RuntimeError,
            TypeError,
        ) as e:
            error_msg = f"文件分割过程中发生错误: {str(e)}"
            self.logger.error(error_msg)
            return [self._create_error_result(input_file, error_msg)]

    def _get_file_duration(self, input_file: Path) -> float:
        """获取音频文件的时长"""
        cmd_duration = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(input_file),
        ]

        result = subprocess.run(
            cmd_duration, capture_output=True, text=True, check=False
        )
        return float(result.stdout.strip()) if result.returncode == 0 else 0

    def _process_segment(
        self, input_file: Path, segment_index: int, segment_duration: int
    ) -> ProcessingResult:
        """处理单个音频分段"""
        start_time = segment_index * segment_duration
        output_filename = f"{input_file.stem}_part{segment_index+1:03d}.wav"
        output_path = self.output_dir / output_filename

        cmd = [
            "ffmpeg",
            "-i",
            str(input_file),
            "-ss",
            str(start_time),
            "-t",
            str(segment_duration),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-acodec",
            "pcm_s16le",
            "-y",
            str(output_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        if result.returncode == 0:
            return ProcessingResult(
                input_file=input_file,
                output_file=output_path,
                success=True,
                output_size=output_path.stat().st_size,
            )

        return ProcessingResult(
            input_file=input_file,
            output_file=output_path,
            success=False,
            error_message=f"分割片段 {segment_index+1} 失败: {result.stderr}",
        )

    def _create_error_result(
        self, input_file: Path, error_message: str
    ) -> ProcessingResult:
        """创建错误结果对象"""
        return ProcessingResult(
            input_file=input_file,
            output_file=Path(),
            success=False,
            error_message=error_message,
        )

    def cleanup_temp_files(self):
        """清理临时文件"""
        # 可以添加临时文件清理逻辑
        pass


def process_media_file(
    media_file: MediaFile, output_dir: str = "processed_audio"
) -> ProcessingResult:
    """处理单个媒体文件的便捷函数"""
    audio_processor = AudioProcessor(output_dir)

    if media_file.file_type == "video":
        return audio_processor.extract_audio_from_video(media_file)
    return audio_processor.convert_to_wav(media_file)


if __name__ == "__main__":
    # 测试代码
    processor = AudioProcessor()
    print(f"ffmpeg 可用: {processor.check_ffmpeg_available()}")
