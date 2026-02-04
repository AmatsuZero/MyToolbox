"""
音视频格式转换模块 - 提供多种音视频格式转换和处理功能

该模块基于ffmpeg实现了音视频文件的格式转换、处理和优化功能，主要包括：
- 从视频文件中提取音频
- 音频格式转换（支持多种常见格式如MP3、WAV、FLAC等）
- 音频标准化处理（音量均衡）
- 批量处理媒体文件
- 针对语音识别的音频优化（采样率、声道数等）
"""

import logging
import subprocess
import tempfile
import time
import json
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
import shutil

from core.utils.file_scanner import MediaFile


@dataclass
class ConversionResult:
    """格式转换结果类"""

    # pylint: disable=too-many-instance-attributes

    input_file: Path
    output_file: Path
    success: bool
    conversion_type: str  # 'audio_extract', 'format_convert', 'normalize'
    processing_time: float
    input_size: int
    output_size: int
    compression_ratio: float
    error_message: Optional[str] = None


class FormatConverter:
    """格式转换器，支持多种音视频格式转换"""

    # 支持的输入格式
    SUPPORTED_INPUT_FORMATS = {
        # 音频格式
        ".mp3",
        ".wav",
        ".flac",
        ".aac",
        ".ogg",
        ".m4a",
        ".wma",
        ".opus",
        # 视频格式
        ".mp4",
        ".webm",
        ".mkv",
        ".avi",
        ".mov",
        ".wmv",
        ".flv",
        ".m4v",
        ".3gp",
    }

    # 支持的输出格式
    SUPPORTED_OUTPUT_FORMATS = {
        # 音频格式（Whisper 推荐）
        ".wav",
        ".mp3",
        ".flac",
        # 字幕格式
        ".srt",
        ".vtt",
        ".txt",
    }

    # 推荐的音频参数（Whisper 优化）
    WHISPER_AUDIO_PARAMS = {
        "sample_rate": 16000,  # 16kHz
        "channels": 1,  # 单声道
        "codec": "pcm_s16le",  # 16位 PCM
        "bitrate": "128k",
    }

    def __init__(self, output_dir: str = "converted_audio"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.temp_dir = Path(tempfile.mkdtemp())
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger("FormatConverter")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _create_success_result(
        self,
        input_file: Path,
        output_file: Path,
        conversion_type: str,
        processing_info: Dict,
    ) -> ConversionResult:
        """创建成功的转换结果对象的辅助方法

        Args:
            input_file: 输入文件路径
            output_file: 输出文件路径
            conversion_type: 转换类型
            processing_info: 包含处理信息的字典，需要包含以下键:
                - processing_time: 处理时间
                - input_size: 输入文件大小
                - output_size: 输出文件大小
                - compression_ratio: 压缩比
        """
        return ConversionResult(
            input_file=input_file,
            output_file=output_file,
            success=True,
            conversion_type=conversion_type,
            processing_time=processing_info["processing_time"],
            input_size=processing_info["input_size"],
            output_size=processing_info["output_size"],
            compression_ratio=processing_info["compression_ratio"],
            error_message=None,
        )

    def _create_error_result(
        self,
        input_file: Path,
        output_file: Path,
        conversion_type: str,
        processing_time: float,
        input_size: int,
        output_size: int,
        error_message: str,
    ) -> ConversionResult:
        """创建失败的转换结果对象的辅助方法"""
        return ConversionResult(
            input_file=input_file,
            output_file=output_file,
            success=False,
            conversion_type=conversion_type,
            processing_time=processing_time,
            input_size=input_size,
            output_size=output_size,
            compression_ratio=0.0,
            error_message=error_message,
        )

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

    def get_file_info(self, file_path: Path) -> Dict:
        """获取媒体文件信息"""
        if not self.check_ffmpeg_available():
            return {"error": "ffmpeg 未安装"}

        try:
            cmd = [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=codec_name,sample_rate,channels,duration,bit_rate",
                "-show_entries",
                "format=duration,size,bit_rate",
                "-of",
                "json",
                str(file_path),
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, check=False)

            if result.returncode == 0:
                info = json.loads(result.stdout)
                return info
            else:
                return {"error": f"获取文件信息失败: {result.stderr}"}

        except (
            subprocess.SubprocessError,
            FileNotFoundError,
            json.JSONDecodeError,
            OSError,
        ) as e:
            return {"error": f"获取文件信息时发生错误: {e}"}

    def extract_audio_from_video(
        self, video_file: Path, output_format: str = ".wav"
    ) -> ConversionResult:
        """从视频文件中提取音频"""
        # 检查前置条件
        if not self.check_ffmpeg_available():
            return self._create_error_result(
                input_file=video_file,
                output_file=Path(),
                conversion_type="audio_extract",
                processing_time=0.0,
                input_size=0,
                output_size=0,
                error_message="ffmpeg 未安装",
            )

        if output_format not in self.SUPPORTED_OUTPUT_FORMATS:
            return self._create_error_result(
                input_file=video_file,
                output_file=Path(),
                conversion_type="audio_extract",
                processing_time=0.0,
                input_size=0,
                output_size=0,
                error_message=f"不支持的输出格式: {output_format}",
            )

        output_filename = f"{video_file.stem}_audio{output_format}"
        output_path = self.output_dir / output_filename

        try:
            start_time = time.time()

            # 提取音频的核心处理逻辑
            return self._process_audio_extraction(video_file, output_path, start_time)

        except (
            subprocess.SubprocessError,
            FileNotFoundError,
            PermissionError,
            OSError,
        ) as e:
            error_message = f"音频提取过程中发生错误: {e}"
            self.logger.error("音频提取过程中发生错误: %s", e)
            return self._create_error_result(
                input_file=video_file,
                output_file=output_path,
                conversion_type="audio_extract",
                processing_time=0.0,
                input_size=video_file.stat().st_size if video_file.exists() else 0,
                output_size=output_path.stat().st_size if output_path.exists() else 0,
                error_message=error_message,
            )

    def _process_audio_extraction(
        self, video_file: Path, output_path: Path, start_time: float
    ) -> ConversionResult:
        """处理音频提取的核心逻辑"""
        # 构建 ffmpeg 命令
        cmd = [
            "ffmpeg",
            "-i",
            str(video_file),
            "-vn",  # 不处理视频流
            "-ac",
            str(self.WHISPER_AUDIO_PARAMS["channels"]),
            "-ar",
            str(self.WHISPER_AUDIO_PARAMS["sample_rate"]),
            "-acodec",
            self.WHISPER_AUDIO_PARAMS["codec"],
            "-y",  # 覆盖输出文件
            str(output_path),
        ]

        self.logger.info(
            "正在从视频提取音频: %s -> %s", video_file.name, output_path.name
        )

        # 执行转换
        try:
            # 设置超时时间，避免进程挂起
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=300,  # 5分钟超时，可根据实际需求调整
            )
        except subprocess.TimeoutExpired:
            self.logger.error("音频提取操作超时")
            elapsed_time = time.time() - start_time
            return self._create_error_result(
                input_file=video_file,
                output_file=output_path,
                conversion_type="audio_extract",
                processing_time=elapsed_time,
                input_size=video_file.stat().st_size if video_file.exists() else 0,
                output_size=0,
                error_message="音频提取操作超时（5分钟）",
            )

        processing_time = time.time() - start_time

        # 检查输出文件是否存在且大小合理
        if (
            result.returncode == 0
            and output_path.exists()
            and output_path.stat().st_size > 0
        ):
            # 获取文件大小信息并创建成功结果对象
            input_size = video_file.stat().st_size
            output_size = output_path.stat().st_size
            compression_ratio = output_size / input_size if input_size > 0 else 0

            processing_info = {
                "processing_time": processing_time,
                "input_size": input_size,
                "output_size": output_size,
                "compression_ratio": compression_ratio,
            }

            # 记录成功日志
            self.logger.info(
                "音频提取成功: %s, 耗时: %.2f秒, 压缩比: %.2f",
                output_path.name,
                processing_time,
                compression_ratio,
            )

            # 返回成功结果
            return self._create_success_result(
                input_file=video_file,
                output_file=output_path,
                conversion_type="audio_extract",
                processing_info=processing_info,
            )

        # 处理失败情况（ffmpeg返回非零状态码）
        error_message = f"音频提取失败: {result.stderr}"
        self.logger.error("音频提取失败: %s", result.stderr)

        # 清理不完整的输出文件
        self._cleanup_incomplete_file(output_path)

        # 返回失败结果
        return self._create_error_result(
            input_file=video_file,
            output_file=output_path,
            conversion_type="audio_extract",
            processing_time=processing_time,
            input_size=video_file.stat().st_size if video_file.exists() else 0,
            output_size=output_path.stat().st_size if output_path.exists() else 0,
            error_message=error_message,
        )

    def _cleanup_incomplete_file(self, file_path: Path) -> None:
        """清理不完整的输出文件"""
        if file_path.exists() and file_path.stat().st_size == 0:
            try:
                file_path.unlink()
                self.logger.info("已删除不完整的输出文件: %s", file_path)
            except OSError as e:
                self.logger.warning(
                    "无法删除不完整的输出文件: %s, 错误: %s", file_path, e
                )

    def convert_audio_format(
        self, audio_file: Path, output_format: str = ".wav"
    ) -> ConversionResult:
        """转换音频格式"""

        if not self.check_ffmpeg_available():
            return self._create_error_result(
                input_file=audio_file,
                output_file=Path(),
                conversion_type="format_convert",
                processing_time=0.0,
                input_size=0,
                output_size=0,
                error_message="ffmpeg 未安装",
            )

        if output_format not in self.SUPPORTED_OUTPUT_FORMATS:
            return self._create_error_result(
                input_file=audio_file,
                output_file=Path(),
                conversion_type="format_convert",
                processing_time=0.0,
                input_size=0,
                output_size=0,
                error_message=f"不支持的输出格式: {output_format}",
            )

        output_filename = f"{audio_file.stem}_converted{output_format}"
        output_path = self.output_dir / output_filename

        try:
            start_time = time.time()

            # 构建 ffmpeg 命令
            cmd = [
                "ffmpeg",
                "-i",
                str(audio_file),
                "-ac",
                str(self.WHISPER_AUDIO_PARAMS["channels"]),
                "-ar",
                str(self.WHISPER_AUDIO_PARAMS["sample_rate"]),
                "-acodec",
                self.WHISPER_AUDIO_PARAMS["codec"],
                "-y",
                str(output_path),
            ]

            self.logger.info(
                "正在转换音频格式: %s -> %s", audio_file.name, output_filename
            )

            # 执行转换
            try:
                # 设置超时时间，避免进程挂起
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=300,  # 5分钟超时，可根据实际需求调整
                )
            except subprocess.TimeoutExpired:
                self.logger.error("音频格式转换操作超时")
                return self._create_error_result(
                    input_file=audio_file,
                    output_file=output_path,
                    conversion_type="format_convert",
                    processing_time=time.time() - start_time,
                    input_size=audio_file.stat().st_size if audio_file.exists() else 0,
                    output_size=0,
                    error_message="音频格式转换操作超时（5分钟）",
                )

            processing_time = time.time() - start_time

            # 检查输出文件是否存在且大小合理
            if (
                result.returncode == 0
                and output_path.exists()
                and output_path.stat().st_size > 0
            ):
                # 获取文件大小信息并创建成功结果对象
                input_size = audio_file.stat().st_size
                output_size = output_path.stat().st_size
                compression_ratio = output_size / input_size if input_size > 0 else 0

                processing_info = {
                    "processing_time": processing_time,
                    "input_size": input_size,
                    "output_size": output_size,
                    "compression_ratio": compression_ratio,
                }

                # 记录成功日志
                self.logger.info(
                    "音频格式转换成功: %s, 耗时: %.2f秒, 压缩比: %.2f",
                    output_filename,
                    processing_time,
                    compression_ratio,
                )

                # 返回成功结果
                return self._create_success_result(
                    input_file=audio_file,
                    output_file=output_path,
                    conversion_type="format_convert",
                    processing_info=processing_info,
                )

            # 处理失败情况（ffmpeg返回非零状态码）
            error_message = f"音频格式转换失败: {result.stderr}"
            self.logger.error("音频格式转换失败: %s", result.stderr)

            # 清理不完整的输出文件
            self._cleanup_incomplete_file(output_path)

            # 返回失败结果
            return self._create_error_result(
                input_file=audio_file,
                output_file=output_path,
                conversion_type="format_convert",
                processing_time=processing_time,
                input_size=audio_file.stat().st_size if audio_file.exists() else 0,
                output_size=output_path.stat().st_size if output_path.exists() else 0,
                error_message=error_message,
            )

        except (
            subprocess.SubprocessError,
            FileNotFoundError,
            PermissionError,
            OSError,
        ) as e:
            error_message = f"音频格式转换过程中发生错误: {e}"
            self.logger.error("音频格式转换过程中发生错误: %s", e)
            return self._create_error_result(
                input_file=audio_file,
                output_file=output_path,
                conversion_type="format_convert",
                processing_time=0.0,
                input_size=audio_file.stat().st_size if audio_file.exists() else 0,
                output_size=output_path.stat().st_size if output_path.exists() else 0,
                error_message=error_message,
            )

    def normalize_audio(
        self, audio_file: Path, target_level: float = -23.0
    ) -> ConversionResult:
        """音频标准化（音量均衡）"""

        if not self.check_ffmpeg_available():
            return self._create_error_result(
                input_file=audio_file,
                output_file=Path(),
                conversion_type="normalize",
                processing_time=0.0,
                input_size=0,
                output_size=0,
                error_message="ffmpeg 未安装",
            )

        output_filename = f"{audio_file.stem}_normalized.wav"
        output_path = self.output_dir / output_filename

        try:
            start_time = time.time()

            # 构建 ffmpeg 命令（使用 loudnorm 滤波器）
            cmd = [
                "ffmpeg",
                "-i",
                str(audio_file),
                "-af",
                f"loudnorm=I={target_level}:TP=-1.5:LRA=11:print_format=summary",
                "-ac",
                str(self.WHISPER_AUDIO_PARAMS["channels"]),
                "-ar",
                str(self.WHISPER_AUDIO_PARAMS["sample_rate"]),
                "-acodec",
                self.WHISPER_AUDIO_PARAMS["codec"],
                "-y",
                str(output_path),
            ]

            self.logger.info("正在标准化音频: %s", audio_file.name)

            # 执行标准化 - 使用上下文管理器确保资源正确释放
            try:
                # 设置超时时间，避免进程挂起
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=300,  # 5分钟超时，可根据实际需求调整
                )
            except subprocess.TimeoutExpired:
                self.logger.error("音频标准化操作超时")
                return self._create_error_result(
                    input_file=audio_file,
                    output_file=output_path,
                    conversion_type="normalize",
                    processing_time=time.time() - start_time,
                    input_size=audio_file.stat().st_size if audio_file.exists() else 0,
                    output_size=0,
                    error_message="音频标准化操作超时（5分钟）",
                )

            processing_time = time.time() - start_time

            # 检查输出文件是否存在且大小合理
            if (
                result.returncode == 0
                and output_path.exists()
                and output_path.stat().st_size > 0
            ):
                # 获取文件大小信息并创建成功结果对象
                input_size = audio_file.stat().st_size
                output_size = output_path.stat().st_size
                compression_ratio = output_size / input_size if input_size > 0 else 0

                processing_info = {
                    "processing_time": processing_time,
                    "input_size": input_size,
                    "output_size": output_size,
                    "compression_ratio": compression_ratio,
                }

                # 记录成功日志
                self.logger.info(
                    "音频标准化成功: %s, 耗时: %.2f秒, 压缩比: %.2f",
                    output_filename,
                    processing_time,
                    compression_ratio,
                )

                # 返回成功结果
                return self._create_success_result(
                    input_file=audio_file,
                    output_file=output_path,
                    conversion_type="normalize",
                    processing_info=processing_info,
                )

            # 处理失败情况（ffmpeg返回非零状态码）
            error_message = f"音频标准化失败: {result.stderr}"
            self.logger.error("音频标准化失败: %s", result.stderr)

            # 清理不完整的输出文件
            self._cleanup_incomplete_file(output_path)

            # 返回失败结果
            return self._create_error_result(
                input_file=audio_file,
                output_file=output_path,
                conversion_type="normalize",
                processing_time=processing_time,
                input_size=audio_file.stat().st_size if audio_file.exists() else 0,
                output_size=output_path.stat().st_size if output_path.exists() else 0,
                error_message=error_message,
            )

        except (
            subprocess.SubprocessError,
            FileNotFoundError,
            PermissionError,
            OSError,
        ) as e:
            error_message = f"音频标准化过程中发生错误: {e}"
            self.logger.error("音频标准化过程中发生错误: %s", e)
            return self._create_error_result(
                input_file=audio_file,
                output_file=output_path,
                conversion_type="normalize",
                processing_time=0.0,
                input_size=audio_file.stat().st_size if audio_file.exists() else 0,
                output_size=output_path.stat().st_size if output_path.exists() else 0,
                error_message=error_message,
            )

    def _process_single_file(
        self, media_file: MediaFile, operation: str, index: int, total_files: int
    ) -> ConversionResult:
        """处理单个媒体文件

        将批处理中的单个文件处理逻辑抽取为独立函数，减少主函数的分支复杂度
        """
        self.logger.info("处理文件 %d/%d: %s", index, total_files, media_file.filename)

        try:
            # 根据操作类型执行相应的处理
            if operation == "extract_audio":
                result = self.extract_audio_from_video(media_file.path)
            elif operation == "convert_format":
                result = self.convert_audio_format(media_file.path)
            elif operation == "normalize":
                result = self.normalize_audio(media_file.path)
            else:
                # 理论上不会执行到这里，因为在批处理函数中已经验证了操作类型
                return self._create_error_result(
                    input_file=media_file.path,
                    output_file=Path(),
                    conversion_type=operation,
                    processing_time=0.0,
                    input_size=(
                        media_file.path.stat().st_size
                        if media_file.path.exists()
                        else 0
                    ),
                    output_size=0,
                    error_message=f"不支持的操作类型: {operation}",
                )

            # 记录处理进度
            if result.success:
                self.logger.info("文件 %d/%d 处理成功", index, total_files)
            else:
                self.logger.warning(
                    "文件 %d/%d 处理失败: %s",
                    index,
                    total_files,
                    result.error_message,
                )

            return result

        except (
            subprocess.SubprocessError,
            FileNotFoundError,
            PermissionError,
            OSError,
            ValueError,
            RuntimeError,
            TypeError,
            IOError,
        ) as e:
            self.logger.error("处理文件 %s 时发生异常: %s", media_file.filename, e)
            # 创建错误结果并返回
            return self._create_error_result(
                input_file=media_file.path,
                output_file=Path(),
                conversion_type=operation,
                processing_time=0.0,
                input_size=(
                    media_file.path.stat().st_size if media_file.path.exists() else 0
                ),
                output_size=0,
                error_message=f"处理过程中发生异常: {e}",
            )

    def batch_process(
        self, media_files: List[MediaFile], operation: str = "extract_audio"
    ) -> List[ConversionResult]:
        """批量处理媒体文件"""

        # 检查操作类型是否支持
        supported_operations = {"extract_audio", "convert_format", "normalize"}
        if operation not in supported_operations:
            self.logger.error("不支持的批处理操作: %s", operation)
            self.logger.info("支持的操作: %s", ", ".join(supported_operations))
            return []

        results = []
        total_files = len(media_files)

        if total_files == 0:
            self.logger.warning("没有找到需要处理的媒体文件")
            return results

        self.logger.info("开始批量处理 %d 个文件，操作: %s", total_files, operation)
        start_time = time.time()

        try:
            for i, media_file in enumerate(media_files, 1):
                # 处理单个文件并获取结果
                result = self._process_single_file(
                    media_file, operation, i, total_files
                )
                results.append(result)

                # 添加短暂延迟避免资源竞争
                time.sleep(0.1)

        except (
            subprocess.SubprocessError,
            FileNotFoundError,
            PermissionError,
            OSError,
            ValueError,
            RuntimeError,
            TypeError,
            IOError,
        ) as e:
            self.logger.error("批处理过程中发生严重错误: %s", e)

        # 记录总体处理结果
        total_time = time.time() - start_time
        success_count = sum(1 for r in results if r.success)
        self.logger.info(
            "批处理完成: 总计 %d 个文件, 成功 %d 个, 失败 %d 个, 总耗时: %.2f秒",
            total_files,
            success_count,
            total_files - success_count,
            total_time,
        )

        return results

    def cleanup_temp_files(self):
        """清理临时文件"""
        try:
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                self.logger.info("临时文件已清理")
        except (PermissionError, OSError) as e:
            self.logger.warning("清理临时文件失败: %s", e)

    def __del__(self):
        """析构函数，清理临时文件"""
        self.cleanup_temp_files()


def create_format_converter(output_dir: str = "converted_audio") -> FormatConverter:
    """创建格式转换器的便捷函数"""
    return FormatConverter(output_dir)


if __name__ == "__main__":
    # 测试代码
    converter = FormatConverter()
    print(f"ffmpeg 可用: {converter.check_ffmpeg_available()}")
