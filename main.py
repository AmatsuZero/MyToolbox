#!/usr/bin/env python3
"""
语音转文字工具 - 主程序入口模块

该模块是整个语音转文字系统的核心入口，协调各个组件的工作流程。
基于 OpenAI Whisper 的语音识别系统，支持多种音视频格式和字幕生成。
主要功能包括：
- 系统组件的初始化和资源管理
- 媒体文件的扫描和过滤
- 音视频文件的预处理和格式转换
- 使用Whisper模型进行语音识别
- 生成多种格式的字幕文件
- 批量处理多个文件
- 生成处理摘要报告
- 错误处理和性能监控
- 命令行和交互式操作模式

Classes:
    SpeechToTextSystem: 语音转文字系统的主类，协调各组件工作
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional, Any
import logging
import time
import json
import torch

# 导入自定义模块
from file_scanner import MediaFileScanner, MediaFile
from format_converter import FormatConverter
from whisper_integration import WhisperIntegration, TranscriptionResult, ModelConfig
from subtitle_generator import SubtitleGenerator
from error_handler import ErrorHandler, ErrorLevel, ErrorContext
from config import ConfigManager, ProjectConfig
from cli_interface import CLIInterface


class SpeechToTextSystem:
    """语音转文字系统主类

    该类协调各个组件的工作流程，管理系统资源和处理流程。

    Attributes:
        config: 项目配置对象
        _components: 系统组件字典，包含所有功能模块
        _stats: 处理统计信息
    """

    def __init__(self, config: Optional[ProjectConfig] = None):
        self.config = config or ProjectConfig()

        # 将所有组件（包括logger）整合到一个字典中，减少实例属性数量
        self._components = {
            "file_scanner": None,
            "format_converter": None,
            "whisper_integration": None,
            "subtitle_generator": None,
            "error_handler": ErrorHandler(),
            "config_manager": ConfigManager(),
            "logger": self._setup_logger(),  # 将logger也移入组件字典
        }

        # 处理统计
        self._stats = {
            "total_files": 0,
            "successful_files": 0,
            "failed_files": 0,
            "total_processing_time": 0.0,
            "average_confidence": 0.0,
            "start_time": None,
            "end_time": None,
        }

    @property
    def file_scanner(self):
        """获取文件扫描器组件"""
        return self._components["file_scanner"]

    @file_scanner.setter
    def file_scanner(self, value):
        """设置文件扫描器组件"""
        self._components["file_scanner"] = value

    @property
    def format_converter(self):
        """获取格式转换器组件"""
        return self._components["format_converter"]

    @format_converter.setter
    def format_converter(self, value):
        """设置格式转换器组件"""
        self._components["format_converter"] = value

    @property
    def whisper_integration(self):
        """获取Whisper集成组件"""
        return self._components["whisper_integration"]

    @whisper_integration.setter
    def whisper_integration(self, value):
        """设置Whisper集成组件"""
        self._components["whisper_integration"] = value

    @property
    def subtitle_generator(self):
        """获取字幕生成器组件"""
        return self._components["subtitle_generator"]

    @subtitle_generator.setter
    def subtitle_generator(self, value):
        """设置字幕生成器组件"""
        self._components["subtitle_generator"] = value

    @property
    def error_handler(self):
        """获取错误处理器组件"""
        return self._components["error_handler"]

    @error_handler.setter
    def error_handler(self, value):
        """设置错误处理器组件"""
        self._components["error_handler"] = value

    @property
    def config_manager(self):
        """获取配置管理器组件"""
        return self._components["config_manager"]

    @config_manager.setter
    def config_manager(self, value):
        """设置配置管理器组件"""
        self._components["config_manager"] = value

    @property
    def logger(self):
        """获取日志记录器组件"""
        return self._components["logger"]

    @logger.setter
    def logger(self, value):
        """设置日志记录器组件"""
        self._components["logger"] = value

    @property
    def processing_stats(self):
        """获取处理统计信息"""
        return self._stats

    @processing_stats.setter
    def processing_stats(self, value):
        """设置处理统计信息"""
        self._stats = value

    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger("SpeechToTextSystem")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def initialize_components(self) -> bool:
        """初始化所有组件"""

        try:
            self.logger.info("正在初始化系统组件...")

            # 初始化文件扫描器
            scan_dir = "media"  # 默认值
            if self.config.file_scanner is not None and hasattr(
                self.config.file_scanner, "scan_directory"
            ):
                scan_dir = self.config.file_scanner.scan_directory

            self.file_scanner = MediaFileScanner(scan_dir)

            # 初始化格式转换器
            self.format_converter = FormatConverter(
                self.config.format_converter.output_directory
                if self.config.format_converter is not None
                else "converted_audio"
            )

            # 初始化Whisper集成
            whisper_config = self.config.whisper

            # 创建一个包含默认值的ModelConfig
            model_config = ModelConfig(
                name="whisper_base",  # 默认值，后面会更新
                size="base",  # 默认值，后面会更新
            )

            # 设置基本属性
            if whisper_config is not None and hasattr(whisper_config, "model_size"):
                model_config.name = f"whisper_{whisper_config.model_size}"
                model_config.size = whisper_config.model_size
            if whisper_config is not None and hasattr(whisper_config, "language"):
                model_config.language = whisper_config.language
            if whisper_config is not None and hasattr(whisper_config, "device"):
                model_config.device = whisper_config.device
            if whisper_config is not None and hasattr(whisper_config, "compute_type"):
                model_config.compute_type = whisper_config.compute_type
            if whisper_config is not None and hasattr(whisper_config, "engine"):
                model_config.engine = whisper_config.engine

            # 设置logprob_threshold默认值为-1.0
            model_config.logprob_threshold = -1.0
            self.whisper_integration = WhisperIntegration(model_config)

            # 初始化字幕生成器
            self.subtitle_generator = SubtitleGenerator(
                self.config.subtitle_generator.output_directory
                if self.config.subtitle_generator is not None
                else "subtitles"
            )

            # 启动性能监控
            enable_monitoring = False
            monitoring_interval = 60  # 默认值，60秒

            if self.config.error_handler is not None:
                if hasattr(self.config.error_handler, "enable_performance_monitoring"):
                    enable_monitoring = (
                        self.config.error_handler.enable_performance_monitoring
                    )
                if hasattr(
                    self.config.error_handler, "performance_monitoring_interval"
                ):
                    monitoring_interval = (
                        self.config.error_handler.performance_monitoring_interval
                    )

            if enable_monitoring:
                self.error_handler.start_performance_monitoring(monitoring_interval)

            self.logger.info("系统组件初始化完成")
            return True

        except (ValueError, ImportError, RuntimeError, OSError) as e:
            self.error_handler.handle_exception(
                module="SpeechToTextSystem",
                function="initialize_components",
                exception=e,
            )
            return False

    def scan_media_files(self, scan_directory: Optional[str] = None) -> List[MediaFile]:
        """扫描媒体文件"""

        try:
            # 检查组件是否已初始化
            if self.file_scanner is None:
                self.logger.error(
                    "文件扫描器未初始化，请先调用 initialize_components()"
                )
                return []

            directory = scan_directory or self.file_scanner.scan_directory

            self.logger.info("正在扫描目录: %s", directory)

            # 更新扫描器目录
            self.file_scanner.scan_directory = Path(directory)

            # 扫描文件
            recursive = False  # 默认值
            if self.config.file_scanner is not None and hasattr(
                self.config.file_scanner, "recursive"
            ):
                recursive = self.config.file_scanner.recursive

            media_files = self.file_scanner.scan_files(recursive=recursive)

            if not media_files:
                self.logger.warning("未找到支持的媒体文件")
                return []

            # 应用文件大小过滤
            min_size_mb = 0  # 默认最小值 0MB
            max_size_mb = 1024  # 默认最大值 1GB

            if self.config.file_scanner is not None:
                if hasattr(self.config.file_scanner, "min_file_size_mb"):
                    min_size_mb = self.config.file_scanner.min_file_size_mb
                if hasattr(self.config.file_scanner, "max_file_size_mb"):
                    max_size_mb = self.config.file_scanner.max_file_size_mb

            media_files = self.file_scanner.filter_by_size(
                min_size=min_size_mb * 1024 * 1024,
                max_size=max_size_mb * 1024 * 1024,
            )

            stats = self.file_scanner.get_statistics()
            self.logger.info(
                "找到 %s 个媒体文件（音频: %s, 视频: %s, 总大小: %.2fMB）",
                stats["total_files"],
                stats["audio_files"],
                stats["video_files"],
                stats["total_size_mb"],
            )

            return media_files

        except (
            ValueError,
            RuntimeError,
            IOError,
            OSError,
            FileNotFoundError,
            PermissionError,
        ) as e:
            self.error_handler.handle_exception(
                module="SpeechToTextSystem", function="scan_media_files", exception=e
            )
            return []

    def preprocess_media_file(self, media_file: MediaFile) -> Optional[Path]:
        """预处理媒体文件（格式转换）"""

        try:
            # 检查组件是否已初始化
            if self.format_converter is None:
                self.logger.error(
                    "格式转换器未初始化，请先调用 initialize_components()"
                )
                return None

            self.logger.info("预处理文件: %s", media_file.filename)

            if media_file.file_type == "video":
                # 从视频提取音频
                result = self.format_converter.extract_audio_from_video(media_file.path)
            else:
                # 转换音频格式
                result = self.format_converter.convert_audio_format(media_file.path)

            if not result.success:
                # 创建错误上下文对象
                error_context = ErrorContext(
                    file_path=str(media_file.path), file_size=media_file.size
                )
                # 创建日志错误参数对象
                error_params = self.error_handler.LogErrorParams(
                    level=ErrorLevel.ERROR,
                    module="SpeechToTextSystem",
                    function="preprocess_media_file",
                    message=f"预处理失败: {result.error_message}",
                    context=error_context,
                )

                self.error_handler.log_error(error_params=error_params)
                return None

            self.logger.info("预处理完成: %s", result.output_file.name)
            return result.output_file

        except (ValueError, RuntimeError, IOError, OSError) as e:
            self.error_handler.handle_exception(
                module="SpeechToTextSystem",
                function="preprocess_media_file",
                exception=e,
                file_path=str(media_file.path),
                file_size=media_file.size,
            )
            return None

    def transcribe_audio(self, audio_file: Path) -> Optional[TranscriptionResult]:
        """转录音频文件"""

        try:
            self.logger.info("开始语音识别: %s", audio_file.name)

            if self.whisper_integration is None:
                # 创建错误上下文对象
                error_context = ErrorContext(file_path=str(audio_file))

                # 创建日志错误参数对象
                error_params = self.error_handler.LogErrorParams(
                    level=ErrorLevel.ERROR,
                    module="SpeechToTextSystem",
                    function="transcribe_audio",
                    message="Whisper集成尚未初始化",
                    context=error_context,
                )

                self.error_handler.log_error(error_params=error_params)
                return None

            # 加载模型（如果尚未加载）
            if self.whisper_integration.model is None:
                model_size = "base"  # 默认值
                if (
                    hasattr(self.config, "whisper")
                    and self.config.whisper is not None
                    and hasattr(self.config.whisper, "model_size")
                ):
                    model_size = self.config.whisper.model_size

                if not self.whisper_integration.load_model(model_size):
                    # 创建错误上下文对象
                    error_context = ErrorContext(file_path=str(audio_file))

                    # 创建日志错误参数对象
                    error_params = self.error_handler.LogErrorParams(
                        level=ErrorLevel.ERROR,
                        module="SpeechToTextSystem",
                        function="transcribe_audio",
                        message="Whisper模型加载失败",
                        context=error_context,
                    )

                    self.error_handler.log_error(error_params=error_params)
                    return None

            # 执行转录
            language = None
            if (
                hasattr(self.config, "whisper")
                and self.config.whisper is not None
                and hasattr(self.config.whisper, "language")
            ):
                language = self.config.whisper.language

            result = self.whisper_integration.transcribe_audio(
                audio_file, language=language
            )

            if not result.success:
                # 创建错误上下文对象
                error_context = ErrorContext(file_path=str(audio_file))

                # 创建日志错误参数对象
                error_params = self.error_handler.LogErrorParams(
                    level=ErrorLevel.ERROR,
                    module="SpeechToTextSystem",
                    function="transcribe_audio",
                    message=f"语音识别失败: {result.error_message}",
                    context=error_context,
                )

                self.error_handler.log_error(error_params=error_params)
                return None

            self.logger.info(
                "语音识别完成: %s, 置信度: %.2f", audio_file.name, result.confidence
            )
            return result

        except (
            ValueError,
            RuntimeError,
            IOError,
            OSError,
            FileNotFoundError,
            PermissionError,
            torch.cuda.CudaError,
            torch.cuda.OutOfMemoryError,
        ) as e:
            self.error_handler.handle_exception(
                module="SpeechToTextSystem",
                function="transcribe_audio",
                exception=e,
                file_path=str(audio_file),
            )
            return None

    def generate_subtitles(
        self,
        transcription_result: TranscriptionResult,
        output_formats: Optional[List[str]] = None,
    ) -> Dict[str, Path]:
        """生成字幕文件"""

        try:
            self.logger.info("生成字幕文件...")

            if self.subtitle_generator is None:
                # 创建错误上下文对象
                error_context = ErrorContext()

                # 创建日志错误参数对象
                error_params = self.error_handler.LogErrorParams(
                    level=ErrorLevel.ERROR,
                    module="SpeechToTextSystem",
                    function="generate_subtitles",
                    message="字幕生成器尚未初始化",
                    context=error_context,
                )

                self.error_handler.log_error(error_params=error_params)
                return {}

            if output_formats is None:
                output_formats = []
                if self.config.subtitle_generator is not None and hasattr(
                    self.config.subtitle_generator, "supported_formats"
                ):
                    output_formats = (
                        self.config.subtitle_generator.supported_formats or []
                    )

            # 创建字幕片段
            segments = self.subtitle_generator.create_segments_from_transcription(
                transcription_result
            )

            subtitle_files = {}

            # 生成各种格式的字幕
            for format_type in output_formats:
                if format_type == "srt":
                    file_path = self.subtitle_generator.generate_srt(
                        segments, f"{transcription_result.audio_file.stem}.srt"
                    )
                    subtitle_files["srt"] = file_path

                elif format_type == "vtt":
                    file_path = self.subtitle_generator.generate_vtt(
                        segments, f"{transcription_result.audio_file.stem}.vtt"
                    )
                    subtitle_files["vtt"] = file_path

                elif format_type == "txt":
                    file_path = self.subtitle_generator.generate_txt(
                        segments, f"{transcription_result.audio_file.stem}.txt"
                    )
                    subtitle_files["txt"] = file_path

                elif format_type == "json":
                    file_path = self.subtitle_generator.generate_json(
                        segments, f"{transcription_result.audio_file.stem}.json"
                    )
                    subtitle_files["json"] = file_path

                elif format_type == "csv":
                    file_path = self.subtitle_generator.generate_csv(
                        segments, f"{transcription_result.audio_file.stem}.csv"
                    )
                    subtitle_files["csv"] = file_path

            self.logger.info("字幕文件生成完成: %s 个文件", len(subtitle_files))
            return subtitle_files

        except (
            ValueError,
            RuntimeError,
            IOError,
            OSError,
            FileNotFoundError,
            PermissionError,
            TypeError,
            KeyError,
            IndexError,
        ) as e:
            self.error_handler.handle_exception(
                module="SpeechToTextSystem", function="generate_subtitles", exception=e
            )
            return {}

    def process_single_file(self, media_file: MediaFile) -> Dict[str, Any]:
        """处理单个文件"""

        result = {
            "input_file": str(media_file.path),
            "filename": media_file.filename,
            "file_type": media_file.file_type,
            "size_mb": media_file.size / (1024 * 1024),
            "success": False,
            "processing_time": 0.0,
            "error_message": None,
            "transcription_result": None,
            "subtitle_files": {},
        }

        start_time = time.time()

        try:
            # 1. 预处理
            audio_file = self.preprocess_media_file(media_file)
            if not audio_file:
                result["error_message"] = "预处理失败"
                return result

            # 2. 语音识别
            transcription_result = self.transcribe_audio(audio_file)
            if not transcription_result:
                result["error_message"] = "语音识别失败"
                return result

            result["transcription_result"] = transcription_result

            # 3. 生成字幕
            subtitle_files = self.generate_subtitles(transcription_result)
            result["subtitle_files"] = {k: str(v) for k, v in subtitle_files.items()}

            # 4. 清理临时文件
            if audio_file != media_file.path:
                try:
                    audio_file.unlink()
                except (OSError, FileNotFoundError, PermissionError):
                    pass

            result["success"] = True
            result["processing_time"] = time.time() - start_time

            return result

        except (ValueError, RuntimeError, IOError, OSError) as e:
            result["error_message"] = str(e)
            result["processing_time"] = time.time() - start_time

            self.error_handler.handle_exception(
                module="SpeechToTextSystem",
                function="process_single_file",
                exception=e,
                file_path=str(media_file.path),
                file_size=media_file.size,
                processing_time=result["processing_time"],
            )

            return result

    def process_batch_files(self, media_files: List[MediaFile]) -> List[Dict[str, Any]]:
        """批量处理文件"""

        self.processing_stats["start_time"] = time.time()
        self.processing_stats["total_files"] = len(media_files)

        results = []

        for i, media_file in enumerate(media_files, 1):
            self.logger.info(
                "处理进度: %s/%s - %s", i, len(media_files), media_file.filename
            )

            # 处理单个文件
            result = self.process_single_file(media_file)
            results.append(result)

            # 更新统计信息
            if result["success"]:
                self.processing_stats["successful_files"] += 1
                self.processing_stats["total_processing_time"] += result[
                    "processing_time"
                ]

                if result["transcription_result"]:
                    confidence = result["transcription_result"].confidence
                    self.processing_stats["average_confidence"] = (
                        self.processing_stats["average_confidence"]
                        * (self.processing_stats["successful_files"] - 1)
                        + confidence
                    ) / self.processing_stats["successful_files"]
            else:
                self.processing_stats["failed_files"] += 1

            # 内存优化
            enable_memory_optimization = False

            if (
                hasattr(self.config, "performance")
                and self.config.performance is not None
            ):
                # 使用 getattr 安全地获取属性值
                enable_memory_optimization = getattr(
                    self.config.performance, "enable_memory_optimization", False
                )

            if enable_memory_optimization and i % 5 == 0:
                self.error_handler.optimize_memory()

        self.processing_stats["end_time"] = time.time()

        return results

    def generate_summary_report(
        self, results: List[Dict[str, Any]], output_dir: str = "output"
    ) -> Path:
        """生成摘要报告"""

        try:
            summary = {
                "processing_summary": self.processing_stats,
                "config_summary": self.config_manager.get_config_summary(self.config),
                "files": results,
                "timestamp": time.time(),
                "version": self.config.version,
            }

            # 计算更多统计信息
            summary["processing_summary"]["total_duration"] = (
                self.processing_stats["end_time"] - self.processing_stats["start_time"]
            )
            summary["processing_summary"]["average_processing_time"] = (
                self.processing_stats["total_processing_time"]
                / max(1, self.processing_stats["successful_files"])
            )
            summary["processing_summary"]["success_rate"] = self.processing_stats[
                "successful_files"
            ] / max(1, self.processing_stats["total_files"])

            # 保存报告
            output_path = Path(output_dir) / "processing_summary.json"

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            self.logger.info("摘要报告已生成: %s", output_path)
            return output_path

        except (
            ValueError,
            RuntimeError,
            IOError,
            OSError,
            FileNotFoundError,
            PermissionError,
            TypeError,
            KeyError,
            json.JSONDecodeError,
        ) as e:
            self.error_handler.handle_exception(
                module="SpeechToTextSystem",
                function="generate_summary_report",
                exception=e,
            )
            raise

    def cleanup(self):
        """清理资源"""

        self.logger.info("正在清理资源...")

        # 停止性能监控
        self.error_handler.stop_performance_monitoring()

        # 保存错误日志
        self.error_handler.save_logs()

        # 清理临时文件
        if self.format_converter:
            self.format_converter.cleanup_temp_files()

        self.logger.info("资源清理完成")

    def run(self, scan_directory: Optional[str] = None) -> bool:
        """运行语音转文字系统"""

        try:
            # 初始化组件
            if not self.initialize_components():
                return False

            # 扫描文件
            media_files = self.scan_media_files(scan_directory)
            if not media_files:
                self.logger.error("未找到可处理的文件")
                return False

            # 批量处理
            self.logger.info("开始批量处理文件...")
            results = self.process_batch_files(media_files)

            # 生成摘要报告
            # 获取字幕输出目录
            output_dir = "subtitles"
            if self.config.subtitle_generator is not None:
                output_dir = self.config.subtitle_generator.output_directory

            self.generate_summary_report(results, output_dir)

            # 输出结果
            self.logger.info(
                "处理完成: 成功 %s/%s, 失败 %s",
                self.processing_stats["successful_files"],
                self.processing_stats["total_files"],
                self.processing_stats["failed_files"],
            )

            return self.processing_stats["failed_files"] == 0

        except KeyboardInterrupt:
            self.logger.info("处理被用户中断")
            self.cleanup()
            return False

        except (ValueError, RuntimeError, IOError, OSError) as e:
            self.error_handler.handle_exception(
                module="SpeechToTextSystem", function="run", exception=e
            )
            self.cleanup()
            return False

        finally:
            self.cleanup()


def main():
    """主函数"""

    # 检查命令行参数
    if len(sys.argv) > 1:
        # 使用命令行接口
        cli = CLIInterface()
        return cli.run(sys.argv[1:])

    # 交互式模式
    print("语音转文字工具 - 交互式模式")
    print("=" * 50)

    try:
        # 加载配置
        config_manager = ConfigManager()
        config = config_manager.load_config()

        # 创建系统实例
        system = SpeechToTextSystem(config)

        # 获取扫描目录
        scan_directory = input("请输入要扫描的目录路径（默认: downloads）: ").strip()
        if not scan_directory:
            scan_directory = "downloads"

        # 运行系统
        success = system.run(scan_directory)

        if success:
            print("\n处理完成！")
            return 0
        print("\n处理过程中出现错误，请查看日志文件")
        return 1

    except KeyboardInterrupt:
        print("\n程序被用户中断")
        return 130

    except (ValueError, ImportError, RuntimeError, OSError) as e:
        print(f"\n程序运行错误: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())