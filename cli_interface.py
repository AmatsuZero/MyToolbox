#!/usr/bin/env python3
"""
语音转文字工具 - 命令行接口模块

该模块提供了一个完整的命令行界面，用于处理音视频文件的语音识别和字幕生成。
主要功能包括：
- 命令行参数解析和验证
- 媒体文件扫描和过滤
- 音频提取和格式转换
- 使用Whisper模型进行语音识别
- 生成多种格式的字幕文件（SRT、VTT、ASS、TXT、JSON、CSV）
- 批量处理多个文件
- 配置文件加载和保存
- 性能监控和内存优化
- 错误处理和日志记录
- 生成处理摘要报告

Classes:
    CLIInterface: 命令行接口类，处理参数解析、文件处理和结果输出
"""

import argparse
import fnmatch
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from error_handler import ErrorHandler
from file_scanner import MediaFile, MediaFileScanner
from format_converter import FormatConverter
from subtitle_generator import SubtitleGenerator
from whisper_config import WhisperConfigManager
from whisper_integration import WhisperIntegration, create_model_config


class CLIInterface:
    """命令行接口类"""

    def __init__(self):
        self.parser = self._setup_parser()
        self.logger = self._setup_logger()
        self.error_handler = ErrorHandler()
        self.config_manager = WhisperConfigManager()

    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger("CLIInterface")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _setup_parser(self) -> argparse.ArgumentParser:
        """设置命令行参数解析器"""

        parser = argparse.ArgumentParser(
            description="语音转文字工具 - 支持多种音视频格式的语音识别和字幕生成",
            epilog="示例: python cli_interface.py --input downloads/ --model medium "
            "--language zh --output subtitles/",
        )

        # 输入参数
        input_group = parser.add_argument_group("输入参数")
        input_group.add_argument(
            "-i", "--input", type=str, required=True, help="输入文件或目录路径"
        )
        input_group.add_argument(
            "--recursive", action="store_true", help="递归扫描子目录"
        )
        input_group.add_argument(
            "--file-pattern",
            type=str,
            default="*",
            help="文件匹配模式（例如：*.mp4,*.wav）",
        )

        # 模型参数
        model_group = parser.add_argument_group("模型参数")
        model_group.add_argument(
            "-m",
            "--model",
            type=str,
            choices=["tiny", "base", "small", "medium", "large"],
            default="base",
            help="Whisper模型大小（默认：base）",
        )
        model_group.add_argument(
            "-l",
            "--language",
            type=str,
            default="auto",
            help="目标语言（例如：zh, en, ja, ko, fr, de, es），默认自动检测",
        )
        model_group.add_argument(
            "--device",
            type=str,
            choices=["auto", "cpu", "cuda", "mps"],
            default="auto",
            help="计算设备（默认：auto）",
        )
        model_group.add_argument(
            "--engine",
            type=str,
            choices=["auto", "whisperkit", "faster-whisper", "openai-whisper"],
            default="auto",
            help="转录引擎（默认：auto）",
        )
        model_group.add_argument(
            "--compute-type",
            type=str,
            choices=["default", "int8", "int8_float16", "float16", "float32"],
            default="default",
            help="量化类型（默认：default）",
        )

        # 输出参数
        output_group = parser.add_argument_group("输出参数")
        output_group.add_argument(
            "-o",
            "--output",
            type=str,
            default="output",
            help="输出目录路径（默认：output）",
        )
        output_group.add_argument(
            "--format",
            type=str,
            choices=["srt", "vtt", "ass", "txt", "json", "csv", "all"],
            default="all",
            help="输出字幕格式（默认：all）",
        )
        output_group.add_argument(
            "--encoding", type=str, default="utf-8", help="输出文件编码（默认：utf-8）"
        )

        # 处理参数
        processing_group = parser.add_argument_group("处理参数")
        processing_group.add_argument(
            "--batch-size", type=int, default=1, help="批处理大小（默认：1）"
        )
        processing_group.add_argument(
            "--max-workers", type=int, default=1, help="最大工作进程数（默认：1）"
        )
        processing_group.add_argument(
            "--chunk-length", type=int, default=30, help="音频分块长度（秒，默认：30）"
        )
        processing_group.add_argument(
            "--temperature", type=float, default=0.0, help="采样温度（默认：0.0）"
        )
        processing_group.add_argument(
            "--beam-size", type=int, default=5, help="束搜索大小（默认：5）"
        )
        processing_group.add_argument(
            "--best-of", type=int, default=5, help="最佳结果数量（默认：5）"
        )

        # 性能参数
        performance_group = parser.add_argument_group("性能参数")
        performance_group.add_argument(
            "--optimize-memory", action="store_true", help="启用内存优化"
        )
        performance_group.add_argument(
            "--enable-gpu", action="store_true", help="启用GPU加速"
        )
        performance_group.add_argument(
            "--max-memory", type=int, help="最大内存使用量（MB）"
        )
        performance_group.add_argument(
            "--timeout", type=int, default=3600, help="处理超时时间（秒，默认：3600）"
        )

        # 配置参数
        config_group = parser.add_argument_group("配置参数")
        config_group.add_argument("--config", type=str, help="配置文件路径")
        config_group.add_argument("--save-config", type=str, help="保存配置到文件")
        config_group.add_argument("--verbose", action="store_true", help="详细输出模式")
        config_group.add_argument("--quiet", action="store_true", help="静默模式")
        config_group.add_argument(
            "--dry-run", action="store_true", help="试运行模式（不实际处理文件）"
        )

        return parser

    def parse_args(self, args: Optional[List[str]] = None) -> argparse.Namespace:
        """解析命令行参数"""
        return self.parser.parse_args(args)

    def validate_args(self, args: argparse.Namespace) -> bool:
        """验证参数有效性"""
        is_valid = True

        # 检查输入路径
        input_path = Path(args.input)
        if not input_path.exists():
            self.logger.error("输入路径不存在: %s", args.input)
            is_valid = False

        # 检查输出目录
        if is_valid:
            output_path = Path(args.output)
            if not output_path.exists():
                try:
                    output_path.mkdir(parents=True, exist_ok=True)
                    self.logger.info("创建输出目录: %s", output_path)
                except (IOError, OSError, PermissionError) as e:
                    self.logger.error("无法创建输出目录: %s", str(e))
                    is_valid = False

        # 检查配置文件
        if is_valid and args.config:
            config_path = Path(args.config)
            if not config_path.exists():
                self.logger.error("配置文件不存在: %s", args.config)
                is_valid = False

        # 验证批处理参数
        if is_valid and args.batch_size < 1:
            self.logger.error("批处理大小必须大于0")
            is_valid = False

        if is_valid and args.max_workers < 1:
            self.logger.error("工作进程数必须大于0")
            is_valid = False

        if is_valid and args.timeout < 1:
            self.logger.error("超时时间必须大于0")
            is_valid = False

        return is_valid

    def load_config(self, config_path: Path) -> Optional[Dict]:
        """加载配置文件"""

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                if config_path.suffix == ".json":
                    return json.load(f)
                elif config_path.suffix in [".yaml", ".yml"]:
                    return yaml.safe_load(f)
                else:
                    # 尝试 JSON
                    return json.load(f)

        except (IOError, OSError, json.JSONDecodeError, yaml.YAMLError) as e:
            self.logger.error("加载配置文件失败: %s", str(e))
            return None

    def save_config(self, config: Dict, config_path: Path):
        """保存配置文件"""

        try:
            with open(config_path, "w", encoding="utf-8") as f:
                if config_path.suffix == ".json":
                    json.dump(config, f, indent=2, ensure_ascii=False)
                elif config_path.suffix in [".yaml", ".yml"]:
                    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
                else:
                    # 默认使用 JSON
                    json.dump(config, f, indent=2, ensure_ascii=False)

            self.logger.info("配置已保存: %s", config_path)

        except (IOError, OSError, TypeError, ValueError) as e:
            self.logger.error("保存配置失败: %s", str(e))

    def scan_files(self, args: argparse.Namespace) -> List[MediaFile]:
        """扫描媒体文件"""

        scanner = MediaFileScanner(args.input)

        if not scanner.scan_directory_exists():
            self.logger.error("扫描目录不存在: %s", args.input)
            return []

        self.logger.info("正在扫描目录: %s", args.input)

        media_files = scanner.scan_files(recursive=args.recursive)

        if not media_files:
            self.logger.warning("未找到支持的媒体文件")
            return []

        # 应用文件模式过滤
        if args.file_pattern != "*":
            pattern = args.file_pattern
            media_files = [
                f for f in media_files if fnmatch.fnmatch(f.filename, pattern)
            ]

        stats = scanner.get_statistics()
        self.logger.info(
            "找到 %d 个媒体文件（音频: %d, 视频: %d, 总大小: %.2fMB）",
            stats["total_files"],
            stats["audio_files"],
            stats["video_files"],
            stats["total_size_mb"],
        )

        return media_files

    def _prepare_audio_file(self, media_file: MediaFile) -> tuple:
        """准备音频文件，如果是视频则提取音频"""
        converter = FormatConverter()
        error_message = None
        audio_file = None

        if media_file.file_type == "video":
            self.logger.info("从视频提取音频...")
            conversion_result = converter.extract_audio_from_video(media_file.path)

            if not conversion_result.success:
                error_message = f"音频提取失败: {conversion_result.error_message}"
                return audio_file, error_message

            audio_file = conversion_result.output_file
            self.logger.info("音频提取成功: %s", audio_file.name)
        else:
            audio_file = media_file.path

        return audio_file, error_message

    def _generate_subtitle_files(
        self,
        segments,
        media_file: MediaFile,
        args: argparse.Namespace,
        subtitle_generator: SubtitleGenerator,
    ) -> Dict:
        """生成各种格式的字幕文件"""
        subtitle_files = {}
        formats_to_generate = {
            "srt": subtitle_generator.generate_srt,
            "vtt": subtitle_generator.generate_vtt,
            "txt": subtitle_generator.generate_txt,
            "json": subtitle_generator.generate_json,
            "csv": subtitle_generator.generate_csv,
        }

        for fmt, generator_func in formats_to_generate.items():
            if args.format in [fmt, "all"]:
                output_file = generator_func(segments, f"{media_file.path.stem}.{fmt}")
                subtitle_files[fmt] = str(output_file)

        return subtitle_files

    def process_file(
        self,
        media_file: MediaFile,
        args: argparse.Namespace,
        whisper_integration: WhisperIntegration,
        subtitle_generator: SubtitleGenerator,
    ) -> Dict:
        """处理单个文件"""

        result = {
            "input_file": str(media_file.path),
            "filename": media_file.filename,
            "file_type": media_file.file_type,
            "size_mb": media_file.size / (1024 * 1024),
            "success": False,
            "transcription_result": None,
            "subtitle_files": [],
            "processing_time": 0.0,
            "error_message": None,
        }

        try:
            start_time = time.time()
            self.logger.info("开始处理文件: %s", media_file.filename)

            # 1. 准备音频文件
            audio_file, error_message = self._prepare_audio_file(media_file)
            if error_message:
                result["error_message"] = error_message
                return result

            # 2. 语音识别
            self.logger.info("开始语音识别...")
            transcription_result = whisper_integration.transcribe_audio(
                audio_file, language=args.language
            )

            if not transcription_result.success:
                result["error_message"] = (
                    f"语音识别失败: {transcription_result.error_message}"
                )
                return result

            result["transcription_result"] = transcription_result
            self.logger.info(
                "语音识别完成，置信度: %.2f", transcription_result.confidence
            )

            # 3. 生成字幕
            self.logger.info("生成字幕文件...")
            segments = subtitle_generator.create_segments_from_transcription(
                transcription_result
            )

            # 生成各种格式的字幕文件
            subtitle_files = self._generate_subtitle_files(
                segments, media_file, args, subtitle_generator
            )

            result["subtitle_files"] = subtitle_files
            result["processing_time"] = time.time() - start_time
            result["success"] = True

            self.logger.info(
                "文件处理完成: %s, 耗时: %.2f秒",
                media_file.filename,
                result["processing_time"],
            )

            return result

        except (IOError, OSError, ValueError, RuntimeError, KeyError, IndexError) as e:
            result["error_message"] = str(e)
            self.logger.error("处理文件失败: %s, 错误: %s", media_file.filename, str(e))
            return result

    def _configure_logging(self, args: argparse.Namespace) -> None:
        """根据参数配置日志级别"""
        if args.verbose:
            self.logger.setLevel(logging.DEBUG)
        elif args.quiet:
            self.logger.setLevel(logging.ERROR)

    def _load_configuration(self, args: argparse.Namespace) -> Dict:
        """加载配置文件"""
        config = {}
        if args.config:
            config = self.load_config(Path(args.config)) or {}
        return config

    def _handle_dry_run(self, args: argparse.Namespace) -> int:
        """处理试运行模式"""
        self.logger.info("试运行模式 - 不实际处理文件")
        media_files = self.scan_files(args)

        if not media_files:
            self.logger.warning("未找到可处理的文件")
            return 0

        self.logger.info("将处理 %d 个文件:", len(media_files))
        for media_file in media_files:
            self.logger.info(
                "  - %s (%s, %.2fMB)",
                media_file.filename,
                media_file.file_type,
                media_file.size / (1024 * 1024),
            )

        return 0

    def _process_files(self, args: argparse.Namespace, config: Dict) -> int:
        """处理文件的主要逻辑"""
        # 扫描文件
        media_files = self.scan_files(args)
        if not media_files:
            self.logger.error("未找到可处理的文件")
            return 1

        # 初始化组件
        self.logger.info("初始化组件...")
        engine_choice = args.engine if args.engine != "auto" else config.get("engine")
        compute_type = (
            args.compute_type
            if args.compute_type != "default"
            else config.get("compute_type", "default")
        )
        device_choice = (
            args.device if args.device != "auto" else config.get("device", "auto")
        )
        model_config = create_model_config(
            model_size=args.model,
            language=args.language,
            device=device_choice,
            compute_type=compute_type,
            engine=engine_choice,
        )
        whisper_integration = WhisperIntegration(model_config)
        subtitle_generator = SubtitleGenerator(args.output)

        config.update(
            {
                "model": args.model,
                "language": args.language,
                "device": device_choice,
                "engine": engine_choice,
                "compute_type": compute_type,
            }
        )

        # 加载模型
        if not whisper_integration.load_model(args.model):
            self.logger.error("模型加载失败")
            return 1

        # 处理文件
        results = self._process_file_batch(
            media_files, args, whisper_integration, subtitle_generator
        )

        # 生成摘要报告
        self.generate_summary_report(results, args.output)

        # 保存配置（如果需要）
        if args.save_config:
            config_path = Path(args.save_config)
            self.save_config(config, config_path)

        # 统计结果
        successful_files = sum(1 for r in results if r["success"])
        failed_files = len(results) - successful_files

        # 输出结果
        self.logger.info(
            "处理完成: 成功 %d/%d, 失败 %d",
            successful_files,
            len(media_files),
            failed_files,
        )

        return 0 if failed_files == 0 else 2

    def _process_file_batch(
        self,
        media_files: List[MediaFile],
        args: argparse.Namespace,
        whisper_integration: WhisperIntegration,
        subtitle_generator: SubtitleGenerator,
    ) -> List[Dict]:
        """批量处理文件"""
        results = []

        for i, media_file in enumerate(media_files, 1):
            self.logger.info("处理进度: %d/%d", i, len(media_files))

            result = self.process_file(
                media_file, args, whisper_integration, subtitle_generator
            )
            results.append(result)

            # 内存优化
            if args.optimize_memory and i % 5 == 0:
                self.error_handler.optimize_memory()

        return results

    def run(self, args: Optional[List[str]] = None) -> int:
        """运行命令行接口"""
        exit_code = 0

        try:
            # 解析参数
            parsed_args = self.parse_args(args)

            # 验证参数
            if not self.validate_args(parsed_args):
                return 1

            # 设置日志级别
            self._configure_logging(parsed_args)

            # 加载配置
            config = self._load_configuration(parsed_args)

            # 开始性能监控
            self.error_handler.start_performance_monitoring()

            # 处理逻辑
            if parsed_args.dry_run:
                exit_code = self._handle_dry_run(parsed_args)
            else:
                exit_code = self._process_files(parsed_args, config)

        except KeyboardInterrupt:
            self.logger.info("处理被用户中断")
            exit_code = 130
        except Exception as e:
            self.logger.error("处理过程中发生错误: %s", str(e))
            exit_code = 1
        finally:
            # 清理资源
            self.error_handler.stop_performance_monitoring()
            self.error_handler.save_logs()

        return exit_code

    def generate_summary_report(self, results: List[Dict], output_dir: str):
        """生成摘要报告"""

        def get_confidence(r):
            """安全获取转录结果的置信度"""
            tr = r.get("transcription_result")
            if tr is None:
                return 0
            # TranscriptionResult 是数据类，使用属性访问
            if hasattr(tr, "confidence"):
                return tr.confidence or 0
            # 兼容字典格式
            if isinstance(tr, dict):
                return tr.get("confidence", 0)
            return 0

        summary = {
            "total_files": len(results),
            "successful_files": sum(1 for r in results if r["success"]),
            "failed_files": sum(1 for r in results if not r["success"]),
            "total_processing_time": sum(
                r["processing_time"] for r in results if r["success"]
            ),
            "average_confidence": sum(
                get_confidence(r) for r in results if r["success"]
            )
            / max(1, sum(1 for r in results if r["success"])),
            "files": [],
        }

        for result in results:
            file_info = {
                "filename": result["filename"],
                "file_type": result["file_type"],
                "size_mb": result["size_mb"],
                "success": result["success"],
                "processing_time": result["processing_time"],
                "confidence": get_confidence(result),
                "subtitle_files": result.get("subtitle_files", {}),
                "error_message": result.get("error_message"),
            }
            summary["files"].append(file_info)

        # 保存摘要报告
        report_path = Path(output_dir) / "summary_report.json"

        try:
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            self.logger.info("摘要报告已生成: %s", report_path)

        except (IOError, OSError, TypeError, ValueError) as e:
            self.logger.error("生成摘要报告失败: %s", str(e))


def main():
    """主函数"""
    cli = CLIInterface()
    return cli.run()


if __name__ == "__main__":
    sys.exit(main())