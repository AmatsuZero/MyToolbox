"""
Whisper 字幕生成器模块

用于训练模式中自动为音频/视频文件生成字幕。
当用户仅提供媒体文件而无对应字幕时，该模块可调用 Whisper 自动生成字幕，
使其可用于 Mimic3 语音模型训练。
"""

import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Callable
from enum import Enum

from modules.whisper.whisper_integration import WhisperIntegration, ModelConfig, TranscriptionResult, create_model_config
from modules.training.train_data_manager import MediaFile, MediaType, TrainDataPair

logger = logging.getLogger(__name__)


class SubtitleFormat(Enum):
    """字幕输出格式"""
    SRT = "srt"
    VTT = "vtt"
    TXT = "txt"


@dataclass
class SubtitleGenerationResult:
    """字幕生成结果"""
    media_file: Path
    subtitle_path: Optional[Path]
    success: bool
    error_message: Optional[str] = None
    processing_time: float = 0.0
    detected_language: str = ""
    text_length: int = 0
    segment_count: int = 0


@dataclass 
class GenerationStatistics:
    """生成统计信息"""
    total_files: int = 0
    successful: int = 0
    failed: int = 0
    total_processing_time: float = 0.0
    total_audio_duration: float = 0.0
    results: list[SubtitleGenerationResult] = field(default_factory=list)
    
    def add_result(self, result: SubtitleGenerationResult) -> None:
        """添加结果并更新统计"""
        self.results.append(result)
        self.total_files += 1
        if result.success:
            self.successful += 1
        else:
            self.failed += 1
        self.total_processing_time += result.processing_time
    
    def summary(self) -> str:
        """生成统计摘要"""
        success_rate = (self.successful / self.total_files * 100) if self.total_files > 0 else 0
        return (
            f"字幕生成统计: "
            f"总计 {self.total_files} 个文件, "
            f"成功 {self.successful} 个 ({success_rate:.1f}%), "
            f"失败 {self.failed} 个, "
            f"总耗时 {self.total_processing_time:.2f} 秒"
        )


class WhisperSubtitleGenerator:
    """
    Whisper 字幕生成器
    
    用于训练模式中自动为音频/视频文件生成字幕。
    封装了 WhisperIntegration，提供针对训练数据准备优化的接口。
    """
    
    def __init__(
        self,
        model_size: str = "base",
        language: str = "auto",
        device: str = "auto",
        output_dir: Optional[Path] = None,
        output_format: SubtitleFormat = SubtitleFormat.SRT
    ):
        """
        初始化字幕生成器
        
        Args:
            model_size: Whisper 模型大小 (tiny, base, small, medium, large)
            language: 目标语言，"auto" 为自动检测
            device: 计算设备 (auto, cpu, cuda, mps)
            output_dir: 字幕输出目录，默认与媒体文件同目录
            output_format: 输出字幕格式
        """
        self.model_size = model_size
        self.language = language
        self.device = device
        self.output_dir = output_dir
        self.output_format = output_format
        
        self._whisper: Optional[WhisperIntegration] = None
        self._model_loaded = False
        
    def _ensure_model_loaded(self) -> bool:
        """确保模型已加载"""
        if self._model_loaded and self._whisper is not None:
            return True
            
        config = create_model_config(
            model_size=self.model_size,
            language=self.language,
            device=self.device
        )
        
        self._whisper = WhisperIntegration(config)
        
        logger.info(f"正在加载 Whisper 模型 (size={self.model_size}, device={self.device})...")
        
        if self._whisper.load_model():
            self._model_loaded = True
            logger.info("Whisper 模型加载成功")
            return True
        else:
            logger.error("Whisper 模型加载失败")
            return False
    
    def generate_subtitle(
        self,
        media_path: Path,
        output_path: Optional[Path] = None
    ) -> SubtitleGenerationResult:
        """
        为单个媒体文件生成字幕
        
        Args:
            media_path: 音频或视频文件路径
            output_path: 输出字幕文件路径，默认自动生成
            
        Returns:
            SubtitleGenerationResult: 生成结果
        """
        # 检查文件存在
        if not media_path.exists():
            return SubtitleGenerationResult(
                media_file=media_path,
                subtitle_path=None,
                success=False,
                error_message=f"媒体文件不存在: {media_path}"
            )
        
        # 确保模型已加载
        if not self._ensure_model_loaded():
            return SubtitleGenerationResult(
                media_file=media_path,
                subtitle_path=None,
                success=False,
                error_message="Whisper 模型加载失败"
            )
        
        # 确定输出路径
        if output_path is None:
            if self.output_dir:
                self.output_dir.mkdir(parents=True, exist_ok=True)
                output_path = self.output_dir / f"{media_path.stem}.{self.output_format.value}"
            else:
                output_path = media_path.with_suffix(f".{self.output_format.value}")
        
        logger.info(f"正在为 {media_path.name} 生成字幕...")
        
        try:
            # 调用 Whisper 进行转录
            result: TranscriptionResult = self._whisper.transcribe_audio(
                media_path,
                language=self.language if self.language != "auto" else None
            )
            
            if not result.success:
                return SubtitleGenerationResult(
                    media_file=media_path,
                    subtitle_path=None,
                    success=False,
                    error_message=result.error_message or "转录失败",
                    processing_time=result.processing_time
                )
            
            # 写入字幕文件
            self._write_subtitle_file(result, output_path)
            
            logger.info(f"字幕生成成功: {output_path}")
            
            return SubtitleGenerationResult(
                media_file=media_path,
                subtitle_path=output_path,
                success=True,
                processing_time=result.processing_time,
                detected_language=result.language,
                text_length=len(result.text),
                segment_count=len(result.segments)
            )
            
        except Exception as e:
            error_msg = f"生成字幕时发生错误: {str(e)}"
            logger.error(error_msg)
            return SubtitleGenerationResult(
                media_file=media_path,
                subtitle_path=None,
                success=False,
                error_message=error_msg
            )
    
    def _write_subtitle_file(
        self,
        result: TranscriptionResult,
        output_path: Path
    ) -> None:
        """
        将转录结果写入字幕文件
        
        Args:
            result: 转录结果
            output_path: 输出文件路径
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.output_format == SubtitleFormat.SRT:
            self._write_srt(result, output_path)
        elif self.output_format == SubtitleFormat.VTT:
            self._write_vtt(result, output_path)
        else:  # TXT
            self._write_txt(result, output_path)
    
    def _write_srt(self, result: TranscriptionResult, output_path: Path) -> None:
        """写入 SRT 格式字幕"""
        lines = []
        for i, segment in enumerate(result.segments, 1):
            start_time = self._format_timestamp_srt(float(segment.get("start", 0)))
            end_time = self._format_timestamp_srt(float(segment.get("end", 0)))
            text = str(segment.get("text", "")).strip()
            
            lines.append(str(i))
            lines.append(f"{start_time} --> {end_time}")
            lines.append(text)
            lines.append("")
        
        output_path.write_text("\n".join(lines), encoding="utf-8")
    
    def _write_vtt(self, result: TranscriptionResult, output_path: Path) -> None:
        """写入 VTT 格式字幕"""
        lines = ["WEBVTT", ""]
        for segment in result.segments:
            start_time = self._format_timestamp_vtt(float(segment.get("start", 0)))
            end_time = self._format_timestamp_vtt(float(segment.get("end", 0)))
            text = str(segment.get("text", "")).strip()
            
            lines.append(f"{start_time} --> {end_time}")
            lines.append(text)
            lines.append("")
        
        output_path.write_text("\n".join(lines), encoding="utf-8")
    
    def _write_txt(self, result: TranscriptionResult, output_path: Path) -> None:
        """写入纯文本格式"""
        output_path.write_text(result.text.strip(), encoding="utf-8")
    
    @staticmethod
    def _format_timestamp_srt(seconds: float) -> str:
        """格式化 SRT 时间戳 (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    
    @staticmethod
    def _format_timestamp_vtt(seconds: float) -> str:
        """格式化 VTT 时间戳 (HH:MM:SS.mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
    
    def batch_generate(
        self,
        media_files: list[MediaFile],
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> GenerationStatistics:
        """
        批量为媒体文件生成字幕
        
        Args:
            media_files: 媒体文件列表
            progress_callback: 进度回调函数 (current, total, filename)
            
        Returns:
            GenerationStatistics: 生成统计信息
        """
        stats = GenerationStatistics()
        total = len(media_files)
        
        logger.info(f"开始批量生成字幕，共 {total} 个文件")
        
        for i, media in enumerate(media_files, 1):
            if progress_callback:
                progress_callback(i, total, media.path.name)
            
            logger.info(f"[{i}/{total}] 处理: {media.path.name}")
            
            result = self.generate_subtitle(media.path)
            stats.add_result(result)
            
            if result.success:
                logger.info(f"  ✓ 成功 (耗时: {result.processing_time:.2f}s)")
            else:
                logger.warning(f"  ✗ 失败: {result.error_message}")
        
        logger.info(stats.summary())
        return stats
    
    def generate_for_unmatched(
        self,
        unmatched_media: list[MediaFile],
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> tuple[list[TrainDataPair], GenerationStatistics]:
        """
        为未匹配的媒体文件生成字幕，并创建训练数据对
        
        当媒体文件没有对应的字幕文件时，自动使用 Whisper 生成字幕，
        然后创建新的训练数据对。
        
        Args:
            unmatched_media: 未匹配字幕的媒体文件列表
            progress_callback: 进度回调函数
            
        Returns:
            tuple: (新创建的训练数据对列表, 生成统计信息)
        """
        # 过滤出音频和视频文件
        media_to_process = [
            m for m in unmatched_media 
            if m.media_type in (MediaType.VIDEO, MediaType.AUDIO)
        ]
        
        if not media_to_process:
            logger.info("没有需要生成字幕的媒体文件")
            return [], GenerationStatistics()
        
        logger.info(f"发现 {len(media_to_process)} 个媒体文件需要生成字幕")
        
        # 批量生成字幕
        stats = self.batch_generate(media_to_process, progress_callback)
        
        # 创建训练数据对
        new_pairs: list[TrainDataPair] = []
        for result in stats.results:
            if result.success and result.subtitle_path:
                # 找到对应的媒体文件
                media = next(
                    (m for m in media_to_process if m.path == result.media_file),
                    None
                )
                
                if media:
                    # 确定音频路径
                    if media.media_type == MediaType.AUDIO:
                        audio_path = media.path
                        source_video = None
                    else:
                        # 视频文件需要提取音频，这里先用视频路径占位
                        # 实际提取将由 TrainDataManager 完成
                        audio_path = media.path
                        source_video = media.path
                    
                    pair = TrainDataPair(
                        audio_path=audio_path,
                        subtitle_path=result.subtitle_path,
                        source_video=source_video
                    )
                    new_pairs.append(pair)
                    logger.debug(f"创建训练数据对: {pair}")
        
        logger.info(f"成功创建 {len(new_pairs)} 个新的训练数据对")
        return new_pairs, stats
    
    def get_model_info(self) -> dict:
        """获取当前模型信息"""
        if not self._model_loaded or self._whisper is None:
            return {
                "status": "未加载",
                "model_size": self.model_size,
                "language": self.language,
                "device": self.device
            }
        
        info = self._whisper.get_model_info()
        info["output_format"] = self.output_format.value
        return info
    
    def unload_model(self) -> None:
        """卸载模型以释放内存"""
        self._whisper = None
        self._model_loaded = False
        logger.info("Whisper 模型已卸载")


def main():
    """测试入口"""
    import sys
    
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if len(sys.argv) < 2:
        print("用法: python whisper_subtitle_generator.py <音频/视频文件>")
        sys.exit(1)
    
    media_path = Path(sys.argv[1])
    
    generator = WhisperSubtitleGenerator(
        model_size="base",
        language="auto",
        output_format=SubtitleFormat.SRT
    )
    
    result = generator.generate_subtitle(media_path)
    
    if result.success:
        print(f"字幕生成成功!")
        print(f"  输出文件: {result.subtitle_path}")
        print(f"  检测语言: {result.detected_language}")
        print(f"  文本长度: {result.text_length}")
        print(f"  片段数量: {result.segment_count}")
        print(f"  处理耗时: {result.processing_time:.2f}s")
    else:
        print(f"字幕生成失败: {result.error_message}")
        sys.exit(1)


if __name__ == "__main__":
    main()
