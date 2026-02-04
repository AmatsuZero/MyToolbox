"""
音频-字幕对齐处理器模块

该模块负责将音频文件与字幕文件进行对齐，
切分音频为训练片段，并生成符合 Mimic3 训练格式的数据集结构。
"""

import os
import re
import json
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from datetime import timedelta

from modules.training.train_config import TrainConfig


@dataclass
class SubtitleSegment:
    """字幕片段数据类"""
    index: int                    # 字幕序号
    start_time: float             # 开始时间（秒）
    end_time: float               # 结束时间（秒）
    text: str                     # 字幕文本
    
    @property
    def duration(self) -> float:
        """片段时长（秒）"""
        return self.end_time - self.start_time


@dataclass
class AlignedSegment:
    """对齐后的音频-文本片段"""
    index: int                    # 片段序号
    audio_path: Path              # 音频片段文件路径
    text: str                     # 对应文本
    start_time: float             # 原音频中的开始时间
    end_time: float               # 原音频中的结束时间
    duration: float               # 片段时长


@dataclass
class AlignmentResult:
    """对齐处理结果"""
    success: bool                 # 是否成功
    source_audio: Path            # 源音频文件
    source_subtitle: Path         # 源字幕文件
    output_dir: Path              # 输出目录
    segments: list[AlignedSegment] = field(default_factory=list)  # 对齐后的片段列表
    total_duration: float = 0.0   # 总音频时长
    total_segments: int = 0       # 总片段数
    error_message: str = ""       # 错误信息


class AudioAligner:
    """
    音频-字幕对齐处理器
    
    负责解析字幕文件，根据时间戳切分音频，
    并生成符合 Mimic3 训练格式的数据集。
    """
    
    # 支持的字幕格式
    SUPPORTED_SUBTITLE_FORMATS = {'.srt', '.vtt', '.txt'}
    
    # SRT 时间戳正则表达式
    SRT_TIMESTAMP_PATTERN = re.compile(
        r'(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})'
    )
    
    # VTT 时间戳正则表达式
    VTT_TIMESTAMP_PATTERN = re.compile(
        r'(\d{2}):(\d{2}):(\d{2})\.(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})\.(\d{3})'
    )
    
    def __init__(self, config: TrainConfig):
        """
        初始化对齐处理器
        
        Args:
            config: 训练配置对象
        """
        self.config = config
        self.sample_rate = config.sample_rate
    
    def align(
        self,
        audio_path: Path,
        subtitle_path: Path,
        output_dir: Optional[Path] = None
    ) -> AlignmentResult:
        """
        执行音频-字幕对齐
        
        Args:
            audio_path: 音频文件路径
            subtitle_path: 字幕文件路径
            output_dir: 输出目录，默认使用配置中的输出路径
            
        Returns:
            AlignmentResult: 对齐结果
        """
        audio_path = Path(audio_path)
        subtitle_path = Path(subtitle_path)
        output_dir = Path(output_dir) if output_dir else self.config.output_path / "aligned"
        
        # 创建输出目录
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 验证输入文件
        if not audio_path.exists():
            return AlignmentResult(
                success=False,
                source_audio=audio_path,
                source_subtitle=subtitle_path,
                output_dir=output_dir,
                error_message=f"音频文件不存在: {audio_path}"
            )
        
        if not subtitle_path.exists():
            return AlignmentResult(
                success=False,
                source_audio=audio_path,
                source_subtitle=subtitle_path,
                output_dir=output_dir,
                error_message=f"字幕文件不存在: {subtitle_path}"
            )
        
        # 解析字幕文件
        segments = self._parse_subtitle(subtitle_path)
        if not segments:
            return AlignmentResult(
                success=False,
                source_audio=audio_path,
                source_subtitle=subtitle_path,
                output_dir=output_dir,
                error_message="字幕文件解析失败或无有效内容"
            )
        
        # 切分音频
        aligned_segments = self._split_audio(audio_path, segments, output_dir)
        if not aligned_segments:
            return AlignmentResult(
                success=False,
                source_audio=audio_path,
                source_subtitle=subtitle_path,
                output_dir=output_dir,
                error_message="音频切分失败"
            )
        
        # 生成 Mimic3 训练数据格式
        self._generate_mimic3_dataset(aligned_segments, output_dir)
        
        # 计算统计信息
        total_duration = sum(seg.duration for seg in aligned_segments)
        
        return AlignmentResult(
            success=True,
            source_audio=audio_path,
            source_subtitle=subtitle_path,
            output_dir=output_dir,
            segments=aligned_segments,
            total_duration=total_duration,
            total_segments=len(aligned_segments)
        )
    
    def _parse_subtitle(self, subtitle_path: Path) -> list[SubtitleSegment]:
        """
        解析字幕文件
        
        Args:
            subtitle_path: 字幕文件路径
            
        Returns:
            字幕片段列表
        """
        suffix = subtitle_path.suffix.lower()
        
        if suffix == '.srt':
            return self._parse_srt(subtitle_path)
        elif suffix == '.vtt':
            return self._parse_vtt(subtitle_path)
        elif suffix == '.txt':
            return self._parse_txt(subtitle_path)
        else:
            print(f"不支持的字幕格式: {suffix}")
            return []
    
    def _parse_srt(self, srt_path: Path) -> list[SubtitleSegment]:
        """
        解析 SRT 字幕文件
        
        Args:
            srt_path: SRT 文件路径
            
        Returns:
            字幕片段列表
        """
        segments = []
        
        try:
            with open(srt_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            # 尝试其他编码
            with open(srt_path, 'r', encoding='latin-1') as f:
                content = f.read()
        
        # 按空行分割字幕块
        blocks = re.split(r'\n\s*\n', content.strip())
        
        for block in blocks:
            lines = block.strip().split('\n')
            if len(lines) < 2:
                continue
            
            # 查找时间戳行
            timestamp_match = None
            text_start_idx = 0
            
            for i, line in enumerate(lines):
                match = self.SRT_TIMESTAMP_PATTERN.search(line)
                if match:
                    timestamp_match = match
                    text_start_idx = i + 1
                    break
            
            if not timestamp_match:
                continue
            
            # 解析时间戳
            start_time = self._parse_timestamp(
                int(timestamp_match.group(1)),
                int(timestamp_match.group(2)),
                int(timestamp_match.group(3)),
                int(timestamp_match.group(4))
            )
            end_time = self._parse_timestamp(
                int(timestamp_match.group(5)),
                int(timestamp_match.group(6)),
                int(timestamp_match.group(7)),
                int(timestamp_match.group(8))
            )
            
            # 提取文本（可能有多行）
            text = ' '.join(lines[text_start_idx:]).strip()
            
            # 清理 HTML 标签和特殊字符
            text = self._clean_text(text)
            
            if text:
                segments.append(SubtitleSegment(
                    index=len(segments) + 1,
                    start_time=start_time,
                    end_time=end_time,
                    text=text
                ))
        
        return segments
    
    def _parse_vtt(self, vtt_path: Path) -> list[SubtitleSegment]:
        """
        解析 VTT 字幕文件
        
        Args:
            vtt_path: VTT 文件路径
            
        Returns:
            字幕片段列表
        """
        segments = []
        
        try:
            with open(vtt_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(vtt_path, 'r', encoding='latin-1') as f:
                content = f.read()
        
        # 跳过 WEBVTT 头部
        if content.startswith('WEBVTT'):
            content = content.split('\n', 1)[1] if '\n' in content else ''
        
        # 按空行分割
        blocks = re.split(r'\n\s*\n', content.strip())
        
        for block in blocks:
            lines = block.strip().split('\n')
            if not lines:
                continue
            
            # 查找时间戳
            timestamp_match = None
            text_start_idx = 0
            
            for i, line in enumerate(lines):
                match = self.VTT_TIMESTAMP_PATTERN.search(line)
                if match:
                    timestamp_match = match
                    text_start_idx = i + 1
                    break
            
            if not timestamp_match:
                continue
            
            start_time = self._parse_timestamp(
                int(timestamp_match.group(1)),
                int(timestamp_match.group(2)),
                int(timestamp_match.group(3)),
                int(timestamp_match.group(4))
            )
            end_time = self._parse_timestamp(
                int(timestamp_match.group(5)),
                int(timestamp_match.group(6)),
                int(timestamp_match.group(7)),
                int(timestamp_match.group(8))
            )
            
            text = ' '.join(lines[text_start_idx:]).strip()
            text = self._clean_text(text)
            
            if text:
                segments.append(SubtitleSegment(
                    index=len(segments) + 1,
                    start_time=start_time,
                    end_time=end_time,
                    text=text
                ))
        
        return segments
    
    def _parse_txt(self, txt_path: Path) -> list[SubtitleSegment]:
        """
        解析纯文本字幕文件（带时间戳的格式）
        
        支持格式：
        [00:00:00] 文本内容
        或
        00:00:00 - 00:00:05 文本内容
        
        Args:
            txt_path: TXT 文件路径
            
        Returns:
            字幕片段列表
        """
        segments = []
        
        # 匹配 [HH:MM:SS] 或 HH:MM:SS 格式
        timestamp_pattern = re.compile(
            r'\[?(\d{2}):(\d{2}):(\d{2})\]?\s*(?:-\s*\[?(\d{2}):(\d{2}):(\d{2})\]?)?\s*(.+)'
        )
        
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except UnicodeDecodeError:
            with open(txt_path, 'r', encoding='latin-1') as f:
                lines = f.readlines()
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            match = timestamp_pattern.match(line)
            if match:
                start_time = self._parse_timestamp(
                    int(match.group(1)),
                    int(match.group(2)),
                    int(match.group(3)),
                    0
                )
                
                # 如果有结束时间
                if match.group(4):
                    end_time = self._parse_timestamp(
                        int(match.group(4)),
                        int(match.group(5)),
                        int(match.group(6)),
                        0
                    )
                else:
                    # 默认 5 秒片段
                    end_time = start_time + 5.0
                
                text = self._clean_text(match.group(7))
                
                if text:
                    segments.append(SubtitleSegment(
                        index=len(segments) + 1,
                        start_time=start_time,
                        end_time=end_time,
                        text=text
                    ))
        
        return segments
    
    def _parse_timestamp(
        self,
        hours: int,
        minutes: int,
        seconds: int,
        milliseconds: int
    ) -> float:
        """
        将时间戳转换为秒数
        
        Args:
            hours: 小时
            minutes: 分钟
            seconds: 秒
            milliseconds: 毫秒
            
        Returns:
            总秒数（浮点数）
        """
        return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000
    
    def _clean_text(self, text: str) -> str:
        """
        清理文本内容
        
        Args:
            text: 原始文本
            
        Returns:
            清理后的文本
        """
        # 移除 HTML 标签
        text = re.sub(r'<[^>]+>', '', text)
        # 移除特殊字符（保留基本标点）
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
        # 规范化空白字符
        text = ' '.join(text.split())
        return text.strip()
    
    def _split_audio(
        self,
        audio_path: Path,
        segments: list[SubtitleSegment],
        output_dir: Path
    ) -> list[AlignedSegment]:
        """
        根据字幕时间戳切分音频
        
        Args:
            audio_path: 音频文件路径
            segments: 字幕片段列表
            output_dir: 输出目录
            
        Returns:
            对齐后的片段列表
        """
        aligned_segments = []
        audio_output_dir = output_dir / "wavs"
        audio_output_dir.mkdir(parents=True, exist_ok=True)
        
        for segment in segments:
            # 生成输出文件名
            output_filename = f"{segment.index:06d}.wav"
            output_path = audio_output_dir / output_filename
            
            # 使用 ffmpeg 切分音频
            success = self._extract_audio_segment(
                audio_path,
                output_path,
                segment.start_time,
                segment.end_time
            )
            
            if success:
                aligned_segments.append(AlignedSegment(
                    index=segment.index,
                    audio_path=output_path,
                    text=segment.text,
                    start_time=segment.start_time,
                    end_time=segment.end_time,
                    duration=segment.duration
                ))
            else:
                print(f"警告: 切分片段 {segment.index} 失败")
        
        return aligned_segments
    
    def _extract_audio_segment(
        self,
        input_path: Path,
        output_path: Path,
        start_time: float,
        end_time: float
    ) -> bool:
        """
        使用 ffmpeg 提取音频片段
        
        Args:
            input_path: 输入音频文件
            output_path: 输出文件路径
            start_time: 开始时间（秒）
            end_time: 结束时间（秒）
            
        Returns:
            是否成功
        """
        duration = end_time - start_time
        
        cmd = [
            'ffmpeg',
            '-y',                      # 覆盖输出文件
            '-i', str(input_path),     # 输入文件
            '-ss', str(start_time),    # 开始时间
            '-t', str(duration),       # 持续时间
            '-ar', str(self.sample_rate),  # 采样率
            '-ac', '1',                # 单声道
            '-acodec', 'pcm_s16le',    # PCM 格式
            '-loglevel', 'error',      # 仅显示错误
            str(output_path)
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            return output_path.exists()
        except subprocess.CalledProcessError as e:
            print(f"ffmpeg 错误: {e.stderr}")
            return False
        except FileNotFoundError:
            print("错误: 未找到 ffmpeg，请确保已安装 ffmpeg")
            return False
    
    def _generate_mimic3_dataset(
        self,
        segments: list[AlignedSegment],
        output_dir: Path
    ) -> None:
        """
        生成符合 Mimic3 训练格式的数据集
        
        Mimic3 训练数据格式：
        - wavs/ 目录包含所有音频片段
        - metadata.csv 包含文件名和对应文本
        - speaker_info.json 包含说话人信息
        
        Args:
            segments: 对齐后的片段列表
            output_dir: 输出目录
        """
        # 生成 metadata.csv
        metadata_path = output_dir / "metadata.csv"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            for segment in segments:
                # 格式: 文件名|文本|文本（规范化）
                filename = segment.audio_path.stem
                f.write(f"{filename}|{segment.text}|{segment.text}\n")
        
        # 生成 speaker_info.json
        speaker_info = {
            "name": self.config.speaker_name,
            "language": "en",
            "sample_rate": self.sample_rate,
            "num_samples": len(segments),
            "total_duration": sum(seg.duration for seg in segments)
        }
        
        speaker_info_path = output_dir / "speaker_info.json"
        with open(speaker_info_path, 'w', encoding='utf-8') as f:
            json.dump(speaker_info, f, indent=2, ensure_ascii=False)
        
        # 生成训练配置文件
        train_info = {
            "dataset_path": str(output_dir),
            "wavs_path": str(output_dir / "wavs"),
            "metadata_file": str(metadata_path),
            "speaker_name": self.config.speaker_name,
            "sample_rate": self.sample_rate,
            "segments": [
                {
                    "index": seg.index,
                    "audio": str(seg.audio_path),
                    "text": seg.text,
                    "duration": seg.duration
                }
                for seg in segments
            ]
        }
        
        train_info_path = output_dir / "train_info.json"
        with open(train_info_path, 'w', encoding='utf-8') as f:
            json.dump(train_info, f, indent=2, ensure_ascii=False)
        
        print(f"已生成 Mimic3 训练数据集:")
        print(f"  - metadata.csv: {len(segments)} 条记录")
        print(f"  - speaker_info.json: 说话人 '{self.config.speaker_name}'")
        print(f"  - train_info.json: 训练配置")
    
    def get_alignment_stats(self, result: AlignmentResult) -> dict:
        """
        获取对齐统计信息
        
        Args:
            result: 对齐结果
            
        Returns:
            统计信息字典
        """
        if not result.success:
            return {"error": result.error_message}
        
        durations = [seg.duration for seg in result.segments]
        
        return {
            "total_segments": result.total_segments,
            "total_duration": result.total_duration,
            "total_duration_formatted": str(timedelta(seconds=int(result.total_duration))),
            "average_duration": sum(durations) / len(durations) if durations else 0,
            "min_duration": min(durations) if durations else 0,
            "max_duration": max(durations) if durations else 0,
            "output_directory": str(result.output_dir)
        }


def format_time(seconds: float) -> str:
    """
    格式化时间显示
    
    Args:
        seconds: 秒数
        
    Returns:
        格式化的时间字符串 (HH:MM:SS.mmm)
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
