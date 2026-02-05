"""
训练数据管理模块

负责扫描、组织和管理用于语音模型训练的数据文件。
支持视频、音频和字幕文件的自动匹配与处理。
支持生成 LJSpeech 格式数据集（Coqui TTS 所需格式）。
"""

import os
import csv
import subprocess
import logging
import shutil
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
from enum import Enum

logger = logging.getLogger(__name__)


class MediaType(Enum):
    """媒体文件类型枚举"""
    VIDEO = "video"
    AUDIO = "audio"
    SUBTITLE = "subtitle"


@dataclass
class MediaFile:
    """媒体文件信息"""
    path: Path
    media_type: MediaType
    stem: str  # 不含扩展名的文件名，用于匹配
    
    @classmethod
    def from_path(cls, path: Path, media_type: MediaType) -> "MediaFile":
        """从路径创建媒体文件对象"""
        return cls(
            path=path,
            media_type=media_type,
            stem=path.stem
        )


@dataclass
class TrainDataPair:
    """训练数据对（音频 + 字幕）"""
    audio_path: Path
    subtitle_path: Path
    source_video: Optional[Path] = None  # 如果音频是从视频提取的，记录源视频
    
    def __str__(self) -> str:
        return f"TrainDataPair(audio={self.audio_path.name}, subtitle={self.subtitle_path.name})"


@dataclass
class ScanResult:
    """扫描结果"""
    video_files: list[MediaFile] = field(default_factory=list)
    audio_files: list[MediaFile] = field(default_factory=list)
    subtitle_files: list[MediaFile] = field(default_factory=list)
    
    @property
    def total_count(self) -> int:
        """总文件数"""
        return len(self.video_files) + len(self.audio_files) + len(self.subtitle_files)
    
    def summary(self) -> str:
        """生成扫描摘要"""
        return (
            f"扫描结果: "
            f"视频文件 {len(self.video_files)} 个, "
            f"音频文件 {len(self.audio_files)} 个, "
            f"字幕文件 {len(self.subtitle_files)} 个"
        )


@dataclass
class DatasetStats:
    """数据集统计信息"""
    total_clips: int = 0            # 总片段数
    total_duration: float = 0.0     # 总时长（秒）
    avg_duration: float = 0.0       # 平均片段时长（秒）
    min_duration: float = 0.0       # 最短片段时长
    max_duration: float = 0.0       # 最长片段时长
    total_characters: int = 0       # 总字符数
    unique_characters: int = 0      # 唯一字符数
    vocabulary_size: int = 0        # 词汇量（中文按字计算）
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "total_clips": self.total_clips,
            "total_duration_seconds": round(self.total_duration, 2),
            "total_duration_formatted": self._format_duration(self.total_duration),
            "avg_duration": round(self.avg_duration, 2),
            "min_duration": round(self.min_duration, 2),
            "max_duration": round(self.max_duration, 2),
            "total_characters": self.total_characters,
            "unique_characters": self.unique_characters,
            "vocabulary_size": self.vocabulary_size,
        }
    
    @staticmethod
    def _format_duration(seconds: float) -> str:
        """格式化时长"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if hours > 0:
            return f"{hours}小时{minutes}分{secs}秒"
        elif minutes > 0:
            return f"{minutes}分{secs}秒"
        else:
            return f"{secs}秒"


class TrainDataManager:
    """训练数据管理器"""
    
    # 支持的文件扩展名
    VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm'}
    AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'}
    SUBTITLE_EXTENSIONS = {'.srt', '.vtt', '.txt', '.ass', '.ssa'}
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        初始化训练数据管理器
        
        Args:
            output_dir: 输出目录，用于存放提取的音频等临时文件
        """
        self.output_dir = output_dir or Path("./train_output")
        self._extracted_audio_dir = self.output_dir / "extracted_audio"
        
    def scan_directory(self, directory: Path, recursive: bool = True) -> ScanResult:
        """
        扫描目录，查找所有支持的媒体文件
        
        Args:
            directory: 要扫描的目录
            recursive: 是否递归扫描子目录
            
        Returns:
            ScanResult: 扫描结果
        """
        result = ScanResult()
        
        if not directory.exists():
            logger.error(f"目录不存在: {directory}")
            return result
            
        if not directory.is_dir():
            logger.error(f"路径不是目录: {directory}")
            return result
        
        # 选择遍历方法
        if recursive:
            files = directory.rglob("*")
        else:
            files = directory.glob("*")
        
        for file_path in files:
            if not file_path.is_file():
                continue
                
            suffix = file_path.suffix.lower()
            
            if suffix in self.VIDEO_EXTENSIONS:
                result.video_files.append(
                    MediaFile.from_path(file_path, MediaType.VIDEO)
                )
            elif suffix in self.AUDIO_EXTENSIONS:
                result.audio_files.append(
                    MediaFile.from_path(file_path, MediaType.AUDIO)
                )
            elif suffix in self.SUBTITLE_EXTENSIONS:
                result.subtitle_files.append(
                    MediaFile.from_path(file_path, MediaType.SUBTITLE)
                )
        
        logger.info(result.summary())
        return result
    
    def scan_files(
        self,
        video_files: Optional[list[Path]] = None,
        audio_files: Optional[list[Path]] = None,
        subtitle_files: Optional[list[Path]] = None
    ) -> ScanResult:
        """
        从指定的文件列表创建扫描结果
        
        Args:
            video_files: 视频文件列表
            audio_files: 音频文件列表
            subtitle_files: 字幕文件列表
            
        Returns:
            ScanResult: 扫描结果
        """
        result = ScanResult()
        
        for path in (video_files or []):
            if path.exists() and path.is_file():
                result.video_files.append(
                    MediaFile.from_path(path, MediaType.VIDEO)
                )
            else:
                logger.warning(f"视频文件不存在: {path}")
                
        for path in (audio_files or []):
            if path.exists() and path.is_file():
                result.audio_files.append(
                    MediaFile.from_path(path, MediaType.AUDIO)
                )
            else:
                logger.warning(f"音频文件不存在: {path}")
                
        for path in (subtitle_files or []):
            if path.exists() and path.is_file():
                result.subtitle_files.append(
                    MediaFile.from_path(path, MediaType.SUBTITLE)
                )
            else:
                logger.warning(f"字幕文件不存在: {path}")
        
        logger.info(result.summary())
        return result
    
    def match_audio_subtitle(
        self,
        scan_result: ScanResult,
        extract_from_video: bool = True
    ) -> tuple[list[TrainDataPair], list[MediaFile]]:
        """
        匹配音频和字幕文件
        
        基于文件名（不含扩展名）进行匹配。如果设置了 extract_from_video，
        会自动从视频文件中提取音频。
        
        Args:
            scan_result: 扫描结果
            extract_from_video: 是否从视频提取音频
            
        Returns:
            tuple: (匹配成功的数据对列表, 未匹配的文件列表)
        """
        matched_pairs: list[TrainDataPair] = []
        unmatched: list[MediaFile] = []
        
        # 构建字幕文件的查找字典（按stem索引）
        subtitle_map: dict[str, MediaFile] = {}
        for sub in scan_result.subtitle_files:
            # 使用小写的stem作为key，支持大小写不敏感匹配
            key = sub.stem.lower()
            if key in subtitle_map:
                logger.warning(f"发现同名字幕文件: {sub.path} 和 {subtitle_map[key].path}")
            subtitle_map[key] = sub
        
        # 已匹配的字幕stem集合
        matched_subtitle_stems: set[str] = set()
        
        # 1. 匹配现有音频文件
        for audio in scan_result.audio_files:
            key = audio.stem.lower()
            if key in subtitle_map:
                matched_pairs.append(TrainDataPair(
                    audio_path=audio.path,
                    subtitle_path=subtitle_map[key].path
                ))
                matched_subtitle_stems.add(key)
                logger.debug(f"匹配成功: {audio.path.name} <-> {subtitle_map[key].path.name}")
            else:
                unmatched.append(audio)
                logger.debug(f"未找到匹配字幕: {audio.path.name}")
        
        # 2. 处理视频文件
        if extract_from_video:
            for video in scan_result.video_files:
                key = video.stem.lower()
                if key in subtitle_map and key not in matched_subtitle_stems:
                    # 有对应字幕，提取音频
                    try:
                        audio_path = self.extract_audio_from_video(video.path)
                        matched_pairs.append(TrainDataPair(
                            audio_path=audio_path,
                            subtitle_path=subtitle_map[key].path,
                            source_video=video.path
                        ))
                        matched_subtitle_stems.add(key)
                        logger.info(f"从视频提取音频并匹配: {video.path.name}")
                    except Exception as e:
                        logger.error(f"从视频提取音频失败: {video.path}, 错误: {e}")
                        unmatched.append(video)
                else:
                    unmatched.append(video)
                    if key in matched_subtitle_stems:
                        logger.debug(f"字幕已被使用: {video.path.name}")
                    else:
                        logger.debug(f"未找到匹配字幕: {video.path.name}")
        else:
            unmatched.extend(scan_result.video_files)
        
        # 3. 检查未匹配的字幕文件
        for key, sub in subtitle_map.items():
            if key not in matched_subtitle_stems:
                unmatched.append(sub)
                logger.debug(f"未找到匹配音频/视频: {sub.path.name}")
        
        logger.info(f"匹配完成: {len(matched_pairs)} 对成功, {len(unmatched)} 个未匹配")
        return matched_pairs, unmatched
    
    def extract_audio_from_video(
        self,
        video_path: Path,
        output_format: str = "wav",
        sample_rate: int = 22050
    ) -> Path:
        """
        从视频文件中提取音频
        
        Args:
            video_path: 视频文件路径
            output_format: 输出音频格式
            sample_rate: 采样率
            
        Returns:
            Path: 提取的音频文件路径
            
        Raises:
            RuntimeError: 提取失败时抛出
        """
        # 确保输出目录存在
        self._extracted_audio_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成输出文件名
        output_path = self._extracted_audio_dir / f"{video_path.stem}.{output_format}"
        
        # 如果已存在，直接返回
        if output_path.exists():
            logger.info(f"音频文件已存在，跳过提取: {output_path}")
            return output_path
        
        # 使用 ffmpeg 提取音频
        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-vn",  # 不处理视频
            "-acodec", "pcm_s16le" if output_format == "wav" else "libmp3lame",
            "-ar", str(sample_rate),
            "-ac", "1",  # 单声道
            "-y",  # 覆盖已存在的文件
            str(output_path)
        ]
        
        logger.info(f"正在从视频提取音频: {video_path.name}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            logger.info(f"音频提取成功: {output_path}")
            return output_path
        except subprocess.CalledProcessError as e:
            logger.error(f"ffmpeg 执行失败: {e.stderr}")
            raise RuntimeError(f"从视频提取音频失败: {video_path}") from e
        except FileNotFoundError:
            logger.error("未找到 ffmpeg，请确保已安装 ffmpeg 并添加到 PATH")
            raise RuntimeError("未找到 ffmpeg，请先安装 ffmpeg")
    
    def get_unmatched_media_for_whisper(
        self,
        unmatched: list[MediaFile]
    ) -> list[MediaFile]:
        """
        获取需要使用 Whisper 生成字幕的媒体文件
        
        筛选出没有字幕的视频和音频文件，这些文件需要先用 Whisper
        生成字幕后才能用于训练。
        
        Args:
            unmatched: 未匹配的文件列表
            
        Returns:
            list[MediaFile]: 需要生成字幕的媒体文件列表
        """
        result = []
        for media in unmatched:
            if media.media_type in (MediaType.VIDEO, MediaType.AUDIO):
                result.append(media)
        
        if result:
            logger.info(f"发现 {len(result)} 个媒体文件需要使用 Whisper 生成字幕")
        
        return result
    
    def validate_data_pairs(
        self,
        pairs: list[TrainDataPair]
    ) -> tuple[list[TrainDataPair], list[TrainDataPair]]:
        """
        验证训练数据对的有效性
        
        检查文件是否存在、是否可读等。
        
        Args:
            pairs: 训练数据对列表
            
        Returns:
            tuple: (有效的数据对, 无效的数据对)
        """
        valid = []
        invalid = []
        
        for pair in pairs:
            if not pair.audio_path.exists():
                logger.error(f"音频文件不存在: {pair.audio_path}")
                invalid.append(pair)
                continue
                
            if not pair.subtitle_path.exists():
                logger.error(f"字幕文件不存在: {pair.subtitle_path}")
                invalid.append(pair)
                continue
            
            # 检查文件大小
            if pair.audio_path.stat().st_size == 0:
                logger.error(f"音频文件为空: {pair.audio_path}")
                invalid.append(pair)
                continue
                
            if pair.subtitle_path.stat().st_size == 0:
                logger.error(f"字幕文件为空: {pair.subtitle_path}")
                invalid.append(pair)
                continue
            
            valid.append(pair)
        
        logger.info(f"数据验证完成: {len(valid)} 个有效, {len(invalid)} 个无效")
        return valid, invalid
    
    def get_statistics(self, pairs: list[TrainDataPair]) -> dict:
        """
        获取训练数据统计信息
        
        Args:
            pairs: 训练数据对列表
            
        Returns:
            dict: 统计信息字典
        """
        total_audio_size = 0
        total_subtitle_size = 0
        
        for pair in pairs:
            if pair.audio_path.exists():
                total_audio_size += pair.audio_path.stat().st_size
            if pair.subtitle_path.exists():
                total_subtitle_size += pair.subtitle_path.stat().st_size
        
        return {
            "pair_count": len(pairs),
            "total_audio_size_mb": round(total_audio_size / (1024 * 1024), 2),
            "total_subtitle_size_kb": round(total_subtitle_size / 1024, 2),
            "from_video_count": sum(1 for p in pairs if p.source_video is not None)
        }
    
    # ========== LJSpeech 格式数据集生成功能（Coqui TTS 支持）==========
    
    def generate_ljspeech_dataset(
        self,
        aligned_data: List[Dict],
        output_dir: Path,
        sample_rate: int = 22050,
        max_duration: float = 10.0,
        min_duration: float = 0.5
    ) -> Tuple[Path, DatasetStats]:
        """
        生成 LJSpeech 格式的数据集
        
        LJSpeech 格式是 Coqui TTS 支持的标准数据集格式。
        
        目录结构:
        output_dir/
        ├── wavs/
        │   ├── audio_0001.wav
        │   ├── audio_0002.wav
        │   └── ...
        └── metadata.csv
        
        Args:
            aligned_data: 对齐后的数据列表，每项包含 audio_path, text, start_time, end_time
            output_dir: 输出目录
            sample_rate: 目标采样率
            max_duration: 最大片段时长（秒）
            min_duration: 最小片段时长（秒）
            
        Returns:
            Tuple[Path, DatasetStats]: (数据集路径, 统计信息)
        """
        logger.info(f"开始生成 LJSpeech 格式数据集: {output_dir}")
        
        # 创建目录结构
        output_dir = Path(output_dir)
        wavs_dir = output_dir / "wavs"
        wavs_dir.mkdir(parents=True, exist_ok=True)
        
        # 准备统计数据
        stats = DatasetStats()
        durations = []
        all_text = ""
        unique_chars = set()
        
        # 处理数据并写入 metadata.csv
        metadata_path = output_dir / "metadata.csv"
        valid_entries = []
        
        for idx, item in enumerate(aligned_data):
            audio_path = Path(item.get("audio_path", ""))
            text = item.get("text", "").strip()
            start_time = item.get("start_time", 0)
            end_time = item.get("end_time", 0)
            
            if not audio_path.exists():
                logger.warning(f"音频文件不存在，跳过: {audio_path}")
                continue
            
            if not text:
                logger.warning(f"文本为空，跳过: {audio_path}")
                continue
            
            # 计算时长
            duration = end_time - start_time if end_time > start_time else self._get_audio_duration(audio_path)
            
            if duration < min_duration:
                logger.debug(f"片段时长 {duration:.2f}s 小于最小值 {min_duration}s，跳过")
                continue
            
            if duration > max_duration:
                logger.debug(f"片段时长 {duration:.2f}s 大于最大值 {max_duration}s，跳过")
                continue
            
            # 生成文件名
            clip_id = f"audio_{idx+1:05d}"
            output_wav = wavs_dir / f"{clip_id}.wav"
            
            # 转换音频格式
            try:
                success = self._convert_audio(audio_path, output_wav, sample_rate, start_time, end_time)
                if not success:
                    logger.warning(f"音频转换产生无效输出，跳过: {audio_path}")
                    continue
            except Exception as e:
                logger.error(f"音频转换失败: {e}")
                continue
            
            # 清理文本
            cleaned_text = self._clean_text_for_tts(text)
            
            # 记录统计数据
            durations.append(duration)
            all_text += cleaned_text
            unique_chars.update(cleaned_text)
            
            valid_entries.append({
                "id": clip_id,
                "text": cleaned_text,
                "normalized_text": cleaned_text  # LJSpeech 格式需要两列文本
            })
        
        # 写入 metadata.csv
        with open(metadata_path, 'w', newline='', encoding='utf-8') as f:
            for entry in valid_entries:
                # LJSpeech 格式: file_id|text|normalized_text
                f.write(f"{entry['id']}|{entry['text']}|{entry['normalized_text']}\n")
        
        # 计算统计信息
        if durations:
            stats.total_clips = len(valid_entries)
            stats.total_duration = sum(durations)
            stats.avg_duration = stats.total_duration / len(durations)
            stats.min_duration = min(durations)
            stats.max_duration = max(durations)
        
        stats.total_characters = len(all_text)
        stats.unique_characters = len(unique_chars)
        stats.vocabulary_size = len(set(all_text.split())) if ' ' in all_text else len(unique_chars)
        
        logger.info(f"数据集生成完成: {stats.total_clips} 个片段, 总时长: {stats.total_duration:.2f}秒")
        
        # 保存统计信息
        stats_path = output_dir / "dataset_stats.json"
        import json
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats.to_dict(), f, ensure_ascii=False, indent=2)
        
        return output_dir, stats
    
    def generate_ljspeech_from_pairs(
        self,
        pairs: List[TrainDataPair],
        output_dir: Path,
        sample_rate: int = 22050,
        max_duration: float = 10.0,
        min_duration: float = 0.5
    ) -> Tuple[Path, DatasetStats]:
        """
        从训练数据对生成 LJSpeech 格式数据集
        
        此方法会解析字幕文件，切割音频，并生成 LJSpeech 格式数据集。
        
        Args:
            pairs: 训练数据对列表
            output_dir: 输出目录
            sample_rate: 目标采样率
            max_duration: 最大片段时长（秒）
            min_duration: 最小片段时长（秒）
            
        Returns:
            Tuple[Path, DatasetStats]: (数据集路径, 统计信息)
        """
        logger.info(f"从 {len(pairs)} 个数据对生成 LJSpeech 数据集")
        
        aligned_data = []
        
        for pair in pairs:
            # 解析字幕
            subtitles = self._parse_subtitle(pair.subtitle_path)
            
            for sub in subtitles:
                aligned_data.append({
                    "audio_path": pair.audio_path,
                    "text": sub["text"],
                    "start_time": sub["start"],
                    "end_time": sub["end"]
                })
        
        return self.generate_ljspeech_dataset(
            aligned_data, output_dir, sample_rate, max_duration, min_duration
        )
    
    def _parse_subtitle(self, subtitle_path: Path) -> List[Dict]:
        """
        解析字幕文件
        
        Args:
            subtitle_path: 字幕文件路径
            
        Returns:
            List[Dict]: 字幕条目列表，每项包含 text, start, end
        """
        suffix = subtitle_path.suffix.lower()
        
        if suffix == '.srt':
            return self._parse_srt(subtitle_path)
        elif suffix == '.vtt':
            return self._parse_vtt(subtitle_path)
        elif suffix == '.txt':
            return self._parse_txt(subtitle_path)
        else:
            logger.warning(f"不支持的字幕格式: {suffix}，尝试作为文本解析")
            return self._parse_txt(subtitle_path)
    
    def _parse_srt(self, srt_path: Path) -> List[Dict]:
        """解析 SRT 字幕文件"""
        entries = []
        
        try:
            with open(srt_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(srt_path, 'r', encoding='gbk') as f:
                content = f.read()
        
        # SRT 格式: 序号 -> 时间码 -> 文本 -> 空行
        blocks = content.strip().split('\n\n')
        
        for block in blocks:
            lines = block.strip().split('\n')
            if len(lines) >= 3:
                # 解析时间码
                time_line = lines[1]
                if ' --> ' in time_line:
                    start_str, end_str = time_line.split(' --> ')
                    start = self._parse_srt_time(start_str.strip())
                    end = self._parse_srt_time(end_str.strip())
                    text = ' '.join(lines[2:])
                    
                    entries.append({
                        "text": text,
                        "start": start,
                        "end": end
                    })
        
        return entries
    
    def _parse_srt_time(self, time_str: str) -> float:
        """解析 SRT 时间格式 (00:00:00,000)"""
        # 移除可能的逗号后的毫秒
        time_str = time_str.replace(',', '.')
        
        parts = time_str.split(':')
        if len(parts) == 3:
            hours = float(parts[0])
            minutes = float(parts[1])
            seconds = float(parts[2])
            return hours * 3600 + minutes * 60 + seconds
        return 0.0
    
    def _parse_vtt(self, vtt_path: Path) -> List[Dict]:
        """解析 VTT 字幕文件"""
        entries = []
        
        try:
            with open(vtt_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(vtt_path, 'r', encoding='gbk') as f:
                content = f.read()
        
        # 跳过 WEBVTT 头
        if content.startswith('WEBVTT'):
            content = content.split('\n\n', 1)[1] if '\n\n' in content else content
        
        blocks = content.strip().split('\n\n')
        
        for block in blocks:
            lines = block.strip().split('\n')
            for i, line in enumerate(lines):
                if ' --> ' in line:
                    start_str, end_str = line.split(' --> ')
                    start = self._parse_vtt_time(start_str.strip())
                    end = self._parse_vtt_time(end_str.strip().split()[0])  # 移除可能的位置信息
                    text = ' '.join(lines[i+1:])
                    
                    entries.append({
                        "text": text,
                        "start": start,
                        "end": end
                    })
                    break
        
        return entries
    
    def _parse_vtt_time(self, time_str: str) -> float:
        """解析 VTT 时间格式 (00:00:00.000)"""
        parts = time_str.split(':')
        if len(parts) == 3:
            hours = float(parts[0])
            minutes = float(parts[1])
            seconds = float(parts[2])
            return hours * 3600 + minutes * 60 + seconds
        elif len(parts) == 2:
            minutes = float(parts[0])
            seconds = float(parts[1])
            return minutes * 60 + seconds
        return 0.0
    
    def _parse_txt(self, txt_path: Path) -> List[Dict]:
        """解析纯文本文件（整个文件作为一个条目）"""
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
        except UnicodeDecodeError:
            with open(txt_path, 'r', encoding='gbk') as f:
                content = f.read().strip()
        
        # 整个文件作为一个条目，不包含时间信息
        return [{
            "text": content,
            "start": 0,
            "end": 0
        }]
    
    def _get_audio_duration(self, audio_path: Path) -> float:
        """获取音频时长"""
        try:
            result = subprocess.run(
                ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
                 '-of', 'default=noprint_wrappers=1:nokey=1', str(audio_path)],
                capture_output=True, text=True
            )
            return float(result.stdout.strip())
        except Exception:
            return 0.0
    
    def _convert_audio(
        self,
        input_path: Path,
        output_path: Path,
        sample_rate: int,
        start_time: float = 0,
        end_time: float = 0
    ) -> bool:
        """
        转换音频格式
        
        Args:
            input_path: 输入音频路径
            output_path: 输出音频路径
            sample_rate: 目标采样率
            start_time: 起始时间（秒），仅用于未切割的原始音频
            end_time: 结束时间（秒），仅用于未切割的原始音频
            
        Returns:
            bool: 转换是否成功且输出文件有效
        """
        try:
            cmd = ['ffmpeg', '-y', '-i', str(input_path)]
            
            # 检查是否是已经切割好的音频片段（来自 aligned 目录）
            # 如果是，则不应该再使用 start_time 和 end_time 进行二次切割
            is_aligned_segment = 'aligned' in str(input_path) and '/wavs/' in str(input_path)
            
            # 只有在处理原始音频时才添加时间裁剪参数
            if not is_aligned_segment:
                if start_time > 0:
                    cmd.extend(['-ss', str(start_time)])
                if end_time > start_time:
                    cmd.extend(['-t', str(end_time - start_time)])
            
            # 输出格式参数
            cmd.extend([
                '-ar', str(sample_rate),
                '-ac', '1',  # 单声道
                '-acodec', 'pcm_s16le',
                str(output_path)
            ])
            
            # 执行转换
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # 检查 ffmpeg 返回码
            if result.returncode != 0:
                logger.warning(f"ffmpeg 转换失败 (返回码 {result.returncode}): {input_path}")
                if result.stderr:
                    logger.debug(f"ffmpeg 错误信息: {result.stderr[:500]}")
                # 清理可能产生的无效文件
                if output_path.exists():
                    output_path.unlink()
                return False
            
            # 验证输出文件
            if not output_path.exists():
                logger.warning(f"音频转换失败，输出文件不存在: {output_path}")
                return False
            
            # 检查文件大小（WAV 文件头至少 44 字节）
            file_size = output_path.stat().st_size
            if file_size < 1000:  # 小于 1KB 的音频文件可能是空的
                logger.warning(f"输出音频文件过小 ({file_size} 字节)，可能是空文件: {output_path}")
                output_path.unlink()  # 删除无效文件
                return False
            
            # 验证音频时长
            duration = self._get_audio_duration(output_path)
            if duration < 0.1:  # 音频时长小于 0.1 秒认为无效
                logger.warning(f"输出音频时长过短 ({duration:.3f}s)，跳过: {output_path}")
                output_path.unlink()  # 删除无效文件
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"音频转换异常: {e}")
            # 清理可能产生的无效文件
            if output_path.exists():
                output_path.unlink()
            return False
    
    def _clean_text_for_tts(self, text: str) -> str:
        """
        清理文本用于 TTS 训练
        
        Args:
            text: 原始文本
            
        Returns:
            str: 清理后的文本
        """
        import re
        
        # 移除特殊标记（如 [音乐]、[笑声] 等）
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'\(.*?\)', '', text)
        text = re.sub(r'（.*?）', '', text)
        
        # 移除多余的空白
        text = ' '.join(text.split())
        
        # 移除特殊字符（保留基本标点）
        text = re.sub(r'[^\w\s\u4e00-\u9fff，。！？、；：""'',.!?;:\'\"-]', '', text)
        
        return text.strip()


def main():
    """测试入口"""
    import sys
    
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if len(sys.argv) < 2:
        print("用法: python train_data_manager.py <目录路径>")
        sys.exit(1)
    
    directory = Path(sys.argv[1])
    manager = TrainDataManager()
    
    # 扫描目录
    scan_result = manager.scan_directory(directory)
    print(scan_result.summary())
    
    # 匹配文件
    pairs, unmatched = manager.match_audio_subtitle(scan_result)
    
    print(f"\n匹配成功的数据对:")
    for pair in pairs:
        print(f"  {pair}")
    
    print(f"\n未匹配的文件:")
    for media in unmatched:
        print(f"  [{media.media_type.value}] {media.path.name}")
    
    # 统计信息
    if pairs:
        stats = manager.get_statistics(pairs)
        print(f"\n统计信息: {stats}")


if __name__ == "__main__":
    main()
