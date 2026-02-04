"""
训练数据管理模块

负责扫描、组织和管理用于 Mimic3 语音模型训练的数据文件。
支持视频、音频和字幕文件的自动匹配与处理。
"""

import os
import subprocess
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
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
