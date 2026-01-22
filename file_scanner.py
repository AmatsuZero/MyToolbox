"""
媒体文件扫描模块 - 用于扫描、识别和管理音视频文件

该模块提供了用于扫描目录中的媒体文件（音频和视频）的功能，
支持多种常见格式如MP3、WAV、MP4、MKV等。主要功能包括：
- 递归或非递归扫描目录
- 识别和分类音频和视频文件
- 提供文件统计信息
- 按类型和大小过滤文件
"""

from pathlib import Path
from typing import List, Dict, Union, Optional
from dataclasses import dataclass


@dataclass
class MediaFile:
    """媒体文件信息类"""

    path: Path
    filename: str
    extension: str
    size: int
    file_type: str  # 'audio' 或 'video'
    supported: bool

    def __str__(self) -> str:
        return f"{self.filename} ({self.file_type}, {self.size / (1024*1024):.2f}MB)"


class MediaFileScanner:
    """媒体文件扫描器"""

    # 支持的音频格式
    AUDIO_EXTENSIONS = {
        ".mp3",
        ".wav",
        ".flac",
        ".aac",
        ".ogg",
        ".m4a",
        ".wma",
        ".opus",
    }

    # 支持的视频格式
    VIDEO_EXTENSIONS = {
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

    # 所有支持的格式
    SUPPORTED_EXTENSIONS = AUDIO_EXTENSIONS.union(VIDEO_EXTENSIONS)

    def __init__(self, scan_directory: str = "downloads"):
        self.scan_directory = Path(scan_directory)
        self.media_files: List[MediaFile] = []

    def scan_directory_exists(self) -> bool:
        """检查扫描目录是否存在"""
        return self.scan_directory.exists() and self.scan_directory.is_dir()

    def get_file_type(self, extension: str) -> str:
        """根据文件扩展名判断文件类型"""
        if extension.lower() in self.AUDIO_EXTENSIONS:
            return "audio"
        elif extension.lower() in self.VIDEO_EXTENSIONS:
            return "video"
        else:
            return "unknown"

    def is_supported_format(self, extension: str) -> bool:
        """检查文件格式是否支持"""
        return extension.lower() in self.SUPPORTED_EXTENSIONS

    def scan_files(self, recursive: bool = True) -> List[MediaFile]:
        """扫描目录中的媒体文件"""
        if not self.scan_directory_exists():
            print(f"警告: 扫描目录 '{self.scan_directory}' 不存在")
            return []

        self.media_files.clear()

        # 选择扫描方法
        scan_method = (
            self.scan_directory.rglob if recursive else self.scan_directory.glob
        )

        for file_path in scan_method("*"):
            if file_path.is_file():
                extension = file_path.suffix.lower()

                if self.is_supported_format(extension):
                    file_type = self.get_file_type(extension)

                    media_file = MediaFile(
                        path=file_path,
                        filename=file_path.name,
                        extension=extension,
                        size=file_path.stat().st_size,
                        file_type=file_type,
                        supported=True,
                    )

                    self.media_files.append(media_file)

        return self.media_files

    def get_statistics(self) -> Dict[str, Union[int, float]]:
        """获取扫描统计信息"""
        audio_count = sum(1 for f in self.media_files if f.file_type == "audio")
        video_count = sum(1 for f in self.media_files if f.file_type == "video")
        total_size = sum(f.size for f in self.media_files)

        return {
            "total_files": len(self.media_files),
            "audio_files": audio_count,
            "video_files": video_count,
            "total_size_mb": total_size / (1024 * 1024),
        }

    def filter_by_type(self, file_type: str) -> List[MediaFile]:
        """按文件类型过滤"""
        return [f for f in self.media_files if f.file_type == file_type]

    def filter_by_size(
        self, min_size: float = 0, max_size: Optional[float] = None
    ) -> List[MediaFile]:
        """按文件大小过滤"""
        filtered = []
        for f in self.media_files:
            if f.size >= min_size:
                if max_size is None or (max_size is not None and f.size <= max_size):
                    filtered.append(f)
        return filtered

    def print_summary(self):
        """打印扫描摘要"""
        stats = self.get_statistics()

        print(f"扫描目录: {self.scan_directory}")
        print(f"找到文件: {stats['total_files']} 个")
        print(f"音频文件: {stats['audio_files']} 个")
        print(f"视频文件: {stats['video_files']} 个")
        print(f"总大小: {stats['total_size_mb']:.2f} MB")

        if self.media_files:
            print("\n文件列表:")
            for i, media_file in enumerate(self.media_files, 1):
                print(f"{i:2d}. {media_file}")


def scan_media_files(
    directory: str = "downloads", recursive: bool = True
) -> List[MediaFile]:
    """快速扫描媒体文件的便捷函数"""
    scanner = MediaFileScanner(directory)
    return scanner.scan_files(recursive)


if __name__ == "__main__":
    # 测试代码
    test_scanner = MediaFileScanner("downloads")
    files = test_scanner.scan_files()
    test_scanner.print_summary()
