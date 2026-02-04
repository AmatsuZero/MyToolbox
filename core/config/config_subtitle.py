"""
字幕生成器配置模块。
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class SubtitleGeneratorConfig:
    """字幕生成器配置

    使用分层结构组织配置参数，减少直接属性数量，解决"Too many instance attributes"问题。
    """

    @dataclass
    class OutputConfig:
        """输出配置"""

        directory: str = "subtitles"
        supported_formats: Optional[List[str]] = None
        default_format: str = "all"  # srt, vtt, ass, txt, json, csv, all
        encoding: str = "utf-8"

        def __post_init__(self):
            if self.supported_formats is None:
                self.supported_formats = ["srt", "vtt", "ass", "txt", "json", "csv"]

    @dataclass
    class ContentConfig:
        """内容配置"""

        include_timestamps: bool = True
        include_speaker_labels: bool = True
        include_confidence_scores: bool = True
        max_line_length: int = 42

    @dataclass
    class SegmentConfig:
        """分段配置"""

        max_duration_per_segment: float = 10.0
        min_duration_per_segment: float = 0.5
        merge_short_segments: bool = True
        merge_threshold: float = 1.0

    @dataclass
    class StyleConfig:
        """样式配置"""

        font_family: str = "Arial"
        font_size: int = 20
        font_color: str = "#FFFFFF"
        background_color: str = "#00000080"
        alignment: str = "center"
        position: str = "bottom"

    # 主要配置属性
    output: Optional[OutputConfig] = None
    content: Optional[ContentConfig] = None
    segment: Optional[SegmentConfig] = None
    style: Optional[StyleConfig] = None

    def __post_init__(self):
        if self.output is None:
            self.output = self.OutputConfig()
        if self.content is None:
            self.content = self.ContentConfig()
        if self.segment is None:
            self.segment = self.SegmentConfig()
        if self.style is None:
            self.style = self.StyleConfig()

    # 为了保持向后兼容性，提供属性访问器
    @property
    def output_directory(self) -> str:
        """获取字幕输出目录路径

        Returns:
            str: 字幕文件的输出目录路径
        """
        if self.output is None:
            return "subtitles"  # 默认值
        return self.output.directory

    @property
    def supported_formats(self) -> List[str]:
        """获取支持的格式列表

        Returns:
            List[str]: 支持的字幕格式列表
        """
        # 确保返回非空列表，避免类型错误
        if self.output is None:
            return []
        return (
            self.output.supported_formats
            if self.output.supported_formats is not None
            else []
        )

    @property
    def default_format(self) -> str:
        """获取默认字幕格式

        Returns:
            str: 默认的字幕输出格式（如srt, vtt, ass, txt, json, csv, all）
        """
        if self.output is None:
            return "all"  # 默认值
        return self.output.default_format

    @property
    def encoding(self) -> str:
        """获取字幕文件编码格式

        Returns:
            str: 字幕文件的编码格式（如utf-8, gbk等）
        """
        if self.output is None:
            return "utf-8"  # 默认值
        return self.output.encoding

    @property
    def include_timestamps(self) -> bool:
        """获取是否包含时间戳

        Returns:
            bool: 如果启用时间戳则为True，否则为False
        """
        if self.content is None:
            return True  # 默认值
        return self.content.include_timestamps

    @property
    def include_speaker_labels(self) -> bool:
        """获取是否包含说话人标签

        Returns:
            bool: 如果启用说话人标签则为True，否则为False
        """
        if self.content is None:
            return True  # 默认值
        return self.content.include_speaker_labels

    @property
    def include_confidence_scores(self) -> bool:
        """获取是否包含置信度分数

        Returns:
            bool: 如果启用置信度分数则为True，否则为False
        """
        if self.content is None:
            return True  # 默认值
        return self.content.include_confidence_scores

    @property
    def max_line_length(self) -> int:
        """获取字幕每行的最大长度

        Returns:
            int: 字幕每行的最大字符数
        """
        if self.content is None:
            return 42  # 默认值
        return self.content.max_line_length

    @property
    def max_duration_per_segment(self) -> float:
        """获取每个字幕段的最大持续时间

        Returns:
            float: 每个字幕段的最大持续时间（秒）
        """
        if self.segment is None:
            return 10.0  # 默认值
        return self.segment.max_duration_per_segment

    @property
    def min_duration_per_segment(self) -> float:
        """获取每个字幕段的最小持续时间

        Returns:
            float: 每个字幕段的最小持续时间（秒）
        """
        if self.segment is None:
            return 0.5  # 默认值
        return self.segment.min_duration_per_segment

    @property
    def merge_short_segments(self) -> bool:
        """获取是否合并短字幕段

        Returns:
            bool: 如果启用短字幕段合并则为True，否则为False
        """
        if self.segment is None:
            return True  # 默认值
        return self.segment.merge_short_segments

    @property
    def merge_threshold(self) -> float:
        """获取短字幕段合并阈值

        Returns:
            float: 短字幕段合并的时间阈值（秒）
        """
        if self.segment is None:
            return 1.0  # 默认值
        return self.segment.merge_threshold

    @property
    def style_config(self) -> Dict[str, Any]:
        """获取字幕样式配置

        Returns:
            Dict[str, Any]: 包含字幕样式相关配置的字典
        """
        if self.style is None:
            # 返回默认样式配置
            return {
                "font_family": "Arial",
                "font_size": 20,
                "font_color": "#FFFFFF",
                "background_color": "#00000080",
                "alignment": "center",
                "position": "bottom",
            }
        return {
            "font_family": self.style.font_family,
            "font_size": self.style.font_size,
            "font_color": self.style.font_color,
            "background_color": self.style.background_color,
            "alignment": self.style.alignment,
            "position": self.style.position,
        }
