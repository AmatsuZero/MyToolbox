"""
字幕模型与样式定义模块。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class SubtitleSegment:
    """字幕片段类"""

    id: int
    start_time: float  # 秒
    end_time: float  # 秒
    text: str
    speaker: Optional[str] = None
    confidence: Optional[float] = None
    words: Optional[List[Dict]] = None

    def duration(self) -> float:
        """计算片段时长"""
        return self.end_time - self.start_time

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "id": self.id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "text": self.text,
            "speaker": self.speaker,
            "confidence": self.confidence,
            "duration": self.duration(),
            "words": self.words or [],
        }


@dataclass
class FontStyle:
    """字体样式类"""

    font_family: str = "Arial"
    font_size: int = 20
    font_color: str = "#FFFFFF"
    bold: bool = False
    italic: bool = False
    underline: bool = False


@dataclass
class LayoutStyle:
    """布局样式类"""

    alignment: str = "center"  # left, center, right
    position: str = "bottom"  # top, middle, bottom
    margin_vertical: int = 20
    margin_horizontal: int = 20


@dataclass
class BackgroundStyle:
    """背景样式类"""

    color: str = "#00000080"


@dataclass
class SubtitleStyle:
    """字幕样式类"""

    font: FontStyle = field(default_factory=FontStyle)
    layout: LayoutStyle = field(default_factory=LayoutStyle)
    background: BackgroundStyle = field(default_factory=BackgroundStyle)

    def __post_init__(self):
        """初始化后处理"""
        # 已经使用 default_factory 初始化，不需要额外处理
        pass

    # 属性访问器，保持向后兼容性
    @property
    def font_family(self) -> str:
        """获取字体名称"""
        return self.font.font_family

    @font_family.setter
    def font_family(self, value: str):
        """设置字体名称"""
        self.font.font_family = value

    @property
    def font_size(self) -> int:
        """获取字体大小"""
        return self.font.font_size

    @font_size.setter
    def font_size(self, value: int):
        """设置字体大小"""
        self.font.font_size = value

    @property
    def font_color(self) -> str:
        """获取字体颜色"""
        return self.font.font_color

    @font_color.setter
    def font_color(self, value: str):
        """设置字体颜色"""
        self.font.font_color = value

    @property
    def background_color(self) -> str:
        """获取背景颜色"""
        return self.background.color

    @background_color.setter
    def background_color(self, value: str):
        """设置背景颜色"""
        self.background.color = value

    @property
    def alignment(self) -> str:
        """获取文本对齐方式"""
        return self.layout.alignment

    @alignment.setter
    def alignment(self, value: str):
        """设置文本对齐方式"""
        self.layout.alignment = value

    @property
    def position(self) -> str:
        """获取字幕位置"""
        return self.layout.position

    @position.setter
    def position(self, value: str):
        """设置字幕位置"""
        self.layout.position = value

    @property
    def margin_vertical(self) -> int:
        """获取垂直边距"""
        return self.layout.margin_vertical

    @margin_vertical.setter
    def margin_vertical(self, value: int):
        """设置垂直边距"""
        self.layout.margin_vertical = value

    @property
    def margin_horizontal(self) -> int:
        """获取水平边距"""
        return self.layout.margin_horizontal

    @margin_horizontal.setter
    def margin_horizontal(self, value: int):
        """设置水平边距"""
        self.layout.margin_horizontal = value

    @property
    def bold(self) -> bool:
        """获取粗体状态"""
        return self.font.bold

    @bold.setter
    def bold(self, value: bool):
        """设置粗体状态"""
        self.font.bold = value

    @property
    def italic(self) -> bool:
        """获取斜体状态"""
        return self.font.italic

    @italic.setter
    def italic(self, value: bool):
        """设置斜体状态"""
        self.font.italic = value

    @property
    def underline(self) -> bool:
        """获取下划线状态"""
        return self.font.underline

    @underline.setter
    def underline(self, value: bool):
        """设置下划线状态"""
        self.font.underline = value
