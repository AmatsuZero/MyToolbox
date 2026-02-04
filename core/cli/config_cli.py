"""
命令行配置模块。
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class CLIConfig:
    """命令行接口配置

    使用分层结构组织配置参数，减少直接属性数量，解决"Too many instance attributes"问题。
    """

    @dataclass
    class DisplayConfig:
        """显示相关配置"""

        verbose: bool = False
        quiet: bool = False
        color_output: bool = True
        progress_bar: bool = True
        output_format: str = "table"  # table, json, yaml, csv
        max_display_files: int = 50
        show_file_details: bool = False

    @dataclass
    class InteractionConfig:
        """交互相关配置"""

        interactive: bool = False
        confirm_prompts: bool = True
        enable_keyboard_shortcuts: bool = True
        auto_complete: bool = True

    @dataclass
    class LoggingConfig:
        """日志和历史相关配置"""

        log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
        history_file: str = "./history.json"
        max_history_entries: int = 1000

    # 主要配置属性
    display: Optional[DisplayConfig] = None
    interaction: Optional[InteractionConfig] = None
    logging: Optional[LoggingConfig] = None
    dry_run: bool = False  # 执行模式相关，单独保留

    def __post_init__(self):
        if self.display is None:
            self.display = self.DisplayConfig()
        if self.interaction is None:
            self.interaction = self.InteractionConfig()
        if self.logging is None:
            self.logging = self.LoggingConfig()

    # 为了保持向后兼容性，提供属性访问器
    @property
    def verbose(self) -> bool:
        """获取详细输出模式状态

        Returns:
            bool: 如果启用详细输出模式则为True，否则为False
        """
        if self.display is None:
            return False  # 默认值
        return self.display.verbose

    @property
    def quiet(self) -> bool:
        """获取安静模式状态

        Returns:
            bool: 如果启用安静模式则为True，否则为False
        """
        if self.display is None:
            return False  # 默认值
        return self.display.quiet

    @property
    def color_output(self) -> bool:
        """获取彩色输出状态

        Returns:
            bool: 如果启用彩色输出则为True，否则为False
        """
        if self.display is None:
            return True  # 默认值
        return self.display.color_output

    @property
    def progress_bar(self) -> bool:
        """获取进度条显示状态

        Returns:
            bool: 如果启用进度条显示则为True，否则为False
        """
        if self.display is None:
            return True  # 默认值
        return self.display.progress_bar

    @property
    def output_format(self) -> str:
        """获取输出格式

        Returns:
            str: 输出格式（table, json, yaml, csv等）
        """
        if self.display is None:
            return "table"  # 默认值
        return self.display.output_format

    @property
    def max_display_files(self) -> int:
        """获取最大显示文件数

        Returns:
            int: 最大显示文件数量
        """
        if self.display is None:
            return 50  # 默认值
        return self.display.max_display_files

    @property
    def show_file_details(self) -> bool:
        """获取显示文件详情状态

        Returns:
            bool: 如果启用显示文件详情则为True，否则为False
        """
        if self.display is None:
            return False  # 默认值
        return self.display.show_file_details

    @property
    def interactive(self) -> bool:
        """获取交互模式状态

        Returns:
            bool: 如果启用交互模式则为True，否则为False
        """
        if self.interaction is None:
            return False  # 默认值
        return self.interaction.interactive

    @property
    def confirm_prompts(self) -> bool:
        """获取确认提示状态

        Returns:
            bool: 如果启用确认提示则为True，否则为False
        """
        if self.interaction is None:
            return True  # 默认值
        return self.interaction.confirm_prompts

    @property
    def enable_keyboard_shortcuts(self) -> bool:
        """获取键盘快捷键状态

        Returns:
            bool: 如果启用键盘快捷键则为True，否则为False
        """
        if self.interaction is None:
            return True  # 默认值
        return self.interaction.enable_keyboard_shortcuts

    @property
    def auto_complete(self) -> bool:
        """获取自动完成状态

        Returns:
            bool: 如果启用自动完成则为True，否则为False
        """
        if self.interaction is None:
            return True  # 默认值
        return self.interaction.auto_complete

    @property
    def log_level(self) -> str:
        """获取日志级别

        Returns:
            str: 日志级别（DEBUG, INFO, WARNING, ERROR等）
        """
        if self.logging is None:
            return "INFO"  # 默认值
        return self.logging.log_level

    @property
    def history_file(self) -> str:
        """获取历史记录文件路径

        Returns:
            str: 历史记录文件的路径
        """
        if self.logging is None:
            return "./history.json"  # 默认值
        return self.logging.history_file

    @property
    def max_history_entries(self) -> int:
        """获取最大历史记录条目数

        Returns:
            int: 最大历史记录条目数量
        """
        if self.logging is None:
            return 1000  # 默认值
        return self.logging.max_history_entries
