"""
训练模式配置模块

该模块定义了 Mimic3 语音模型训练所需的配置参数和管理功能。
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
import json
import yaml


@dataclass
class TrainConfig:
    """
    Mimic3 训练配置数据类
    
    包含训练过程中所需的所有配置参数，支持从文件加载和导出。
    """
    
    # 输入数据配置
    input_dir: Optional[Path] = None  # 训练数据输入目录
    audio_files: list[Path] = field(default_factory=list)  # 指定的音频文件列表
    subtitle_files: list[Path] = field(default_factory=list)  # 指定的字幕文件列表
    
    # 输出配置
    output_path: Path = field(default_factory=lambda: Path("./models/custom_voice"))  # 模型输出目录
    
    # 训练超参数
    epochs: int = 100  # 训练轮数
    batch_size: int = 32  # 批处理大小
    learning_rate: float = 0.001  # 学习率
    
    # 音频处理参数
    sample_rate: int = 22050  # 音频采样率 (Hz)
    
    # 说话人配置
    speaker_name: str = "custom_speaker"  # 说话人名称标识
    
    # 训练控制参数
    checkpoint_interval: int = 10  # 检查点保存间隔（每 N 个 epoch）
    early_stopping_patience: int = 10  # 早停耐心值
    
    # 日志和调试
    verbose: bool = False  # 是否输出详细日志
    log_file: Optional[Path] = None  # 日志文件路径
    
    def __post_init__(self):
        """初始化后处理，确保路径类型正确"""
        if self.input_dir is not None and not isinstance(self.input_dir, Path):
            self.input_dir = Path(self.input_dir)
        if not isinstance(self.output_path, Path):
            self.output_path = Path(self.output_path)
        if self.log_file is not None and not isinstance(self.log_file, Path):
            self.log_file = Path(self.log_file)
        
        # 转换列表中的路径
        self.audio_files = [Path(f) if not isinstance(f, Path) else f for f in self.audio_files]
        self.subtitle_files = [Path(f) if not isinstance(f, Path) else f for f in self.subtitle_files]
    
    @classmethod
    def from_dict(cls, data: dict) -> "TrainConfig":
        """
        从字典创建配置对象
        
        Args:
            data: 配置字典
            
        Returns:
            TrainConfig 实例
        """
        # 过滤掉不在数据类字段中的键
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)
    
    @classmethod
    def from_json(cls, json_path: Path) -> "TrainConfig":
        """
        从 JSON 文件加载配置
        
        Args:
            json_path: JSON 配置文件路径
            
        Returns:
            TrainConfig 实例
            
        Raises:
            FileNotFoundError: 配置文件不存在
            json.JSONDecodeError: JSON 格式错误
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "TrainConfig":
        """
        从 YAML 文件加载配置
        
        Args:
            yaml_path: YAML 配置文件路径
            
        Returns:
            TrainConfig 实例
            
        Raises:
            FileNotFoundError: 配置文件不存在
            yaml.YAMLError: YAML 格式错误
        """
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
    
    @classmethod
    def from_file(cls, config_path: Path) -> "TrainConfig":
        """
        根据文件扩展名自动选择加载方式
        
        Args:
            config_path: 配置文件路径（支持 .json, .yaml, .yml）
            
        Returns:
            TrainConfig 实例
            
        Raises:
            ValueError: 不支持的文件格式
        """
        config_path = Path(config_path)
        suffix = config_path.suffix.lower()
        
        if suffix == '.json':
            return cls.from_json(config_path)
        elif suffix in ('.yaml', '.yml'):
            return cls.from_yaml(config_path)
        else:
            raise ValueError(f"不支持的配置文件格式: {suffix}，支持 .json, .yaml, .yml")
    
    def to_dict(self) -> dict:
        """
        将配置导出为字典
        
        Returns:
            配置字典，路径转换为字符串
        """
        data = asdict(self)
        # 将 Path 对象转换为字符串
        for key, value in data.items():
            if isinstance(value, Path):
                data[key] = str(value)
            elif isinstance(value, list) and value and isinstance(value[0], Path):
                data[key] = [str(p) for p in value]
        return data
    
    def to_json(self, json_path: Path, indent: int = 2) -> None:
        """
        将配置保存为 JSON 文件
        
        Args:
            json_path: 输出文件路径
            indent: JSON 缩进空格数
        """
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=indent)
    
    def to_yaml(self, yaml_path: Path) -> None:
        """
        将配置保存为 YAML 文件
        
        Args:
            yaml_path: 输出文件路径
        """
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)
    
    def validate(self) -> list[str]:
        """
        验证配置参数的有效性
        
        Returns:
            错误信息列表，如果为空则表示配置有效
        """
        errors = []
        
        # 验证训练参数
        if self.epochs <= 0:
            errors.append(f"训练轮数 (epochs) 必须为正整数，当前值: {self.epochs}")
        if self.batch_size <= 0:
            errors.append(f"批处理大小 (batch_size) 必须为正整数，当前值: {self.batch_size}")
        if self.learning_rate <= 0:
            errors.append(f"学习率 (learning_rate) 必须为正数，当前值: {self.learning_rate}")
        if self.sample_rate <= 0:
            errors.append(f"采样率 (sample_rate) 必须为正整数，当前值: {self.sample_rate}")
        
        # 验证输入数据
        if self.input_dir is None and not self.audio_files:
            errors.append("必须指定输入目录 (input_dir) 或音频文件列表 (audio_files)")
        
        if self.input_dir is not None and not self.input_dir.exists():
            errors.append(f"输入目录不存在: {self.input_dir}")
        
        # 验证音频文件
        for audio_file in self.audio_files:
            if not audio_file.exists():
                errors.append(f"音频文件不存在: {audio_file}")
        
        # 验证字幕文件
        for subtitle_file in self.subtitle_files:
            if not subtitle_file.exists():
                errors.append(f"字幕文件不存在: {subtitle_file}")
        
        # 验证说话人名称
        if not self.speaker_name or not self.speaker_name.strip():
            errors.append("说话人名称 (speaker_name) 不能为空")
        
        return errors
    
    def ensure_output_dir(self) -> None:
        """确保输出目录存在，如不存在则创建"""
        self.output_path.mkdir(parents=True, exist_ok=True)
    
    def get_summary(self) -> str:
        """
        获取配置摘要信息
        
        Returns:
            配置摘要字符串
        """
        lines = [
            "=" * 50,
            "训练配置摘要",
            "=" * 50,
            f"说话人名称: {self.speaker_name}",
            f"输出目录: {self.output_path}",
            "",
            "训练参数:",
            f"  - 训练轮数: {self.epochs}",
            f"  - 批处理大小: {self.batch_size}",
            f"  - 学习率: {self.learning_rate}",
            f"  - 采样率: {self.sample_rate} Hz",
            "",
            "输入数据:",
        ]
        
        if self.input_dir:
            lines.append(f"  - 输入目录: {self.input_dir}")
        if self.audio_files:
            lines.append(f"  - 音频文件数: {len(self.audio_files)}")
        if self.subtitle_files:
            lines.append(f"  - 字幕文件数: {len(self.subtitle_files)}")
        
        lines.extend([
            "",
            "控制参数:",
            f"  - 检查点间隔: 每 {self.checkpoint_interval} 轮",
            f"  - 早停耐心值: {self.early_stopping_patience}",
            f"  - 详细日志: {'是' if self.verbose else '否'}",
            "=" * 50,
        ])
        
        return "\n".join(lines)


# 默认配置实例
DEFAULT_TRAIN_CONFIG = TrainConfig()


def create_sample_config(output_path: Path) -> None:
    """
    创建示例配置文件
    
    Args:
        output_path: 输出文件路径
    """
    sample_config = TrainConfig(
        input_dir=Path("./training_data"),
        output_path=Path("./models/my_voice"),
        epochs=100,
        batch_size=32,
        sample_rate=22050,
        speaker_name="my_custom_voice",
        verbose=True
    )
    
    suffix = output_path.suffix.lower()
    if suffix == '.json':
        sample_config.to_json(output_path)
    elif suffix in ('.yaml', '.yml'):
        sample_config.to_yaml(output_path)
    else:
        sample_config.to_json(output_path.with_suffix('.json'))
    
    print(f"示例配置文件已创建: {output_path}")
