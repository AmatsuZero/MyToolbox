"""
训练模式配置模块

该模块定义了语音模型训练所需的配置参数和管理功能。
支持 Coqui TTS 训练框架。
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Literal
from enum import Enum
import json
import yaml


class ModelType(Enum):
    """支持的模型架构类型"""
    VITS = "vits"                    # VITS - 端到端模型，推荐使用
    FAST_SPEECH2 = "fast_speech2"    # FastSpeech2 - 快速推理
    TACOTRON2 = "tacotron2"          # Tacotron2 - 经典模型
    GLOW_TTS = "glow_tts"            # Glow-TTS - 流式模型


@dataclass
class AudioConfig:
    """
    音频处理配置
    
    包含 Coqui TTS 所需的音频参数配置
    """
    sample_rate: int = 22050           # 音频采样率 (Hz)
    hop_length: int = 256              # STFT hop length
    win_length: int = 1024             # STFT window length
    fft_size: int = 1024               # FFT size
    num_mels: int = 80                 # Mel 频带数
    mel_fmin: float = 0.0              # Mel 最低频率
    mel_fmax: Optional[float] = None   # Mel 最高频率（None 表示 sample_rate/2）
    preemphasis: float = 0.0           # 预加重系数
    ref_level_db: float = 20.0         # 参考电平 dB
    min_level_db: float = -100.0       # 最小电平 dB
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return asdict(self)


@dataclass
class TrainConfig:
    """
    Coqui TTS 训练配置数据类
    
    包含训练过程中所需的所有配置参数，支持从文件加载和导出。
    """
    
    # 输入数据配置
    input_dir: Optional[Path] = None  # 训练数据输入目录
    audio_files: list[Path] = field(default_factory=list)  # 指定的音频文件列表
    subtitle_files: list[Path] = field(default_factory=list)  # 指定的字幕文件列表
    
    # 输出配置
    output_path: Path = field(default_factory=lambda: Path("./models/custom_voice"))  # 模型输出目录
    
    # 模型配置 (Coqui TTS 特有)
    model_type: str = "vits"  # 模型架构类型: vits, fast_speech2, tacotron2, glow_tts
    
    # 训练超参数
    epochs: int = 100  # 训练轮数
    batch_size: int = 32  # 批处理大小
    learning_rate: float = 0.0002  # 学习率 (Coqui TTS 默认值)
    
    # 音频处理参数
    sample_rate: int = 22050  # 音频采样率 (Hz)
    
    # Coqui TTS 音频配置
    audio_config: AudioConfig = field(default_factory=AudioConfig)
    
    # 说话人配置
    speaker_name: str = "custom_speaker"  # 说话人名称标识
    language: str = "auto"  # 语言代码（auto 表示自动检测）
    
    # 训练控制参数
    checkpoint_interval: int = 1000  # 检查点保存间隔（每 N 步）
    eval_interval: int = 500  # 评估间隔（每 N 步）
    print_interval: int = 50  # 打印间隔（每 N 步）
    early_stopping_patience: int = 10  # 早停耐心值（轮数）
    
    # Coqui TTS 特有参数
    use_phonemes: bool = False  # 是否使用音素
    phoneme_language: str = "auto"  # 音素语言（auto 表示自动检测）
    text_cleaner: str = "basic_cleaners"  # 文本清理器
    max_text_length: int = 500  # 最大文本长度
    max_audio_length: int = 10  # 最大音频长度（秒）
    min_audio_length: float = 0.5  # 最小音频长度（秒）
    
    # 数据增强
    use_data_augmentation: bool = False  # 是否使用数据增强
    augmentation_noise_level: float = 0.001  # 噪声增强级别
    
    # GPU/设备配置
    device: str = "auto"  # 训练设备: auto, cuda, mps, cpu
    mixed_precision: bool = True  # 是否使用混合精度训练
    num_workers: int = 4  # 数据加载器工作进程数
    
    # 恢复训练
    resume_checkpoint: Optional[Path] = None  # 恢复训练的检查点路径
    
    # 日志和调试
    verbose: bool = False  # 是否输出详细日志
    log_file: Optional[Path] = None  # 日志文件路径
    tensorboard: bool = True  # 是否启用 TensorBoard 日志
    
    def __post_init__(self):
        """初始化后处理，确保路径类型正确"""
        if self.input_dir is not None and not isinstance(self.input_dir, Path):
            self.input_dir = Path(self.input_dir)
        if not isinstance(self.output_path, Path):
            self.output_path = Path(self.output_path)
        if self.log_file is not None and not isinstance(self.log_file, Path):
            self.log_file = Path(self.log_file)
        if self.resume_checkpoint is not None and not isinstance(self.resume_checkpoint, Path):
            self.resume_checkpoint = Path(self.resume_checkpoint)
        
        # 转换列表中的路径
        self.audio_files = [Path(f) if not isinstance(f, Path) else f for f in self.audio_files]
        self.subtitle_files = [Path(f) if not isinstance(f, Path) else f for f in self.subtitle_files]
        
        # 确保 audio_config 是 AudioConfig 实例
        if isinstance(self.audio_config, dict):
            self.audio_config = AudioConfig(**self.audio_config)
        
        # 同步 sample_rate
        if self.audio_config.sample_rate != self.sample_rate:
            self.audio_config.sample_rate = self.sample_rate
    
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
        
        # 处理 audio_config
        if 'audio_config' in filtered_data and isinstance(filtered_data['audio_config'], dict):
            filtered_data['audio_config'] = AudioConfig(**filtered_data['audio_config'])
        
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
        
        # 验证模型类型
        valid_model_types = ["vits", "fast_speech2", "tacotron2", "glow_tts"]
        if self.model_type not in valid_model_types:
            errors.append(f"不支持的模型类型: {self.model_type}，支持: {', '.join(valid_model_types)}")
        
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
        
        # 验证恢复检查点
        if self.resume_checkpoint is not None and not self.resume_checkpoint.exists():
            errors.append(f"恢复检查点不存在: {self.resume_checkpoint}")
        
        # 验证设备
        valid_devices = ["auto", "cuda", "mps", "cpu"]
        if self.device not in valid_devices:
            errors.append(f"不支持的设备类型: {self.device}，支持: {', '.join(valid_devices)}")
        
        return errors
    
    def ensure_output_dir(self) -> None:
        """确保输出目录存在，如不存在则创建"""
        self.output_path.mkdir(parents=True, exist_ok=True)
    
    def get_effective_device(self) -> str:
        """
        获取实际使用的设备
        
        Returns:
            str: "cuda", "mps" 或 "cpu"
        """
        if self.device != "auto":
            return self.device
        
        try:
            import torch
            
            # 优先检查 CUDA
            if torch.cuda.is_available():
                return "cuda"
            
            # 检查 MPS (Apple Silicon)
            if hasattr(torch.backends, 'mps'):
                try:
                    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                        return "mps"
                except Exception:
                    # MPS 检测失败，回退到 CPU
                    pass
            
            # 默认使用 CPU
            return "cpu"
            
        except ImportError:
            return "cpu"
    
    def get_model_type_enum(self) -> ModelType:
        """
        获取模型类型枚举值
        
        Returns:
            ModelType: 模型类型枚举
        """
        type_map = {
            "vits": ModelType.VITS,
            "fast_speech2": ModelType.FAST_SPEECH2,
            "tacotron2": ModelType.TACOTRON2,
            "glow_tts": ModelType.GLOW_TTS,
        }
        return type_map.get(self.model_type, ModelType.VITS)
    
    def get_summary(self) -> str:
        """
        获取配置摘要信息
        
        Returns:
            配置摘要字符串
        """
        device = self.get_effective_device()
        
        lines = [
            "=" * 60,
            "Coqui TTS 训练配置摘要",
            "=" * 60,
            f"说话人名称: {self.speaker_name}",
            f"语言: {self.language}",
            f"模型类型: {self.model_type.upper()}",
            f"输出目录: {self.output_path}",
            "",
            "训练参数:",
            f"  - 训练轮数: {self.epochs}",
            f"  - 批处理大小: {self.batch_size}",
            f"  - 学习率: {self.learning_rate}",
            f"  - 采样率: {self.sample_rate} Hz",
            f"  - 训练设备: {device.upper()}",
            f"  - 混合精度: {'是' if self.mixed_precision else '否'}",
            "",
            "音频配置:",
            f"  - Mel 频带数: {self.audio_config.num_mels}",
            f"  - FFT 大小: {self.audio_config.fft_size}",
            f"  - Hop Length: {self.audio_config.hop_length}",
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
            f"  - 检查点间隔: 每 {self.checkpoint_interval} 步",
            f"  - 评估间隔: 每 {self.eval_interval} 步",
            f"  - 早停耐心值: {self.early_stopping_patience} 轮",
            f"  - 详细日志: {'是' if self.verbose else '否'}",
            f"  - TensorBoard: {'是' if self.tensorboard else '否'}",
        ])
        
        if self.resume_checkpoint:
            lines.append(f"  - 恢复检查点: {self.resume_checkpoint}")
        
        lines.append("=" * 60)
        
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
        model_type="vits",
        epochs=100,
        batch_size=16,
        sample_rate=22050,
        speaker_name="my_custom_voice",
        language="zh-cn",
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


def get_recommended_config_for_model(model_type: str) -> dict:
    """
    获取特定模型类型的推荐配置
    
    Args:
        model_type: 模型类型
        
    Returns:
        dict: 推荐配置字典
    """
    configs = {
        "vits": {
            "batch_size": 16,
            "learning_rate": 0.0002,
            "checkpoint_interval": 1000,
            "eval_interval": 500,
        },
        "fast_speech2": {
            "batch_size": 32,
            "learning_rate": 0.001,
            "checkpoint_interval": 2000,
            "eval_interval": 1000,
        },
        "tacotron2": {
            "batch_size": 32,
            "learning_rate": 0.001,
            "checkpoint_interval": 2000,
            "eval_interval": 1000,
        },
        "glow_tts": {
            "batch_size": 16,
            "learning_rate": 0.0001,
            "checkpoint_interval": 1000,
            "eval_interval": 500,
        },
    }
    return configs.get(model_type, configs["vits"])
