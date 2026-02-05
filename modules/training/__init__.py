"""
训练模块

该模块提供了 Coqui TTS 语音模型训练的完整功能：
- 训练配置管理
- 训练数据预处理
- Coqui TTS 模型训练
- ONNX 模型导出
- 训练流程控制

支持的模型架构：
- VITS (推荐)
- FastSpeech2
- Tacotron2
- Glow-TTS
"""

from .train_config import TrainConfig, AudioConfig, ModelType
from .train_data_manager import TrainDataManager, TrainDataPair, ScanResult, MediaFile, DatasetStats
from .coqui_trainer import CoquiTrainer, TrainingProgress, TrainingResult, TrainingStatus
from .coqui_checker import CoquiChecker, CoquiCheckResult, check_coqui, ensure_coqui_available
from .coqui_exporter import CoquiExporter, ExportResult, export_model
from .train_controller import TrainController, TrainControllerProgress, TrainControllerResult, TrainPhase

__all__ = [
    # 配置
    'TrainConfig',
    'AudioConfig',
    'ModelType',
    
    # 数据管理
    'TrainDataManager',
    'TrainDataPair',
    'ScanResult',
    'MediaFile',
    'DatasetStats',
    
    # 训练器
    'CoquiTrainer',
    'TrainingProgress',
    'TrainingResult',
    'TrainingStatus',
    
    # 依赖检查
    'CoquiChecker',
    'CoquiCheckResult',
    'check_coqui',
    'ensure_coqui_available',
    
    # ONNX 导出
    'CoquiExporter',
    'ExportResult',
    'export_model',
    
    # 控制器
    'TrainController',
    'TrainControllerProgress',
    'TrainControllerResult',
    'TrainPhase',
]
