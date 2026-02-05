#!/usr/bin/env python3
"""
Coqui TTS 训练接口封装模块

封装 Coqui TTS 训练 API，提供训练参数配置、模型输出管理和训练进度回调功能。
"""

import os
import json
import shutil
import time
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable, Optional, List, Dict, Any
from enum import Enum

from modules.training.train_config import TrainConfig, AudioConfig


logger = logging.getLogger(__name__)


class TrainingStatus(Enum):
    """训练状态枚举"""
    IDLE = "idle"               # 空闲状态
    PREPARING = "preparing"     # 准备中
    TRAINING = "training"       # 训练中
    PAUSED = "paused"          # 已暂停
    COMPLETED = "completed"     # 已完成
    FAILED = "failed"          # 失败
    CANCELLED = "cancelled"     # 已取消


@dataclass
class TrainingProgress:
    """训练进度信息"""
    current_epoch: int = 0          # 当前 epoch
    total_epochs: int = 0           # 总 epoch 数
    current_step: int = 0           # 当前步数
    total_steps: int = 0            # 总步数
    loss: float = 0.0               # 当前损失值
    learning_rate: float = 0.0      # 当前学习率
    elapsed_time: float = 0.0       # 已用时间（秒）
    estimated_remaining: float = 0.0 # 预估剩余时间（秒）
    status: TrainingStatus = TrainingStatus.IDLE
    message: str = ""               # 状态消息
    
    # Coqui TTS 特有的进度信息
    avg_loss: float = 0.0           # 平均损失值
    mel_loss: float = 0.0           # Mel 损失
    duration_loss: float = 0.0      # 时长损失
    kl_loss: float = 0.0            # KL 散度损失（VITS）
    
    @property
    def progress_percentage(self) -> float:
        """计算训练进度百分比"""
        if self.total_epochs == 0:
            return 0.0
        return (self.current_epoch / self.total_epochs) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "current_epoch": self.current_epoch,
            "total_epochs": self.total_epochs,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "loss": self.loss,
            "avg_loss": self.avg_loss,
            "mel_loss": self.mel_loss,
            "duration_loss": self.duration_loss,
            "kl_loss": self.kl_loss,
            "learning_rate": self.learning_rate,
            "elapsed_time": self.elapsed_time,
            "estimated_remaining": self.estimated_remaining,
            "progress_percentage": self.progress_percentage,
            "status": self.status.value,
            "message": self.message
        }


@dataclass
class TrainingResult:
    """训练结果信息"""
    success: bool = False           # 是否成功
    model_path: Optional[Path] = None  # 模型文件路径
    total_epochs: int = 0           # 训练的总 epoch 数
    final_loss: float = 0.0         # 最终损失值
    training_time: float = 0.0      # 总训练时间（秒）
    model_size: int = 0             # 模型文件大小（字节）
    error_message: str = ""         # 错误信息（如果失败）
    checkpoint_path: Optional[Path] = None  # 检查点路径（用于断点续训）
    config_path: Optional[Path] = None  # 模型配置文件路径
    logs: List[str] = field(default_factory=list)  # 训练日志
    
    # ONNX 导出相关
    onnx_model_path: Optional[Path] = None  # ONNX 模型路径
    onnx_config_path: Optional[Path] = None  # ONNX 配置路径
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "success": self.success,
            "model_path": str(self.model_path) if self.model_path else None,
            "total_epochs": self.total_epochs,
            "final_loss": self.final_loss,
            "training_time": self.training_time,
            "model_size": self.model_size,
            "error_message": self.error_message,
            "checkpoint_path": str(self.checkpoint_path) if self.checkpoint_path else None,
            "config_path": str(self.config_path) if self.config_path else None,
            "onnx_model_path": str(self.onnx_model_path) if self.onnx_model_path else None,
            "onnx_config_path": str(self.onnx_config_path) if self.onnx_config_path else None
        }


# 进度回调函数类型
ProgressCallback = Callable[[TrainingProgress], None]


class CoquiTrainer:
    """
    Coqui TTS 训练器
    
    封装 Coqui TTS 模型训练功能，支持：
    - 多种模型架构（VITS、FastSpeech2、Tacotron2、Glow-TTS）
    - 训练参数配置
    - 训练进度监控
    - 断点续训
    - 模型输出管理
    - ONNX 导出
    """
    
    def __init__(self, config: TrainConfig, verbose: bool = False):
        """
        初始化训练器
        
        Args:
            config: 训练配置
            verbose: 是否输出详细日志
        """
        self.config = config
        self.verbose = verbose
        self._progress = TrainingProgress()
        self._callbacks: List[ProgressCallback] = []
        self._start_time: float = 0
        self._should_stop: bool = False
        self._logs: List[str] = []
        self._trainer = None  # Coqui TTS Trainer 实例
        
    def add_progress_callback(self, callback: ProgressCallback) -> None:
        """
        添加进度回调函数
        
        Args:
            callback: 回调函数，接收 TrainingProgress 参数
        """
        self._callbacks.append(callback)
    
    def remove_progress_callback(self, callback: ProgressCallback) -> None:
        """
        移除进度回调函数
        
        Args:
            callback: 要移除的回调函数
        """
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    def _notify_progress(self) -> None:
        """通知所有回调函数当前进度"""
        for callback in self._callbacks:
            try:
                callback(self._progress)
            except Exception as e:
                self._log(f"进度回调执行失败: {e}")
    
    def _log(self, message: str) -> None:
        """
        记录日志
        
        Args:
            message: 日志消息
        """
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self._logs.append(log_entry)
        if self.verbose:
            print(log_entry)
        logger.info(message)
    
    def _update_progress(self, **kwargs) -> None:
        """
        更新训练进度
        
        Args:
            **kwargs: 要更新的进度属性
        """
        for key, value in kwargs.items():
            if hasattr(self._progress, key):
                setattr(self._progress, key, value)
        
        # 计算已用时间
        if self._start_time > 0:
            self._progress.elapsed_time = time.time() - self._start_time
            
            # 估算剩余时间
            if self._progress.current_step > 0 and self._progress.total_steps > 0:
                avg_time_per_step = self._progress.elapsed_time / self._progress.current_step
                remaining_steps = self._progress.total_steps - self._progress.current_step
                self._progress.estimated_remaining = avg_time_per_step * remaining_steps
        
        self._notify_progress()
    
    def _apply_bfloat16_fix(self) -> None:
        """
        应用 BFloat16 兼容性修复
        
        在 Apple Silicon 上使用混合精度训练时，PyTorch 会使用 BFloat16 格式，
        但 numpy 不支持 BFloat16，导致在生成训练日志图表时出错。
        这个方法通过 monkey patch 修复 TTS 库中的相关代码。
        
        仅在以下条件下应用修复：
        1. 操作系统为 macOS (Darwin)
        2. CPU 架构为 ARM64 (Apple Silicon)
        3. PyTorch 支持 BFloat16
        """
        try:
            import platform
            import torch
            
            # 检测平台：只在 macOS Apple Silicon 上应用修复
            system = platform.system()
            machine = platform.machine()
            
            # 检查是否为 macOS
            if system != 'Darwin':
                self._log(f"ℹ️  当前平台 ({system}) 不需要 BFloat16 修复，跳过")
                return
            
            # 检查是否为 Apple Silicon (ARM64)
            if machine.lower() not in ['arm64', 'aarch64']:
                self._log(f"ℹ️  当前架构 ({machine}) 不需要 BFloat16 修复，跳过")
                return
            
            # 检查 PyTorch 是否支持 BFloat16
            # 在 Apple Silicon 上，BFloat16 支持通过 MPS 后端
            has_bfloat16_support = False
            try:
                # 检查 MPS 是否可用（Apple Silicon 特有）
                if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    has_bfloat16_support = True
                # 或者检查 CPU 是否支持 BFloat16
                elif hasattr(torch, 'bfloat16'):
                    # 尝试创建一个 BFloat16 张量来验证支持
                    test_tensor = torch.tensor([1.0], dtype=torch.bfloat16)
                    has_bfloat16_support = True
            except Exception:
                pass
            
            if not has_bfloat16_support:
                self._log("ℹ️  当前 PyTorch 不支持 BFloat16，无需应用修复")
                return
            
            # 所有条件满足，应用修复
            self._log(f"🔧 检测到 Apple Silicon ({machine})，准备应用 BFloat16 修复...")
            
            from TTS.vocoder.utils import generic_utils
            
            # 保存原始的 plot_results 函数
            original_plot_results = generic_utils.plot_results
            
            def patched_plot_results(y_hat, y, ap, name_prefix):
                """修复后的 plot_results 函数，支持 BFloat16"""
                # 将 BFloat16 张量转换为 float32
                if isinstance(y_hat, (list, tuple)) and len(y_hat) > 0:
                    if hasattr(y_hat[0], 'dtype') and y_hat[0].dtype == torch.bfloat16:
                        y_hat = [item.float() if hasattr(item, 'float') else item for item in y_hat]
                elif hasattr(y_hat, 'dtype') and y_hat.dtype == torch.bfloat16:
                    y_hat = y_hat.float()
                
                if isinstance(y, (list, tuple)) and len(y) > 0:
                    if hasattr(y[0], 'dtype') and y[0].dtype == torch.bfloat16:
                        y = [item.float() if hasattr(item, 'float') else item for item in y]
                elif hasattr(y, 'dtype') and y.dtype == torch.bfloat16:
                    y = y.float()
                
                # 调用原始函数
                return original_plot_results(y_hat, y, ap, name_prefix)
            
            # 应用 monkey patch 到 generic_utils 模块
            generic_utils.plot_results = patched_plot_results
            
            # 同时 patch 已经导入 plot_results 的模块
            # 因为 vits.py 在导入时已经执行了 from TTS.vocoder.utils.generic_utils import plot_results
            # 所以需要直接修改 vits 模块中的引用
            try:
                from TTS.tts.models import vits
                if hasattr(vits, 'plot_results'):
                    vits.plot_results = patched_plot_results
                
                # 更重要的是：patch VITS 模型的 _log 方法
                # 因为 _log 方法中有多处直接调用 .numpy()，都需要处理 BFloat16
                if hasattr(vits, 'Vits'):
                    original_vits_log = vits.Vits._log
                    
                    def patched_vits_log(self, ap, batch, outputs, name_prefix="train"):
                        """修复后的 VITS._log 方法，自动处理 BFloat16"""
                        # 将所有 BFloat16 张量转换为 float32
                        def convert_bfloat16(data):
                            """递归转换 BFloat16 张量"""
                            if isinstance(data, dict):
                                return {k: convert_bfloat16(v) for k, v in data.items()}
                            elif isinstance(data, (list, tuple)):
                                return type(data)(convert_bfloat16(item) for item in data)
                            elif hasattr(data, 'dtype') and data.dtype == torch.bfloat16:
                                return data.float()
                            return data
                        
                        # 转换 outputs 中的所有 BFloat16 张量
                        outputs = convert_bfloat16(outputs)
                        
                        # 调用原始方法
                        return original_vits_log(self, ap, batch, outputs, name_prefix)
                    
                    vits.Vits._log = patched_vits_log
                    self._log("✅ 已应用 BFloat16 兼容性修复（包括 VITS._log 方法）")
                else:
                    self._log("✅ 已应用 BFloat16 兼容性修复")
            except (ImportError, AttributeError) as e:
                self._log(f"✅ 已应用 BFloat16 兼容性修复（部分功能: {e}）")
            
        except ImportError:
            # 如果 TTS 库未安装，忽略
            pass
        except Exception as e:
            self._log(f"⚠️  应用 BFloat16 修复时出错（可忽略）: {e}")
    
    def _apply_mps_fix(self) -> None:
        """
        应用 MPS 设备支持修复
        
        Trainer 的 setup_torch_training_env 函数不支持 MPS 设备，
        会在 macOS 上尝试调用 torch.cuda.set_device() 导致错误。
        这个方法通过 monkey patch 修复 Trainer 的设备检测逻辑。
        
        仅在以下条件下应用修复：
        1. 操作系统为 macOS (Darwin)
        2. PyTorch 支持 MPS
        3. MPS 可用
        """
        try:
            import platform
            import torch
            
            # 检测平台：只在 macOS 上应用修复
            system = platform.system()
            if system != 'Darwin':
                return
            
            # 检查 MPS 是否可用
            if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
                return
            
            self._log("🔧 检测到 MPS 设备，准备应用 MPS 支持修复...")
            
            # Patch setup_torch_training_env 函数
            from trainer import trainer_utils
            import random
            import numpy as np
            import os
            
            # 保存原始函数
            original_setup = trainer_utils.setup_torch_training_env
            
            def patched_setup_torch_training_env(
                args,
                cudnn_enable,
                cudnn_benchmark,
                cudnn_deterministic,
                use_ddp=False,
                training_seed=54321,
                allow_tf32=False,
                gpu=None,
            ):
                """修复后的 setup_torch_training_env，支持 MPS 设备"""
                
                # 检查是否为 MPS 环境
                is_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
                is_cuda = torch.cuda.is_available()
                
                if is_mps and not is_cuda:
                    # MPS 环境：不调用 CUDA 相关函数
                    
                    # 设置随机种子
                    random.seed(training_seed)
                    os.environ["PYTHONHASHSEED"] = str(training_seed)
                    np.random.seed(training_seed)
                    torch.manual_seed(training_seed)
                    
                    # MPS 不支持 CUDNN，跳过 CUDNN 设置
                    
                    # 返回 use_cuda=False（让 PyTorch 自动使用 MPS）
                    # num_gpus=1 表示有一个加速设备（MPS）
                    return False, 1
                else:
                    # CUDA 环境或 CPU：使用原始函数
                    return original_setup(
                        args,
                        cudnn_enable,
                        cudnn_benchmark,
                        cudnn_deterministic,
                        use_ddp,
                        training_seed,
                        allow_tf32,
                        gpu,
                    )
            
            # 应用 monkey patch 到 trainer_utils 模块
            trainer_utils.setup_torch_training_env = patched_setup_torch_training_env
            
            # 关键：还需要 patch trainer.trainer 模块中的引用
            # 因为 Trainer 类使用了 from trainer_utils import setup_torch_training_env
            # 这会创建一个本地引用，我们需要在 Trainer 导入之前就替换它
            import sys
            if 'trainer.trainer' in sys.modules:
                # 如果 trainer.trainer 已经导入，需要替换其中的引用
                import trainer.trainer as trainer_module
                trainer_module.setup_torch_training_env = patched_setup_torch_training_env
            
            self._log("✅ 已应用 MPS 设备支持修复")
            
        except ImportError:
            # 如果 trainer 库未安装，忽略
            pass
        except Exception as e:
            self._log(f"⚠️  应用 MPS 修复时出错（可忽略）: {e}")
    
    def _prepare_training_directory(self) -> Path:
        """
        准备训练目录结构
        
        Returns:
            训练工作目录路径
        """
        self._log("准备训练目录...")
        
        # 创建输出目录
        output_dir = self.config.output_path
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建训练工作目录
        work_dir = output_dir / "training_workspace"
        work_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建检查点目录
        checkpoint_dir = work_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建日志目录
        log_dir = work_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建模型目录
        model_dir = output_dir / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        self._log(f"训练工作目录: {work_dir}")
        return work_dir
    
    def _create_coqui_config(self, dataset_path: Path) -> Any:
        """
        创建 Coqui TTS 训练配置
        
        Args:
            dataset_path: 数据集路径
            
        Returns:
            Coqui TTS 配置对象
        """
        self._log(f"创建 {self.config.model_type.upper()} 模型配置...")
        
        try:
            from TTS.tts.configs.vits_config import VitsConfig
            from TTS.tts.configs.glow_tts_config import GlowTTSConfig
            from TTS.tts.configs.fastspeech2_config import Fastspeech2Config
            from TTS.tts.configs.tacotron2_config import Tacotron2Config
            from TTS.config import BaseAudioConfig, BaseDatasetConfig
            from TTS.tts.models.vits import Vits
        except ImportError as e:
            self._log(f"导入 Coqui TTS 配置失败: {e}")
            raise ImportError("Coqui TTS 未安装，请运行: pip install TTS>=0.22.0")
        
        # 创建音频配置
        audio_config = BaseAudioConfig(
            sample_rate=self.config.sample_rate,
            hop_length=self.config.audio_config.hop_length,
            win_length=self.config.audio_config.win_length,
            fft_size=self.config.audio_config.fft_size,
            num_mels=self.config.audio_config.num_mels,
            mel_fmin=self.config.audio_config.mel_fmin,
            mel_fmax=self.config.audio_config.mel_fmax,
        )
        
        # 创建数据集配置
        dataset_config = BaseDatasetConfig(
            formatter="ljspeech",
            meta_file_train="metadata.csv",
            path=str(dataset_path),
            language=self.config.language,
        )
        
        # 检查数据集大小，动态调整验证集参数
        metadata_file = dataset_path / "metadata.csv"
        num_samples = 0
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                num_samples = sum(1 for _ in f)
        
        # 根据数据集大小调整验证集划分
        # 对于小数据集，使用更大的比例或禁用验证
        if num_samples < 10:
            # 小于 10 个样本，禁用验证集
            eval_split_size = 0.0
            run_eval = False
            self._log(f"数据集过小 ({num_samples} 个样本)，禁用验证集")
        elif num_samples < 50:
            # 小于 50 个样本，使用较大的验证比例
            eval_split_size = 0.2
            run_eval = True
            self._log(f"小数据集 ({num_samples} 个样本)，验证集比例设为 20%")
        else:
            # 正常数据集
            eval_split_size = 0.01
            run_eval = True
        
        # 获取有效设备
        device = self.config.get_effective_device()
        
        # 记录设备信息（设备管理由 Trainer 负责，不在 Config 中设置）
        if device in ["cuda", "mps"]:
            self._log(f"✅ 启用 GPU 加速: {device.upper()}")
        else:
            self._log(f"⚠️  使用 CPU 训练（速度较慢）")
        
        # 通用训练参数
        # 注意：新版本的 Coqui TTS 不再在 Config 中设置 use_cuda
        # 设备管理由 Trainer 负责，会自动检测并使用可用的设备
        common_args = {
            "audio": audio_config,
            "batch_size": self.config.batch_size,
            "eval_batch_size": max(1, self.config.batch_size // 2),
            "num_loader_workers": self.config.num_workers,
            "num_eval_loader_workers": 2,
            "run_eval": run_eval,
            "eval_split_size": eval_split_size,
            "test_delay_epochs": 5,
            "epochs": self.config.epochs,
            "lr": self.config.learning_rate,
            "print_step": self.config.print_interval,
            "save_step": self.config.checkpoint_interval,
            "save_n_checkpoints": 3,
            "save_best_after": 1000,
            "datasets": [dataset_config],
            "text_cleaner": self.config.text_cleaner,
            "use_phonemes": self.config.use_phonemes,
            "phoneme_language": self.config.phoneme_language if self.config.use_phonemes else None,
            "mixed_precision": self.config.mixed_precision,
        }
        
        # 根据模型类型创建配置
        model_type = self.config.model_type.lower()
        
        if model_type == "vits":
            config = VitsConfig(**common_args)
            # 设置模型参数（通过 model_args 对象）
            config.model_args.num_chars = 256  # 将根据实际文本调整
            config.model_args.out_channels = 513
            config.model_args.spec_segment_size = 32
            config.model_args.hidden_channels = 192
            config.model_args.hidden_channels_ffn_text_encoder = 768
            config.model_args.num_heads_text_encoder = 2
            config.model_args.num_layers_text_encoder = 6
            config.model_args.kernel_size_text_encoder = 3
            config.model_args.dropout_p_text_encoder = 0.1
            config.model_args.dropout_p_duration_predictor = 0.5
            # hidden_channels_dp, kernel_size_dp, num_layers_dp 在新版本中已不存在
            config.model_args.resblock_type_decoder = "1"
            config.model_args.resblock_kernel_sizes_decoder = [3, 7, 11]
            config.model_args.resblock_dilation_sizes_decoder = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
            config.model_args.upsample_rates_decoder = [8, 8, 2, 2]
            config.model_args.upsample_initial_channel_decoder = 512
            config.model_args.upsample_kernel_sizes_decoder = [16, 16, 4, 4]
            config.model_args.use_sdp = True
        elif model_type == "glow_tts":
            config = GlowTTSConfig(
                **common_args,
            )
        elif model_type == "fast_speech2":
            config = Fastspeech2Config(
                **common_args,
            )
        elif model_type == "tacotron2":
            config = Tacotron2Config(
                **common_args,
            )
        else:
            self._log(f"未知的模型类型 {model_type}，使用默认 VITS")
            config = VitsConfig(**common_args)
        
        return config
    
    def _create_training_callback(self):
        """
        创建 Coqui TTS 训练回调
        
        Returns:
            训练回调类
        """
        trainer = self
        
        class TrainingCallback:
            """Coqui TTS 训练回调"""
            
            def on_train_step_end(self, trainer_obj, outputs: dict, step: int, epoch: int):
                """训练步骤结束时调用"""
                if trainer._should_stop:
                    raise KeyboardInterrupt("用户请求停止训练")
                
                # 提取损失值
                loss = outputs.get("loss", 0.0)
                mel_loss = outputs.get("mel_loss", 0.0) or outputs.get("loss_mel", 0.0)
                duration_loss = outputs.get("duration_loss", 0.0) or outputs.get("loss_duration", 0.0)
                kl_loss = outputs.get("kl_loss", 0.0) or outputs.get("loss_kl", 0.0)
                
                trainer._update_progress(
                    current_step=step,
                    current_epoch=epoch,
                    loss=float(loss) if loss else 0.0,
                    mel_loss=float(mel_loss) if mel_loss else 0.0,
                    duration_loss=float(duration_loss) if duration_loss else 0.0,
                    kl_loss=float(kl_loss) if kl_loss else 0.0,
                    status=TrainingStatus.TRAINING,
                    message=f"训练中: Epoch {epoch}, Step {step}, Loss: {loss:.4f}"
                )
            
            def on_epoch_start(self, trainer_obj):
                """Epoch 开始时调用"""
                trainer._log(f"开始 Epoch {trainer._progress.current_epoch + 1}")
            
            def on_epoch_end(self, trainer_obj):
                """Epoch 结束时调用"""
                trainer._log(f"Epoch {trainer._progress.current_epoch} 完成, Loss: {trainer._progress.loss:.4f}")
            
            def on_train_start(self, trainer_obj):
                """训练开始时调用"""
                trainer._log("训练开始")
                trainer._update_progress(
                    status=TrainingStatus.TRAINING,
                    message="训练开始"
                )
            
            def on_train_end(self, trainer_obj):
                """训练结束时调用"""
                trainer._log("训练结束")
        
        return TrainingCallback()
    
    def _is_coqui_available(self) -> bool:
        """检查 Coqui TTS 是否可用"""
        try:
            import TTS
            from TTS.api import TTS as TTSApi
            return True
        except ImportError:
            return False
    
    def train(self, dataset_path: Path, resume_from: Optional[Path] = None) -> TrainingResult:
        """
        执行训练
        
        Args:
            dataset_path: 数据集路径（LJSpeech 格式）
            resume_from: 断点续训的检查点路径（可选）
            
        Returns:
            训练结果
        """
        result = TrainingResult()
        self._logs = []
        self._should_stop = False
        
        try:
            # 更新状态为准备中
            self._update_progress(
                status=TrainingStatus.PREPARING,
                message="正在准备训练环境..."
            )
            
            # 准备训练目录
            work_dir = self._prepare_training_directory()
            
            # 检查 Coqui TTS 是否可用
            if self._is_coqui_available():
                result = self._run_real_training(dataset_path, work_dir, resume_from)
            else:
                self._log("警告: Coqui TTS 未安装，执行模拟训练")
                result = self._run_simulated_training(dataset_path, work_dir)
            
        except Exception as e:
            self._log(f"训练过程发生错误: {e}")
            import traceback
            self._log(traceback.format_exc())
            self._update_progress(
                status=TrainingStatus.FAILED,
                message=f"训练失败: {e}"
            )
            result.success = False
            result.error_message = str(e)
        
        result.logs = self._logs.copy()
        return result
    
    def _run_real_training(self, dataset_path: Path, work_dir: Path, 
                          resume_from: Optional[Path] = None) -> TrainingResult:
        """
        执行真实的 Coqui TTS 训练
        
        Args:
            dataset_path: 数据集路径
            work_dir: 工作目录
            resume_from: 恢复训练的检查点路径
            
        Returns:
            训练结果
        """
        result = TrainingResult()
        
        try:
            # 重要：必须在导入 Trainer 之前应用修复！
            # 因为 Trainer 导入时会加载 trainer_utils，我们需要在那之前 patch
            
            # 修复 BFloat16 兼容性问题
            # 在 Apple Silicon 上使用混合精度时，PyTorch 会使用 BFloat16
            # 但 numpy 不支持 BFloat16，需要先转换为 float32
            self._apply_bfloat16_fix()
            
            # 修复 MPS 设备支持
            # Trainer 的 setup_torch_training_env 函数不支持 MPS，需要 patch
            self._apply_mps_fix()
            
            # 注意：从 Coqui TTS 0.22.0 开始，Trainer 移到了独立的 trainer 包中
            # 必须在应用 MPS 修复之后再导入！
            from trainer import Trainer, TrainerArgs
            
            # 创建配置
            config = self._create_coqui_config(dataset_path)
            
            # 保存配置文件
            config_path = work_dir / "config.json"
            config.save_json(str(config_path))
            self._log(f"配置文件已保存: {config_path}")
            
            # 设置输出路径
            output_path = self.config.output_path / "model"
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 获取有效设备并设置 GPU 参数
            device = self.config.get_effective_device()
            
            # 设置 GPU 参数
            # - 如果是 CUDA 或 MPS，设置 gpu=0（使用第一个 GPU）
            # - 如果是 CPU，设置 gpu=None（不使用 GPU）
            gpu_id = 0 if device in ["cuda", "mps"] else None
            
            # 创建训练参数
            trainer_args = TrainerArgs(
                best_path=str(output_path),
                restore_path=str(resume_from) if resume_from else "",
                gpu=gpu_id,  # 设置 GPU 设备
            )
            
            # 记录开始时间
            self._start_time = time.time()
            
            # 更新状态
            self._update_progress(
                status=TrainingStatus.TRAINING,
                message="训练进行中...",
                total_epochs=self.config.epochs
            )
            
            # 加载训练数据
            self._log("加载训练数据集...")
            from TTS.tts.datasets import load_tts_samples
            from TTS.tts.models.vits import Vits
            
            # 根据配置决定是否划分验证集
            if config.run_eval and config.eval_split_size > 0:
                train_samples, eval_samples = load_tts_samples(
                    config.datasets,
                    eval_split=True,
                    eval_split_max_size=config.eval_split_max_size,
                    eval_split_size=config.eval_split_size,
                )
            else:
                # 不划分验证集，所有样本用于训练
                train_samples, _ = load_tts_samples(
                    config.datasets,
                    eval_split=False,
                )
                eval_samples = []
                self._log("已禁用验证集，所有样本用于训练")
            
            self._log(f"训练样本数: {len(train_samples)}, 验证样本数: {len(eval_samples)}")
            
            # 创建 VITS 模型
            self._log("创建 VITS 模型...")
            model = Vits.init_from_config(config, samples=train_samples + eval_samples)
            
            # 创建 Trainer
            self._log("创建 Coqui TTS Trainer...")
            trainer = Trainer(
                trainer_args,
                config,
                output_path=str(output_path),
                model=model,
                train_samples=train_samples,
                eval_samples=eval_samples,
                gpu=gpu_id,  # 设置 GPU 设备
            )
            
            self._trainer = trainer
            
            # 开始训练
            self._log("开始训练...")
            trainer.fit()
            
            # 训练完成
            result.success = True
            result.training_time = time.time() - self._start_time
            result.total_epochs = self._progress.current_epoch
            result.final_loss = self._progress.loss
            
            # 查找最佳模型
            model_files = list(output_path.glob("best_model*.pth")) or list(output_path.glob("*.pth"))
            if model_files:
                result.model_path = model_files[0]
                result.model_size = result.model_path.stat().st_size
                self._log(f"模型已保存: {result.model_path}")
            
            # 保存配置
            result.config_path = config_path
            
            # 查找检查点
            checkpoint_files = sorted(output_path.glob("checkpoint_*.pth"))
            if checkpoint_files:
                result.checkpoint_path = checkpoint_files[-1]
            
            self._update_progress(
                status=TrainingStatus.COMPLETED,
                message="训练完成"
            )
            
        except KeyboardInterrupt:
            self._log("训练被用户中断")
            self._update_progress(
                status=TrainingStatus.CANCELLED,
                message="训练已取消"
            )
            result.success = False
            result.error_message = "训练被用户取消"
            
        except Exception as e:
            self._log(f"训练失败: {e}")
            import traceback
            self._log(traceback.format_exc())
            result.success = False
            result.error_message = str(e)
            self._update_progress(
                status=TrainingStatus.FAILED,
                message=f"训练失败: {e}"
            )
        
        finally:
            self._trainer = None
        
        return result
    
    def _run_simulated_training(self, dataset_path: Path, work_dir: Path) -> TrainingResult:
        """
        执行模拟训练（用于测试或 Coqui TTS 未安装时）
        
        Args:
            dataset_path: 数据集路径
            work_dir: 工作目录
            
        Returns:
            训练结果
        """
        result = TrainingResult()
        
        self._log("=" * 60)
        self._log("⚠️ 警告: Coqui TTS 未安装！")
        self._log("=" * 60)
        self._log("将使用【模拟训练模式】，此模式不会生成有效的模型。")
        self._log("模拟训练仅用于测试训练流程。")
        self._log("")
        self._log("如需训练真正的 TTS 模型，请先安装 Coqui TTS:")
        self._log("  pip install TTS>=0.22.0")
        self._log("或使用:")
        self._log("  uv sync --extra training")
        self._log("=" * 60)
        
        self._log("开始模拟训练...")
        self._start_time = time.time()
        
        total_epochs = min(self.config.epochs, 10)  # 模拟训练最多10轮
        steps_per_epoch = 100
        
        try:
            for epoch in range(1, total_epochs + 1):
                if self._should_stop:
                    self._update_progress(
                        status=TrainingStatus.CANCELLED,
                        message="训练已取消"
                    )
                    result.error_message = "训练被用户取消"
                    return result
                
                for step in range(1, steps_per_epoch + 1):
                    if self._should_stop:
                        break
                    
                    # 模拟损失值逐渐下降
                    progress = (epoch - 1) * steps_per_epoch + step
                    total_progress = total_epochs * steps_per_epoch
                    simulated_loss = 2.0 * (1 - progress / total_progress) + 0.1
                    
                    self._update_progress(
                        current_epoch=epoch,
                        total_epochs=total_epochs,
                        current_step=step,
                        total_steps=steps_per_epoch,
                        loss=simulated_loss,
                        mel_loss=simulated_loss * 0.6,
                        kl_loss=simulated_loss * 0.2,
                        learning_rate=self.config.learning_rate * (0.99 ** epoch),
                        status=TrainingStatus.TRAINING,
                        message=f"模拟训练: Epoch {epoch}/{total_epochs}, Step {step}/{steps_per_epoch}"
                    )
                    
                    # 模拟训练时间
                    time.sleep(0.02)
                
                self._log(f"Epoch {epoch}/{total_epochs} 完成, Loss: {self._progress.loss:.4f}")
            
            # 创建模拟的训练信息文件
            model_dir = self.config.output_path / "model"
            model_dir.mkdir(parents=True, exist_ok=True)
            
            info_path = model_dir / f"{self.config.speaker_name}_training_info.json"
            
            model_info = {
                "speaker_name": self.config.speaker_name,
                "model_type": self.config.model_type,
                "sample_rate": self.config.sample_rate,
                "epochs": total_epochs,
                "final_loss": self._progress.loss,
                "language": self.config.language,
                "warning": "⚠️ 这是模拟训练生成的信息文件，不是有效的 TTS 模型！",
                "solution": "请安装 Coqui TTS 后重新训练: pip install TTS>=0.22.0"
            }
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(model_info, f, indent=2, ensure_ascii=False)
            
            result.success = True
            result.model_path = info_path
            result.model_size = info_path.stat().st_size
            result.total_epochs = total_epochs
            result.final_loss = self._progress.loss
            result.training_time = time.time() - self._start_time
            
            self._update_progress(
                status=TrainingStatus.COMPLETED,
                message="模拟训练完成（注意：未生成有效的 TTS 模型）"
            )
            
            self._log("⚠️ 模拟训练已完成，但由于 Coqui TTS 未安装，没有生成有效的模型！")
            self._log(f"训练信息已保存至: {info_path}")
            
        except Exception as e:
            result.success = False
            result.error_message = str(e)
            self._update_progress(
                status=TrainingStatus.FAILED,
                message=f"模拟训练失败: {e}"
            )
        
        return result
    
    def stop(self) -> None:
        """停止训练"""
        self._should_stop = True
        self._log("收到停止训练请求...")
        
        if self._trainer:
            self._log("正在停止训练...")
    
    def pause(self) -> Optional[Path]:
        """
        暂停训练并保存检查点
        
        Returns:
            检查点路径，如果失败返回 None
        """
        self._log("暂停训练...")
        self._update_progress(
            status=TrainingStatus.PAUSED,
            message="训练已暂停"
        )
        
        # 保存当前状态作为检查点
        checkpoint_dir = self.config.output_path / "training_workspace" / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{self._progress.current_epoch}.json"
        
        checkpoint_data = {
            "epoch": self._progress.current_epoch,
            "step": self._progress.current_step,
            "loss": self._progress.loss,
            "learning_rate": self._progress.learning_rate,
            "elapsed_time": self._progress.elapsed_time,
            "config": self.config.to_dict()
        }
        
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
        
        self._log(f"检查点已保存: {checkpoint_path}")
        return checkpoint_path
    
    def get_progress(self) -> TrainingProgress:
        """
        获取当前训练进度
        
        Returns:
            训练进度对象
        """
        return self._progress
    
    def get_logs(self) -> List[str]:
        """
        获取训练日志
        
        Returns:
            日志列表
        """
        return self._logs.copy()


def create_trainer(config: TrainConfig, verbose: bool = False) -> CoquiTrainer:
    """
    创建训练器实例的工厂函数
    
    Args:
        config: 训练配置
        verbose: 是否输出详细日志
        
    Returns:
        CoquiTrainer 实例
    """
    return CoquiTrainer(config, verbose)
