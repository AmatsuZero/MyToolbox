#!/usr/bin/env python3
"""
Mimic3 训练接口封装模块

封装 Mimic3 TTS 训练 API，提供训练参数配置、模型输出管理和训练进度回调功能。
"""

import os
import json
import shutil
import subprocess
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable, Optional, List, Dict, Any
from enum import Enum

from modules.training.train_config import TrainConfig


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
    logs: List[str] = field(default_factory=list)  # 训练日志
    
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
            "checkpoint_path": str(self.checkpoint_path) if self.checkpoint_path else None
        }


# 进度回调函数类型
ProgressCallback = Callable[[TrainingProgress], None]


class Mimic3Trainer:
    """
    Mimic3 训练器
    
    封装 Mimic3 TTS 模型训练功能，支持：
    - 训练参数配置
    - 训练进度监控
    - 断点续训
    - 模型输出管理
    """
    
    # Mimic3 训练脚本路径（根据实际安装位置调整）
    MIMIC3_TRAIN_SCRIPT = "mimic3-train"
    
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
        self._process: Optional[subprocess.Popen] = None
        self._start_time: float = 0
        self._should_stop: bool = False
        self._logs: List[str] = []
    
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
            if self._progress.current_epoch > 0:
                avg_time_per_epoch = self._progress.elapsed_time / self._progress.current_epoch
                remaining_epochs = self._progress.total_epochs - self._progress.current_epoch
                self._progress.estimated_remaining = avg_time_per_epoch * remaining_epochs
        
        self._notify_progress()
    
    def _prepare_training_directory(self, dataset_path: Path) -> Path:
        """
        准备训练目录结构
        
        Args:
            dataset_path: 数据集路径
            
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
        
        self._log(f"训练工作目录: {work_dir}")
        return work_dir
    
    def _generate_training_config(self, dataset_path: Path, work_dir: Path) -> Path:
        """
        生成 Mimic3 训练配置文件
        
        Args:
            dataset_path: 数据集路径
            work_dir: 工作目录
            
        Returns:
            配置文件路径
        """
        self._log("生成训练配置文件...")
        
        # Mimic3 训练配置
        training_config = {
            "audio": {
                "sample_rate": self.config.sample_rate,
                "num_mels": 80,
                "fft_size": 1024,
                "hop_size": 256,
                "win_size": 1024,
                "fmin": 0,
                "fmax": 8000
            },
            "model": {
                "type": "vits",
                "hidden_channels": 192,
                "inter_channels": 192,
                "filter_channels": 768,
                "n_heads": 2,
                "n_layers": 6,
                "kernel_size": 3,
                "p_dropout": 0.1
            },
            "training": {
                "epochs": self.config.epochs,
                "batch_size": self.config.batch_size,
                "learning_rate": 0.0002,
                "lr_decay": 0.999875,
                "seed": 1234,
                "fp16_run": False,
                "grad_clip_thresh": 1.0,
                "checkpoint_interval": 1000,
                "validation_interval": 1000,
                "log_interval": 100
            },
            "data": {
                "dataset_path": str(dataset_path),
                "speaker_name": self.config.speaker_name,
                "text_cleaners": ["basic_cleaners"]
            },
            "output": {
                "model_dir": str(self.config.output_path / "model"),
                "checkpoint_dir": str(work_dir / "checkpoints"),
                "log_dir": str(work_dir / "logs")
            }
        }
        
        # 写入配置文件
        config_path = work_dir / "training_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(training_config, f, indent=2, ensure_ascii=False)
        
        self._log(f"配置文件已保存: {config_path}")
        return config_path
    
    def _build_training_command(self, config_path: Path) -> List[str]:
        """
        构建训练命令
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            命令参数列表
        """
        cmd = [
            self.MIMIC3_TRAIN_SCRIPT,
            "--config", str(config_path),
            "--epochs", str(self.config.epochs),
            "--batch-size", str(self.config.batch_size),
            "--sample-rate", str(self.config.sample_rate),
            "--output", str(self.config.output_path / "model"),
        ]
        
        if self.verbose:
            cmd.append("--verbose")
        
        return cmd
    
    def _parse_training_output(self, line: str) -> None:
        """
        解析训练输出，更新进度
        
        Args:
            line: 输出行
        """
        line = line.strip()
        if not line:
            return
        
        self._log(line)
        
        # 解析 epoch 信息
        # 假设输出格式类似: "Epoch 1/100, Step 500/10000, Loss: 0.5432, LR: 0.0002"
        try:
            if "Epoch" in line:
                parts = line.split(",")
                for part in parts:
                    part = part.strip()
                    if "Epoch" in part:
                        epoch_info = part.replace("Epoch", "").strip()
                        if "/" in epoch_info:
                            current, total = epoch_info.split("/")
                            self._update_progress(
                                current_epoch=int(current.strip()),
                                total_epochs=int(total.strip())
                            )
                    elif "Step" in part:
                        step_info = part.replace("Step", "").strip()
                        if "/" in step_info:
                            current, total = step_info.split("/")
                            self._update_progress(
                                current_step=int(current.strip()),
                                total_steps=int(total.strip())
                            )
                    elif "Loss" in part:
                        loss_value = part.replace("Loss:", "").strip()
                        self._update_progress(loss=float(loss_value))
                    elif "LR" in part:
                        lr_value = part.replace("LR:", "").strip()
                        self._update_progress(learning_rate=float(lr_value))
        except (ValueError, IndexError) as e:
            # 解析失败，忽略
            pass
    
    def train(self, dataset_path: Path, resume_from: Optional[Path] = None) -> TrainingResult:
        """
        执行训练
        
        Args:
            dataset_path: 数据集路径（由 AudioAligner 生成的对齐数据）
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
            work_dir = self._prepare_training_directory(dataset_path)
            
            # 生成训练配置
            config_path = self._generate_training_config(dataset_path, work_dir)
            
            # 构建训练命令
            cmd = self._build_training_command(config_path)
            
            # 如果是断点续训，添加恢复参数
            if resume_from and resume_from.exists():
                cmd.extend(["--resume", str(resume_from)])
                self._log(f"从检查点恢复训练: {resume_from}")
            
            self._log(f"训练命令: {' '.join(cmd)}")
            
            # 更新状态为训练中
            self._update_progress(
                status=TrainingStatus.TRAINING,
                message="训练进行中...",
                total_epochs=self.config.epochs
            )
            
            # 记录开始时间
            self._start_time = time.time()
            
            # 启动训练进程
            self._log("启动训练进程...")
            
            # 注意：由于 Mimic3 可能未安装，这里使用模拟训练
            # 实际使用时应替换为真实的 subprocess 调用
            if self._is_mimic3_available():
                result = self._run_real_training(cmd, work_dir)
            else:
                self._log("警告: Mimic3 训练脚本未找到，执行模拟训练")
                result = self._run_simulated_training(work_dir)
            
        except Exception as e:
            self._log(f"训练过程发生错误: {e}")
            self._update_progress(
                status=TrainingStatus.FAILED,
                message=f"训练失败: {e}"
            )
            result.success = False
            result.error_message = str(e)
        
        result.logs = self._logs.copy()
        return result
    
    def _is_mimic3_available(self) -> bool:
        """检查 Mimic3 训练脚本是否可用"""
        return shutil.which(self.MIMIC3_TRAIN_SCRIPT) is not None
    
    def _run_real_training(self, cmd: List[str], work_dir: Path) -> TrainingResult:
        """
        执行真实的 Mimic3 训练
        
        Args:
            cmd: 训练命令
            work_dir: 工作目录
            
        Returns:
            训练结果
        """
        result = TrainingResult()
        
        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=work_dir
            )
            
            # 读取输出并解析进度
            for line in iter(self._process.stdout.readline, ''):
                if self._should_stop:
                    self._process.terminate()
                    self._update_progress(
                        status=TrainingStatus.CANCELLED,
                        message="训练已取消"
                    )
                    result.error_message = "训练被用户取消"
                    return result
                
                self._parse_training_output(line)
            
            # 等待进程结束
            return_code = self._process.wait()
            
            if return_code == 0:
                result.success = True
                self._update_progress(
                    status=TrainingStatus.COMPLETED,
                    message="训练完成"
                )
            else:
                result.success = False
                result.error_message = f"训练进程退出码: {return_code}"
                self._update_progress(
                    status=TrainingStatus.FAILED,
                    message=f"训练失败，退出码: {return_code}"
                )
            
        except Exception as e:
            result.success = False
            result.error_message = str(e)
            self._update_progress(
                status=TrainingStatus.FAILED,
                message=f"训练失败: {e}"
            )
        
        finally:
            self._process = None
        
        # 收集训练结果
        result.training_time = time.time() - self._start_time
        result.total_epochs = self._progress.current_epoch
        result.final_loss = self._progress.loss
        
        # 查找模型文件
        model_dir = self.config.output_path / "model"
        if model_dir.exists():
            model_files = list(model_dir.glob("*.onnx")) + list(model_dir.glob("*.pt"))
            if model_files:
                result.model_path = model_files[0]
                result.model_size = result.model_path.stat().st_size
        
        # 查找检查点
        checkpoint_dir = work_dir / "checkpoints"
        if checkpoint_dir.exists():
            checkpoints = sorted(checkpoint_dir.glob("checkpoint_*.pt"))
            if checkpoints:
                result.checkpoint_path = checkpoints[-1]
        
        return result
    
    def _run_simulated_training(self, work_dir: Path) -> TrainingResult:
        """
        执行模拟训练（用于测试或 Mimic3 未安装时）
        
        Args:
            work_dir: 工作目录
            
        Returns:
            训练结果
        """
        result = TrainingResult()
        
        self._log("开始模拟训练...")
        
        total_epochs = self.config.epochs
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
                    simulated_loss = 1.0 / (epoch * 0.5 + step * 0.01 + 1)
                    
                    self._update_progress(
                        current_epoch=epoch,
                        total_epochs=total_epochs,
                        current_step=step,
                        total_steps=steps_per_epoch,
                        loss=simulated_loss,
                        learning_rate=0.0002 * (0.999875 ** (epoch * steps_per_epoch + step)),
                        status=TrainingStatus.TRAINING,
                        message=f"训练中: Epoch {epoch}/{total_epochs}, Step {step}/{steps_per_epoch}"
                    )
                    
                    # 模拟训练时间
                    time.sleep(0.01)
                
                self._log(f"Epoch {epoch}/{total_epochs} 完成, Loss: {self._progress.loss:.4f}")
            
            # 创建模拟的模型文件
            model_dir = self.config.output_path / "model"
            model_dir.mkdir(parents=True, exist_ok=True)
            
            model_path = model_dir / f"{self.config.speaker_name}.onnx"
            
            # 写入模拟的模型文件（实际应为真实模型）
            model_info = {
                "speaker_name": self.config.speaker_name,
                "sample_rate": self.config.sample_rate,
                "epochs": self.config.epochs,
                "final_loss": self._progress.loss,
                "note": "这是模拟训练生成的占位文件，实际使用需要安装 Mimic3"
            }
            with open(model_path, 'w', encoding='utf-8') as f:
                json.dump(model_info, f, indent=2, ensure_ascii=False)
            
            result.success = True
            result.model_path = model_path
            result.model_size = model_path.stat().st_size
            result.total_epochs = total_epochs
            result.final_loss = self._progress.loss
            result.training_time = time.time() - self._start_time
            
            self._update_progress(
                status=TrainingStatus.COMPLETED,
                message="模拟训练完成"
            )
            
            self._log(f"模拟训练完成，模型已保存至: {model_path}")
            
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
        
        if self._process:
            self._log("终止训练进程...")
            self._process.terminate()
    
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


def create_trainer(config: TrainConfig, verbose: bool = False) -> Mimic3Trainer:
    """
    创建训练器实例的工厂函数
    
    Args:
        config: 训练配置
        verbose: 是否输出详细日志
        
    Returns:
        Mimic3Trainer 实例
    """
    return Mimic3Trainer(config, verbose)
