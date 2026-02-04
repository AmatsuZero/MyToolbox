#!/usr/bin/env python3
"""
训练进度与结果报告系统

提供训练过程中的进度显示、数据统计和结果报告功能。
支持终端进度条、详细日志输出和训练摘要生成。
"""

import sys
import time
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, TextIO
from datetime import datetime, timedelta

from mimic3_trainer import TrainingProgress, TrainingResult, TrainingStatus


@dataclass
class DatasetStatistics:
    """训练数据集统计信息"""
    total_audio_files: int = 0          # 音频文件总数
    total_subtitle_files: int = 0       # 字幕文件总数
    total_segments: int = 0             # 训练片段总数
    total_audio_duration: float = 0.0   # 音频总时长（秒）
    avg_segment_duration: float = 0.0   # 平均片段时长（秒）
    min_segment_duration: float = 0.0   # 最短片段时长（秒）
    max_segment_duration: float = 0.0   # 最长片段时长（秒）
    total_text_characters: int = 0      # 文本总字符数
    unique_characters: int = 0          # 唯一字符数
    vocabulary_size: int = 0            # 词汇表大小
    sample_rate: int = 22050            # 音频采样率
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "total_audio_files": self.total_audio_files,
            "total_subtitle_files": self.total_subtitle_files,
            "total_segments": self.total_segments,
            "total_audio_duration": self.total_audio_duration,
            "avg_segment_duration": self.avg_segment_duration,
            "min_segment_duration": self.min_segment_duration,
            "max_segment_duration": self.max_segment_duration,
            "total_text_characters": self.total_text_characters,
            "unique_characters": self.unique_characters,
            "vocabulary_size": self.vocabulary_size,
            "sample_rate": self.sample_rate
        }


@dataclass
class TrainingSummary:
    """训练摘要报告"""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    total_duration: float = 0.0         # 总训练时长（秒）
    total_epochs: int = 0               # 训练轮数
    final_loss: float = 0.0             # 最终损失值
    best_loss: float = float('inf')     # 最佳损失值
    best_epoch: int = 0                 # 最佳损失对应的 epoch
    model_path: Optional[Path] = None   # 模型文件路径
    model_size: int = 0                 # 模型文件大小（字节）
    checkpoint_path: Optional[Path] = None  # 检查点路径
    dataset_stats: Optional[DatasetStatistics] = None  # 数据集统计
    success: bool = False               # 是否成功
    error_message: str = ""             # 错误信息
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "total_duration": self.total_duration,
            "total_epochs": self.total_epochs,
            "final_loss": self.final_loss,
            "best_loss": self.best_loss,
            "best_epoch": self.best_epoch,
            "model_path": str(self.model_path) if self.model_path else None,
            "model_size": self.model_size,
            "checkpoint_path": str(self.checkpoint_path) if self.checkpoint_path else None,
            "dataset_stats": self.dataset_stats.to_dict() if self.dataset_stats else None,
            "success": self.success,
            "error_message": self.error_message
        }


class ProgressBar:
    """终端进度条"""
    
    def __init__(self, total: int, width: int = 50, prefix: str = "Progress", 
                 fill: str = "█", empty: str = "░"):
        """
        初始化进度条
        
        Args:
            total: 总数
            width: 进度条宽度
            prefix: 前缀文本
            fill: 填充字符
            empty: 空白字符
        """
        self.total = total
        self.width = width
        self.prefix = prefix
        self.fill = fill
        self.empty = empty
        self.current = 0
        self._start_time = time.time()
    
    def update(self, current: int, suffix: str = "") -> None:
        """
        更新进度条
        
        Args:
            current: 当前进度
            suffix: 后缀文本
        """
        self.current = current
        
        if self.total == 0:
            percentage = 100.0
        else:
            percentage = (current / self.total) * 100
        
        filled_width = int(self.width * current / self.total) if self.total > 0 else self.width
        bar = self.fill * filled_width + self.empty * (self.width - filled_width)
        
        # 计算剩余时间
        elapsed = time.time() - self._start_time
        if current > 0:
            eta = (elapsed / current) * (self.total - current)
            eta_str = self._format_time(eta)
        else:
            eta_str = "--:--:--"
        
        elapsed_str = self._format_time(elapsed)
        
        line = f"\r{self.prefix}: |{bar}| {percentage:5.1f}% [{elapsed_str}<{eta_str}] {suffix}"
        
        sys.stdout.write(line)
        sys.stdout.flush()
    
    def finish(self, message: str = "完成") -> None:
        """完成进度条"""
        self.update(self.total, message)
        sys.stdout.write("\n")
        sys.stdout.flush()
    
    @staticmethod
    def _format_time(seconds: float) -> str:
        """格式化时间"""
        if seconds < 0:
            return "--:--:--"
        hours, remainder = divmod(int(seconds), 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class TrainReporter:
    """
    训练报告器
    
    提供训练过程中的进度显示、数据统计和结果报告功能。
    """
    
    def __init__(self, verbose: bool = False, output_stream: TextIO = None):
        """
        初始化报告器
        
        Args:
            verbose: 是否输出详细日志
            output_stream: 输出流（默认为 stdout）
        """
        self.verbose = verbose
        self.output = output_stream or sys.stdout
        self._progress_bar: Optional[ProgressBar] = None
        self._summary = TrainingSummary()
        self._loss_history: List[float] = []
        self._last_progress: Optional[TrainingProgress] = None
    
    def print(self, message: str, level: str = "info") -> None:
        """
        输出消息
        
        Args:
            message: 消息内容
            level: 日志级别 (info, warning, error, debug)
        """
        if level == "debug" and not self.verbose:
            return
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        prefix_map = {
            "info": "ℹ️ ",
            "warning": "⚠️ ",
            "error": "❌",
            "debug": "🔍",
            "success": "✅"
        }
        prefix = prefix_map.get(level, "")
        
        if self.verbose:
            self.output.write(f"[{timestamp}] {prefix} {message}\n")
        else:
            self.output.write(f"{prefix} {message}\n")
        self.output.flush()
    
    def print_header(self, title: str) -> None:
        """打印标题头"""
        separator = "=" * 60
        self.output.write(f"\n{separator}\n")
        self.output.write(f"  {title}\n")
        self.output.write(f"{separator}\n\n")
        self.output.flush()
    
    def print_section(self, title: str) -> None:
        """打印小节标题"""
        separator = "-" * 40
        self.output.write(f"\n{separator}\n")
        self.output.write(f"  {title}\n")
        self.output.write(f"{separator}\n")
        self.output.flush()
    
    def report_dataset_statistics(self, stats: DatasetStatistics) -> None:
        """
        报告数据集统计信息
        
        Args:
            stats: 数据集统计信息
        """
        self._summary.dataset_stats = stats
        
        self.print_header("训练数据统计")
        
        # 文件统计
        self.output.write(f"  📁 音频文件数量:     {stats.total_audio_files}\n")
        self.output.write(f"  📄 字幕文件数量:     {stats.total_subtitle_files}\n")
        self.output.write(f"  🎵 训练片段数量:     {stats.total_segments}\n")
        self.output.write("\n")
        
        # 时长统计
        total_duration_str = self._format_duration(stats.total_audio_duration)
        avg_duration_str = f"{stats.avg_segment_duration:.2f}s"
        min_duration_str = f"{stats.min_segment_duration:.2f}s"
        max_duration_str = f"{stats.max_segment_duration:.2f}s"
        
        self.output.write(f"  ⏱️  音频总时长:       {total_duration_str}\n")
        self.output.write(f"  📊 平均片段时长:     {avg_duration_str}\n")
        self.output.write(f"  📉 最短片段时长:     {min_duration_str}\n")
        self.output.write(f"  📈 最长片段时长:     {max_duration_str}\n")
        self.output.write("\n")
        
        # 文本统计
        self.output.write(f"  📝 文本总字符数:     {stats.total_text_characters:,}\n")
        self.output.write(f"  🔤 唯一字符数:       {stats.unique_characters}\n")
        self.output.write(f"  📚 词汇表大小:       {stats.vocabulary_size}\n")
        self.output.write("\n")
        
        # 音频参数
        self.output.write(f"  🎛️  音频采样率:       {stats.sample_rate} Hz\n")
        self.output.write("\n")
        self.output.flush()
    
    def start_training(self, total_epochs: int, speaker_name: str) -> None:
        """
        开始训练时调用
        
        Args:
            total_epochs: 总训练轮数
            speaker_name: 说话人名称
        """
        self._summary.start_time = datetime.now()
        self._summary.total_epochs = total_epochs
        
        self.print_header(f"开始训练: {speaker_name}")
        self.output.write(f"  🎯 目标轮数:         {total_epochs}\n")
        self.output.write(f"  🕐 开始时间:         {self._summary.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.output.write("\n")
        self.output.flush()
        
        # 创建进度条
        self._progress_bar = ProgressBar(
            total=total_epochs,
            prefix="训练进度",
            width=40
        )
    
    def update_progress(self, progress: TrainingProgress) -> None:
        """
        更新训练进度
        
        Args:
            progress: 训练进度信息
        """
        self._last_progress = progress
        
        # 记录损失值历史
        if progress.loss > 0:
            self._loss_history.append(progress.loss)
            
            # 更新最佳损失值
            if progress.loss < self._summary.best_loss:
                self._summary.best_loss = progress.loss
                self._summary.best_epoch = progress.current_epoch
        
        # 更新进度条
        if self._progress_bar:
            suffix = f"Epoch {progress.current_epoch}/{progress.total_epochs}, Loss: {progress.loss:.4f}"
            self._progress_bar.update(progress.current_epoch, suffix)
        
        # 详细模式下输出更多信息
        if self.verbose and progress.current_step % 100 == 0:
            self.print(
                f"Step {progress.current_step}/{progress.total_steps}, "
                f"Loss: {progress.loss:.6f}, LR: {progress.learning_rate:.8f}",
                level="debug"
            )
    
    def report_epoch_complete(self, epoch: int, loss: float, 
                               learning_rate: float, epoch_time: float) -> None:
        """
        报告 epoch 完成
        
        Args:
            epoch: 当前 epoch
            loss: 当前损失值
            learning_rate: 当前学习率
            epoch_time: epoch 耗时（秒）
        """
        if self.verbose:
            self.print(
                f"Epoch {epoch} 完成 - Loss: {loss:.6f}, "
                f"LR: {learning_rate:.8f}, Time: {epoch_time:.2f}s",
                level="info"
            )
    
    def finish_training(self, result: TrainingResult) -> None:
        """
        完成训练时调用
        
        Args:
            result: 训练结果
        """
        self._summary.end_time = datetime.now()
        self._summary.total_duration = result.training_time
        self._summary.final_loss = result.final_loss
        self._summary.model_path = result.model_path
        self._summary.model_size = result.model_size
        self._summary.checkpoint_path = result.checkpoint_path
        self._summary.success = result.success
        self._summary.error_message = result.error_message
        
        # 完成进度条
        if self._progress_bar:
            if result.success:
                self._progress_bar.finish("✅ 训练完成")
            else:
                self._progress_bar.finish("❌ 训练失败")
        
        # 打印训练摘要
        self._print_training_summary()
    
    def _print_training_summary(self) -> None:
        """打印训练摘要报告"""
        self.print_header("训练摘要报告")
        
        # 训练状态
        status_icon = "✅" if self._summary.success else "❌"
        status_text = "成功" if self._summary.success else "失败"
        self.output.write(f"  {status_icon} 训练状态:         {status_text}\n")
        
        if self._summary.error_message:
            self.output.write(f"  ⚠️  错误信息:         {self._summary.error_message}\n")
        
        self.output.write("\n")
        
        # 时间统计
        if self._summary.start_time:
            self.output.write(f"  🕐 开始时间:         {self._summary.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        if self._summary.end_time:
            self.output.write(f"  🕑 结束时间:         {self._summary.end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        duration_str = self._format_duration(self._summary.total_duration)
        self.output.write(f"  ⏱️  总耗时:           {duration_str}\n")
        self.output.write("\n")
        
        # 训练统计
        self.output.write(f"  📊 训练轮数:         {self._summary.total_epochs}\n")
        self.output.write(f"  📉 最终损失值:       {self._summary.final_loss:.6f}\n")
        self.output.write(f"  🏆 最佳损失值:       {self._summary.best_loss:.6f} (Epoch {self._summary.best_epoch})\n")
        self.output.write("\n")
        
        # 模型信息
        if self._summary.model_path:
            model_size_str = self._format_file_size(self._summary.model_size)
            self.output.write(f"  💾 模型文件:         {self._summary.model_path}\n")
            self.output.write(f"  📦 模型大小:         {model_size_str}\n")
        
        if self._summary.checkpoint_path:
            self.output.write(f"  💿 检查点文件:       {self._summary.checkpoint_path}\n")
        
        self.output.write("\n")
        
        # 损失值变化趋势
        if len(self._loss_history) > 1:
            self.print_section("损失值变化趋势")
            self._print_loss_trend()
        
        self.output.flush()
    
    def _print_loss_trend(self) -> None:
        """打印损失值变化趋势（简易 ASCII 图表）"""
        if len(self._loss_history) < 2:
            return
        
        # 采样点数量
        num_samples = min(20, len(self._loss_history))
        step = len(self._loss_history) // num_samples
        sampled = [self._loss_history[i * step] for i in range(num_samples)]
        
        # 计算图表参数
        max_loss = max(sampled)
        min_loss = min(sampled)
        height = 8
        
        if max_loss == min_loss:
            # 所有值相同
            self.output.write("  损失值保持稳定\n")
            return
        
        # 绘制简易图表
        self.output.write("\n")
        for row in range(height, -1, -1):
            threshold = min_loss + (max_loss - min_loss) * row / height
            line = "  "
            for val in sampled:
                if val >= threshold:
                    line += "█"
                else:
                    line += " "
            
            # 添加纵轴标签
            if row == height:
                line += f" {max_loss:.4f}"
            elif row == 0:
                line += f" {min_loss:.4f}"
            
            self.output.write(line + "\n")
        
        # 横轴
        self.output.write("  " + "─" * num_samples + "\n")
        self.output.write(f"  1{' ' * (num_samples - 2)}{len(self._loss_history)}\n")
        self.output.write("  (Epoch)\n\n")
    
    def save_report(self, output_path: Path) -> None:
        """
        保存训练报告到文件
        
        Args:
            output_path: 输出路径
        """
        report_data = {
            "summary": self._summary.to_dict(),
            "loss_history": self._loss_history
        }
        
        # 保存 JSON 格式
        json_path = output_path / "training_report.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        self.print(f"训练报告已保存: {json_path}", level="success")
        
        # 保存文本格式
        txt_path = output_path / "training_report.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            self._write_text_report(f)
        
        self.print(f"文本报告已保存: {txt_path}", level="success")
    
    def _write_text_report(self, f: TextIO) -> None:
        """写入文本格式报告"""
        f.write("=" * 60 + "\n")
        f.write("  Mimic3 语音模型训练报告\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"训练状态: {'成功' if self._summary.success else '失败'}\n")
        if self._summary.error_message:
            f.write(f"错误信息: {self._summary.error_message}\n")
        f.write("\n")
        
        if self._summary.start_time:
            f.write(f"开始时间: {self._summary.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        if self._summary.end_time:
            f.write(f"结束时间: {self._summary.end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总耗时: {self._format_duration(self._summary.total_duration)}\n")
        f.write("\n")
        
        f.write(f"训练轮数: {self._summary.total_epochs}\n")
        f.write(f"最终损失值: {self._summary.final_loss:.6f}\n")
        f.write(f"最佳损失值: {self._summary.best_loss:.6f} (Epoch {self._summary.best_epoch})\n")
        f.write("\n")
        
        if self._summary.model_path:
            f.write(f"模型文件: {self._summary.model_path}\n")
            f.write(f"模型大小: {self._format_file_size(self._summary.model_size)}\n")
        
        if self._summary.checkpoint_path:
            f.write(f"检查点文件: {self._summary.checkpoint_path}\n")
        f.write("\n")
        
        # 数据集统计
        if self._summary.dataset_stats:
            stats = self._summary.dataset_stats
            f.write("-" * 40 + "\n")
            f.write("  数据集统计\n")
            f.write("-" * 40 + "\n")
            f.write(f"音频文件数量: {stats.total_audio_files}\n")
            f.write(f"字幕文件数量: {stats.total_subtitle_files}\n")
            f.write(f"训练片段数量: {stats.total_segments}\n")
            f.write(f"音频总时长: {self._format_duration(stats.total_audio_duration)}\n")
            f.write(f"平均片段时长: {stats.avg_segment_duration:.2f}s\n")
            f.write(f"文本总字符数: {stats.total_text_characters:,}\n")
            f.write(f"词汇表大小: {stats.vocabulary_size}\n")
            f.write("\n")
    
    def get_summary(self) -> TrainingSummary:
        """
        获取训练摘要
        
        Returns:
            训练摘要对象
        """
        return self._summary
    
    @staticmethod
    def _format_duration(seconds: float) -> str:
        """格式化时长"""
        if seconds < 60:
            return f"{seconds:.1f} 秒"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f} 分钟"
        else:
            hours = seconds / 3600
            return f"{hours:.1f} 小时"
    
    @staticmethod
    def _format_file_size(size_bytes: int) -> str:
        """格式化文件大小"""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


def create_reporter(verbose: bool = False, output_stream: TextIO = None) -> TrainReporter:
    """
    创建报告器实例的工厂函数
    
    Args:
        verbose: 是否输出详细日志
        output_stream: 输出流
        
    Returns:
        TrainReporter 实例
    """
    return TrainReporter(verbose, output_stream)


def collect_dataset_statistics(dataset_path: Path) -> DatasetStatistics:
    """
    收集数据集统计信息
    
    Args:
        dataset_path: 数据集路径
        
    Returns:
        数据集统计信息
    """
    stats = DatasetStatistics()
    
    if not dataset_path.exists():
        return stats
    
    # 统计文件数量
    audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
    subtitle_extensions = {'.srt', '.vtt', '.txt', '.json'}
    
    all_text = ""
    segment_durations = []
    
    # 遍历数据集目录
    for file_path in dataset_path.rglob("*"):
        if file_path.is_file():
            suffix = file_path.suffix.lower()
            
            if suffix in audio_extensions:
                stats.total_audio_files += 1
                # 尝试获取音频时长（需要额外库支持）
                # 这里使用简化的估算方法
                file_size = file_path.stat().st_size
                # 假设 16-bit 44.1kHz 立体声
                estimated_duration = file_size / (44100 * 2 * 2)
                segment_durations.append(estimated_duration)
                stats.total_audio_duration += estimated_duration
                
            elif suffix in subtitle_extensions:
                stats.total_subtitle_files += 1
                # 读取文本内容
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                        all_text += text
                except Exception:
                    pass
    
    # 检查是否有 metadata.csv 或 metadata.json
    metadata_csv = dataset_path / "metadata.csv"
    metadata_json = dataset_path / "metadata.json"
    
    if metadata_csv.exists():
        try:
            with open(metadata_csv, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                stats.total_segments = len(lines) - 1  # 减去标题行
        except Exception:
            pass
    elif metadata_json.exists():
        try:
            with open(metadata_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    stats.total_segments = len(data)
        except Exception:
            pass
    else:
        stats.total_segments = stats.total_audio_files
    
    # 计算片段时长统计
    if segment_durations:
        stats.avg_segment_duration = sum(segment_durations) / len(segment_durations)
        stats.min_segment_duration = min(segment_durations)
        stats.max_segment_duration = max(segment_durations)
    
    # 计算文本统计
    if all_text:
        stats.total_text_characters = len(all_text)
        stats.unique_characters = len(set(all_text))
        # 简单的词汇统计（按空格分词）
        words = set(all_text.split())
        stats.vocabulary_size = len(words)
    
    return stats
