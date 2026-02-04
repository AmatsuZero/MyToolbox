#!/usr/bin/env python3
"""
训练模式主控制器

整合训练数据管理、音频对齐、Mimic3 训练等模块，
实现完整的语音模型训练工作流程。
"""

import logging
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Dict, Any
from enum import Enum

from modules.training.train_config import TrainConfig
from modules.training.train_data_manager import TrainDataManager, TrainDataPair, ScanResult, MediaFile
from modules.audio_aligner import AudioAligner, AlignmentResult
from modules.training.mimic3_trainer import Mimic3Trainer, TrainingProgress, TrainingResult, TrainingStatus
from modules.training.mimic3_checker import Mimic3Checker, ensure_mimic3_available

# 设置日志
logger = logging.getLogger(__name__)


class TrainPhase(Enum):
    """训练阶段枚举"""
    IDLE = "idle"                       # 空闲
    CHECKING_DEPS = "checking_deps"     # 检查依赖
    SCANNING_DATA = "scanning_data"     # 扫描数据
    MATCHING_FILES = "matching_files"   # 匹配文件
    GENERATING_SUBTITLES = "generating_subtitles"  # 生成字幕
    ALIGNING_DATA = "aligning_data"     # 对齐数据
    TRAINING = "training"               # 训练中
    COMPLETED = "completed"             # 完成
    FAILED = "failed"                   # 失败


@dataclass
class TrainControllerProgress:
    """训练控制器进度信息"""
    phase: TrainPhase = TrainPhase.IDLE
    phase_message: str = ""
    overall_progress: float = 0.0       # 总体进度 (0-100)
    
    # 各阶段详情
    scan_result: Optional[ScanResult] = None
    matched_pairs: int = 0
    unmatched_files: int = 0
    alignment_progress: int = 0         # 已对齐数量
    alignment_total: int = 0            # 总数量
    
    # 训练进度
    training_progress: Optional[TrainingProgress] = None
    
    # 时间统计
    start_time: float = 0
    elapsed_time: float = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "phase": self.phase.value,
            "phase_message": self.phase_message,
            "overall_progress": self.overall_progress,
            "matched_pairs": self.matched_pairs,
            "unmatched_files": self.unmatched_files,
            "alignment_progress": self.alignment_progress,
            "alignment_total": self.alignment_total,
            "elapsed_time": self.elapsed_time,
            "training_progress": self.training_progress.to_dict() if self.training_progress else None
        }


@dataclass
class TrainControllerResult:
    """训练控制器结果"""
    success: bool = False
    error_message: str = ""
    
    # 数据统计
    total_audio_files: int = 0
    total_subtitle_files: int = 0
    matched_pairs: int = 0
    generated_subtitles: int = 0
    aligned_segments: int = 0
    total_audio_duration: float = 0.0   # 总音频时长（秒）
    
    # 训练结果
    training_result: Optional[TrainingResult] = None
    model_path: Optional[Path] = None
    
    # 时间统计
    total_time: float = 0.0
    
    # 日志
    logs: List[str] = field(default_factory=list)
    
    def get_summary(self) -> str:
        """获取结果摘要"""
        lines = [
            "=" * 60,
            "训练结果摘要",
            "=" * 60,
            f"状态: {'成功 ✅' if self.success else '失败 ❌'}",
        ]
        
        if self.error_message:
            lines.append(f"错误: {self.error_message}")
        
        lines.extend([
            "",
            "数据统计:",
            f"  - 音频文件: {self.total_audio_files}",
            f"  - 字幕文件: {self.total_subtitle_files}",
            f"  - 匹配的数据对: {self.matched_pairs}",
            f"  - 自动生成字幕: {self.generated_subtitles}",
            f"  - 对齐的片段: {self.aligned_segments}",
            f"  - 总音频时长: {self._format_duration(self.total_audio_duration)}",
        ])
        
        if self.training_result:
            lines.extend([
                "",
                "训练结果:",
                f"  - 训练轮数: {self.training_result.total_epochs}",
                f"  - 最终损失: {self.training_result.final_loss:.4f}",
                f"  - 训练时间: {self._format_duration(self.training_result.training_time)}",
            ])
            
            if self.model_path:
                lines.append(f"  - 模型路径: {self.model_path}")
                if self.training_result.model_size > 0:
                    size_mb = self.training_result.model_size / (1024 * 1024)
                    lines.append(f"  - 模型大小: {size_mb:.2f} MB")
        
        lines.extend([
            "",
            f"总耗时: {self._format_duration(self.total_time)}",
            "=" * 60,
        ])
        
        return "\n".join(lines)
    
    @staticmethod
    def _format_duration(seconds: float) -> str:
        """格式化时长"""
        if seconds < 60:
            return f"{seconds:.1f} 秒"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes} 分 {secs:.0f} 秒"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours} 小时 {minutes} 分"


# 进度回调类型
ControllerProgressCallback = Callable[[TrainControllerProgress], None]


class TrainController:
    """
    训练模式主控制器
    
    整合完整的训练工作流：
    1. 检查 Mimic3 依赖
    2. 扫描和匹配训练数据
    3. 为无字幕的媒体文件生成字幕（使用 Whisper）
    4. 对齐音频和字幕
    5. 执行 Mimic3 训练
    6. 输出训练结果
    """
    
    def __init__(self, config: TrainConfig, verbose: bool = False):
        """
        初始化训练控制器
        
        Args:
            config: 训练配置
            verbose: 是否输出详细日志
        """
        self.config = config
        self.verbose = verbose
        
        # 初始化各模块
        self._data_manager = TrainDataManager(output_dir=config.output_path)
        self._aligner = AudioAligner(config)
        self._trainer = Mimic3Trainer(config, verbose)
        self._checker = Mimic3Checker()
        
        # 进度跟踪
        self._progress = TrainControllerProgress()
        self._callbacks: List[ControllerProgressCallback] = []
        self._logs: List[str] = []
        self._should_stop = False
        
        # Whisper 生成器（延迟导入）
        self._whisper_generator = None
    
    def add_progress_callback(self, callback: ControllerProgressCallback) -> None:
        """添加进度回调"""
        self._callbacks.append(callback)
    
    def remove_progress_callback(self, callback: ControllerProgressCallback) -> None:
        """移除进度回调"""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    def _notify_progress(self) -> None:
        """通知进度更新"""
        for callback in self._callbacks:
            try:
                callback(self._progress)
            except Exception as e:
                self._log(f"进度回调执行失败: {e}")
    
    def _log(self, message: str) -> None:
        """记录日志"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self._logs.append(log_entry)
        logger.info(message)
        if self.verbose:
            print(log_entry)
    
    def _update_progress(
        self,
        phase: Optional[TrainPhase] = None,
        message: Optional[str] = None,
        overall: Optional[float] = None,
        **kwargs
    ) -> None:
        """更新进度"""
        if phase is not None:
            self._progress.phase = phase
        if message is not None:
            self._progress.phase_message = message
        if overall is not None:
            self._progress.overall_progress = overall
        
        for key, value in kwargs.items():
            if hasattr(self._progress, key):
                setattr(self._progress, key, value)
        
        # 更新已用时间
        if self._progress.start_time > 0:
            self._progress.elapsed_time = time.time() - self._progress.start_time
        
        self._notify_progress()
    
    def run(self, resume_from: Optional[Path] = None) -> TrainControllerResult:
        """
        执行完整的训练流程
        
        Args:
            resume_from: 断点续训的检查点路径（可选）
            
        Returns:
            TrainControllerResult: 训练结果
        """
        result = TrainControllerResult()
        self._logs = []
        self._should_stop = False
        self._progress.start_time = time.time()
        
        try:
            # 阶段 1: 检查依赖
            self._log("=" * 60)
            self._log("开始训练流程")
            self._log("=" * 60)
            
            if not self._check_dependencies():
                result.error_message = "依赖检查失败，请确保已安装 Mimic3"
                result.logs = self._logs.copy()
                return result
            
            if self._should_stop:
                return self._create_cancelled_result()
            
            # 阶段 2: 扫描训练数据
            scan_result = self._scan_training_data()
            if scan_result.total_count == 0:
                result.error_message = "未找到任何训练数据"
                result.logs = self._logs.copy()
                return result
            
            result.total_audio_files = len(scan_result.audio_files)
            result.total_subtitle_files = len(scan_result.subtitle_files)
            
            if self._should_stop:
                return self._create_cancelled_result()
            
            # 阶段 3: 匹配文件
            matched_pairs, unmatched = self._match_files(scan_result)
            result.matched_pairs = len(matched_pairs)
            
            if self._should_stop:
                return self._create_cancelled_result()
            
            # 阶段 4: 为未匹配的媒体文件生成字幕
            if unmatched:
                new_pairs, generated_count = self._generate_missing_subtitles(unmatched)
                matched_pairs.extend(new_pairs)
                result.generated_subtitles = generated_count
                result.matched_pairs = len(matched_pairs)
            
            if not matched_pairs:
                result.error_message = "没有可用的训练数据对"
                result.logs = self._logs.copy()
                return result
            
            if self._should_stop:
                return self._create_cancelled_result()
            
            # 阶段 5: 对齐数据
            alignment_results = self._align_data(matched_pairs)
            
            total_segments = 0
            total_duration = 0.0
            
            for align_result in alignment_results:
                if align_result.success:
                    total_segments += align_result.total_segments
                    total_duration += align_result.total_duration
            
            result.aligned_segments = total_segments
            result.total_audio_duration = total_duration
            
            if total_segments == 0:
                result.error_message = "数据对齐失败，没有生成有效的训练片段"
                result.logs = self._logs.copy()
                return result
            
            if self._should_stop:
                return self._create_cancelled_result()
            
            # 阶段 6: 执行训练
            dataset_path = self.config.output_path / "aligned"
            training_result = self._run_training(dataset_path, resume_from)
            
            result.training_result = training_result
            result.model_path = training_result.model_path
            result.success = training_result.success
            
            if not training_result.success:
                result.error_message = training_result.error_message
            
        except Exception as e:
            self._log(f"训练过程发生错误: {e}")
            self._update_progress(
                phase=TrainPhase.FAILED,
                message=f"训练失败: {e}"
            )
            result.success = False
            result.error_message = str(e)
        
        # 计算总时间
        result.total_time = time.time() - self._progress.start_time
        result.logs = self._logs.copy()
        
        # 输出结果摘要
        self._log(result.get_summary())
        
        return result
    
    def _check_dependencies(self) -> bool:
        """检查依赖"""
        self._update_progress(
            phase=TrainPhase.CHECKING_DEPS,
            message="检查 Mimic3 依赖...",
            overall=5.0
        )
        self._log("检查 Mimic3 依赖...")
        
        check_result = self._checker.check()
        
        if not check_result.is_installed:
            self._log(f"❌ Mimic3 未安装: {check_result.error_message}")
            self._update_progress(
                phase=TrainPhase.FAILED,
                message="Mimic3 未安装"
            )
            self._checker.print_install_guide()
            return False
        
        if not check_result.is_compatible:
            self._log(f"⚠️ Mimic3 版本不兼容: {check_result.error_message}")
            # 版本不兼容时给出警告，但继续执行
        
        self._log(f"✅ Mimic3 检查通过 (版本: {check_result.version})")
        return True
    
    def _scan_training_data(self) -> ScanResult:
        """扫描训练数据"""
        self._update_progress(
            phase=TrainPhase.SCANNING_DATA,
            message="扫描训练数据...",
            overall=10.0
        )
        self._log("扫描训练数据...")
        
        # 根据配置决定扫描方式
        if self.config.input_dir:
            self._log(f"扫描目录: {self.config.input_dir}")
            scan_result = self._data_manager.scan_directory(self.config.input_dir)
        else:
            self._log("从配置的文件列表扫描...")
            scan_result = self._data_manager.scan_files(
                audio_files=self.config.audio_files,
                subtitle_files=self.config.subtitle_files
            )
        
        self._progress.scan_result = scan_result
        self._log(scan_result.summary())
        
        return scan_result
    
    def _match_files(
        self,
        scan_result: ScanResult
    ) -> tuple[List[TrainDataPair], List[MediaFile]]:
        """匹配音频和字幕文件"""
        self._update_progress(
            phase=TrainPhase.MATCHING_FILES,
            message="匹配音频和字幕文件...",
            overall=20.0
        )
        self._log("匹配音频和字幕文件...")
        
        matched_pairs, unmatched = self._data_manager.match_audio_subtitle(
            scan_result,
            extract_from_video=True
        )
        
        self._progress.matched_pairs = len(matched_pairs)
        self._progress.unmatched_files = len(unmatched)
        
        self._log(f"匹配完成: {len(matched_pairs)} 对成功, {len(unmatched)} 个未匹配")
        
        return matched_pairs, unmatched
    
    def _generate_missing_subtitles(
        self,
        unmatched: List[MediaFile]
    ) -> tuple[List[TrainDataPair], int]:
        """为未匹配的媒体文件生成字幕"""
        self._update_progress(
            phase=TrainPhase.GENERATING_SUBTITLES,
            message="使用 Whisper 生成缺失的字幕...",
            overall=30.0
        )
        
        # 获取需要生成字幕的媒体文件
        media_for_whisper = self._data_manager.get_unmatched_media_for_whisper(unmatched)
        
        if not media_for_whisper:
            self._log("没有需要生成字幕的媒体文件")
            return [], 0
        
        self._log(f"需要为 {len(media_for_whisper)} 个媒体文件生成字幕...")
        
        new_pairs = []
        generated_count = 0
        
        # 延迟导入 Whisper 生成器
        try:
            from modules.whisper.whisper_subtitle_generator import WhisperSubtitleGenerator
            
            if self._whisper_generator is None:
                self._whisper_generator = WhisperSubtitleGenerator()
            
            subtitle_output_dir = self.config.output_path / "generated_subtitles"
            subtitle_output_dir.mkdir(parents=True, exist_ok=True)
            
            for i, media in enumerate(media_for_whisper):
                if self._should_stop:
                    break
                
                self._log(f"[{i+1}/{len(media_for_whisper)}] 为 {media.path.name} 生成字幕...")
                
                try:
                    # 使用 Whisper 生成字幕
                    subtitle_path = subtitle_output_dir / f"{media.stem}.srt"
                    
                    gen_result = self._whisper_generator.generate_subtitle(
                        media_path=media.path,
                        output_path=subtitle_path
                    )
                    success = gen_result.success
                    
                    if success and subtitle_path.exists():
                        # 确定音频路径
                        if media.media_type.value == "video":
                            # 从视频提取音频
                            audio_path = self._data_manager.extract_audio_from_video(
                                media.path,
                                sample_rate=self.config.sample_rate
                            )
                        else:
                            audio_path = media.path
                        
                        new_pairs.append(TrainDataPair(
                            audio_path=audio_path,
                            subtitle_path=subtitle_path,
                            source_video=media.path if media.media_type.value == "video" else None
                        ))
                        generated_count += 1
                        self._log(f"  ✅ 字幕生成成功: {subtitle_path.name}")
                    else:
                        self._log(f"  ❌ 字幕生成失败")
                        
                except Exception as e:
                    self._log(f"  ❌ 生成字幕时出错: {e}")
                
                # 更新进度
                progress = 30.0 + (i + 1) / len(media_for_whisper) * 10.0
                self._update_progress(overall=progress)
            
        except ImportError:
            self._log("⚠️ Whisper 字幕生成器不可用，跳过字幕生成")
        except Exception as e:
            self._log(f"⚠️ 字幕生成过程出错: {e}")
        
        self._log(f"字幕生成完成: 成功 {generated_count} 个")
        return new_pairs, generated_count
    
    def _align_data(
        self,
        pairs: List[TrainDataPair]
    ) -> List[AlignmentResult]:
        """对齐音频和字幕数据"""
        self._update_progress(
            phase=TrainPhase.ALIGNING_DATA,
            message="对齐音频和字幕数据...",
            overall=45.0,
            alignment_total=len(pairs),
            alignment_progress=0
        )
        self._log(f"开始对齐 {len(pairs)} 对数据...")
        
        results = []
        
        for i, pair in enumerate(pairs):
            if self._should_stop:
                break
            
            self._log(f"[{i+1}/{len(pairs)}] 对齐: {pair.audio_path.name}")
            
            # 为每对数据创建独立的输出目录
            pair_output_dir = self.config.output_path / "aligned" / f"pair_{i:04d}"
            
            result = self._aligner.align(
                audio_path=pair.audio_path,
                subtitle_path=pair.subtitle_path,
                output_dir=pair_output_dir
            )
            
            results.append(result)
            
            if result.success:
                self._log(f"  ✅ 对齐成功: {result.total_segments} 个片段, "
                         f"{result.total_duration:.1f} 秒")
            else:
                self._log(f"  ❌ 对齐失败: {result.error_message}")
            
            # 更新进度
            self._update_progress(
                alignment_progress=i + 1,
                overall=45.0 + (i + 1) / len(pairs) * 20.0
            )
        
        successful = sum(1 for r in results if r.success)
        self._log(f"数据对齐完成: {successful}/{len(pairs)} 成功")
        
        return results
    
    def _run_training(
        self,
        dataset_path: Path,
        resume_from: Optional[Path] = None
    ) -> TrainingResult:
        """执行训练"""
        self._update_progress(
            phase=TrainPhase.TRAINING,
            message="开始 Mimic3 训练...",
            overall=70.0
        )
        self._log("开始 Mimic3 训练...")
        self._log(self.config.get_summary())
        
        # 添加训练进度回调
        def on_training_progress(progress: TrainingProgress):
            self._progress.training_progress = progress
            # 训练阶段占 70% - 100% 的进度
            training_progress = progress.progress_percentage
            overall = 70.0 + training_progress * 0.3
            self._update_progress(
                overall=overall,
                message=f"训练中: Epoch {progress.current_epoch}/{progress.total_epochs}, "
                       f"Loss: {progress.loss:.4f}"
            )
        
        self._trainer.add_progress_callback(on_training_progress)
        
        try:
            result = self._trainer.train(dataset_path, resume_from)
            
            if result.success:
                self._update_progress(
                    phase=TrainPhase.COMPLETED,
                    message="训练完成",
                    overall=100.0
                )
                self._log("✅ 训练完成")
            else:
                self._update_progress(
                    phase=TrainPhase.FAILED,
                    message=f"训练失败: {result.error_message}"
                )
                self._log(f"❌ 训练失败: {result.error_message}")
            
            return result
            
        finally:
            self._trainer.remove_progress_callback(on_training_progress)
    
    def _create_cancelled_result(self) -> TrainControllerResult:
        """创建取消结果"""
        self._update_progress(
            phase=TrainPhase.FAILED,
            message="训练已取消"
        )
        result = TrainControllerResult()
        result.success = False
        result.error_message = "训练被用户取消"
        result.total_time = time.time() - self._progress.start_time
        result.logs = self._logs.copy()
        return result
    
    def stop(self) -> None:
        """停止训练"""
        self._log("收到停止请求...")
        self._should_stop = True
        self._trainer.stop()
    
    def get_progress(self) -> TrainControllerProgress:
        """获取当前进度"""
        return self._progress
    
    def get_logs(self) -> List[str]:
        """获取日志"""
        return self._logs.copy()


def create_train_controller(config: TrainConfig, verbose: bool = False) -> TrainController:
    """
    创建训练控制器的工厂函数
    
    Args:
        config: 训练配置
        verbose: 是否输出详细日志
        
    Returns:
        TrainController 实例
    """
    return TrainController(config, verbose)


def run_training(
    input_dir: Optional[Path] = None,
    output_path: Optional[Path] = None,
    speaker_name: str = "custom_speaker",
    epochs: int = 100,
    batch_size: int = 32,
    sample_rate: int = 22050,
    verbose: bool = False,
    **kwargs
) -> TrainControllerResult:
    """
    便捷函数：执行训练
    
    Args:
        input_dir: 输入数据目录
        output_path: 输出目录
        speaker_name: 说话人名称
        epochs: 训练轮数
        batch_size: 批处理大小
        sample_rate: 采样率
        verbose: 是否输出详细日志
        **kwargs: 其他训练配置参数
        
    Returns:
        TrainControllerResult: 训练结果
    """
    config = TrainConfig(
        input_dir=input_dir,
        output_path=output_path or Path("./models/custom_voice"),
        speaker_name=speaker_name,
        epochs=epochs,
        batch_size=batch_size,
        sample_rate=sample_rate,
        verbose=verbose,
        **kwargs
    )
    
    controller = TrainController(config, verbose)
    return controller.run()


if __name__ == "__main__":
    import sys
    
    # 配置日志
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 简单的命令行测试
    if len(sys.argv) < 2:
        print("用法: python train_controller.py <训练数据目录> [输出目录] [说话人名称]")
        print("\n示例:")
        print("  python train_controller.py ./training_data")
        print("  python train_controller.py ./training_data ./models/my_voice my_speaker")
        sys.exit(1)
    
    input_dir = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("./models/custom_voice")
    speaker_name = sys.argv[3] if len(sys.argv) > 3 else "custom_speaker"
    
    print(f"\n开始训练:")
    print(f"  输入目录: {input_dir}")
    print(f"  输出目录: {output_path}")
    print(f"  说话人: {speaker_name}")
    print()
    
    result = run_training(
        input_dir=input_dir,
        output_path=output_path,
        speaker_name=speaker_name,
        verbose=True
    )
    
    print(result.get_summary())
    
    sys.exit(0 if result.success else 1)
