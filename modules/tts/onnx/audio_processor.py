"""
音频后处理和格式转换模块

负责将ONNX模型输出转换为音频文件，支持多种格式
"""

import wave
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, Union
import tempfile

from .exceptions import AudioProcessingError
from .logger import logger, handle_errors, log_operation

# 延迟导入
try:
    import numpy as np
    import scipy.signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    np = None
    scipy = None


class AudioProcessor:
    """
    音频后处理器
    
    负责将模型输出转换为音频文件，支持WAV、MP3等格式
    """
    
    # 默认音频参数
    DEFAULT_SAMPLE_RATE = 22050
    DEFAULT_BIT_DEPTH = 16
    
    def __init__(self, 
                 sample_rate: int = DEFAULT_SAMPLE_RATE,
                 bit_depth: int = DEFAULT_BIT_DEPTH):
        """
        初始化音频处理器
        
        Args:
            sample_rate: 采样率
            bit_depth: 位深度
        """
        if not SCIPY_AVAILABLE:
            raise ImportError(
                "scipy未安装。请运行: pip install mytoolbox[onnx]"
            )
        
        self.sample_rate = sample_rate
        self.bit_depth = bit_depth
        
        logger.info(
            f"音频处理器初始化: sample_rate={sample_rate}Hz, "
            f"bit_depth={bit_depth}bit"
        )
    
    @staticmethod
    def check_ffmpeg_available() -> bool:
        """
        检查ffmpeg是否可用
        
        Returns:
            是否可用
        """
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                text=True,
                check=False
            )
            return result.returncode == 0
        except FileNotFoundError:
            logger.warning("ffmpeg未安装，MP3转换功能不可用")
            return False
    
    @handle_errors(AudioProcessingError, "模型输出处理失败")
    def process_model_output(self, 
                            output_dict: Dict[str, np.ndarray],
                            output_type: str = "waveform") -> np.ndarray:
        """
        处理模型输出，转换为音频波形
        
        Args:
            output_dict: 模型输出字典
            output_type: 输出类型 ("waveform", "mel_spectrogram", "linear_spectrogram")
            
        Returns:
            音频波形数组
        """
        # 获取主要输出
        if not output_dict:
            raise AudioProcessingError("模型输出为空")
        
        # 尝试找到音频输出
        audio_output = None
        
        # 常见的输出节点名称
        possible_names = ["audio", "waveform", "output", "wav", "pcm"]
        
        for name in possible_names:
            if name in output_dict:
                audio_output = output_dict[name]
                break
        
        # 如果没找到，使用第一个输出
        if audio_output is None:
            output_name = list(output_dict.keys())[0]
            audio_output = output_dict[output_name]
            logger.warning(f"使用第一个输出节点: {output_name}")
        
        # 根据输出类型处理
        if output_type == "waveform":
            waveform = self._process_waveform(audio_output)
        elif output_type == "mel_spectrogram":
            waveform = self._mel_to_waveform(audio_output)
        elif output_type == "linear_spectrogram":
            waveform = self._spectrogram_to_waveform(audio_output)
        else:
            raise AudioProcessingError(
                f"不支持的输出类型: {output_type}\n"
                f"支持的类型: waveform, mel_spectrogram, linear_spectrogram"
            )
        
        return waveform
    
    def _process_waveform(self, waveform: np.ndarray) -> np.ndarray:
        """
        处理波形输出
        
        Args:
            waveform: 波形数组
            
        Returns:
            处理后的波形
        """
        # 移除batch维度
        if waveform.ndim > 1:
            waveform = waveform.squeeze()
        
        # 确保是1D数组
        if waveform.ndim != 1:
            raise AudioProcessingError(
                f"波形维度错误: {waveform.shape}\n"
                f"期望1D数组"
            )
        
        # 标准化到[-1, 1]范围
        if waveform.dtype != np.float32:
            waveform = waveform.astype(np.float32)
        
        max_val = np.abs(waveform).max()
        if max_val > 0:
            waveform = waveform / max_val
        
        logger.debug(f"波形处理完成: shape={waveform.shape}, dtype={waveform.dtype}")
        
        return waveform
    
    def _mel_to_waveform(self, mel_spectrogram: np.ndarray) -> np.ndarray:
        """
        将梅尔频谱转换为波形（需要声码器）
        
        Args:
            mel_spectrogram: 梅尔频谱
            
        Returns:
            波形数组
        """
        # 这里需要一个声码器模型（如Griffin-Lim或神经声码器）
        # 简化实现：使用Griffin-Lim算法
        logger.warning(
            "梅尔频谱转换使用Griffin-Lim算法，质量可能不佳\n"
            "建议使用专门的声码器模型"
        )
        
        # 移除batch维度
        if mel_spectrogram.ndim > 2:
            mel_spectrogram = mel_spectrogram.squeeze()
        
        # 使用Griffin-Lim算法（简化版本）
        # 实际应用中应该使用更好的声码器
        waveform = self._griffin_lim(mel_spectrogram)
        
        return waveform
    
    def _spectrogram_to_waveform(self, spectrogram: np.ndarray) -> np.ndarray:
        """
        将线性频谱转换为波形
        
        Args:
            spectrogram: 线性频谱
            
        Returns:
            波形数组
        """
        # 移除batch维度
        if spectrogram.ndim > 2:
            spectrogram = spectrogram.squeeze()
        
        # 使用Griffin-Lim算法
        waveform = self._griffin_lim(spectrogram)
        
        return waveform
    
    def _griffin_lim(self, spectrogram: np.ndarray, 
                    n_iter: int = 32) -> np.ndarray:
        """
        Griffin-Lim算法：从频谱重建波形
        
        Args:
            spectrogram: 频谱（幅度）
            n_iter: 迭代次数
            
        Returns:
            波形数组
        """
        # 简化的Griffin-Lim实现
        # 实际应用中建议使用librosa.griffinlim
        
        # 随机初始化相位
        angles = np.exp(2j * np.pi * np.random.rand(*spectrogram.shape))
        
        # 迭代重建
        for _ in range(n_iter):
            # 逆短时傅里叶变换
            waveform = self._istft(spectrogram * angles)
            # 短时傅里叶变换
            _, angles = self._stft(waveform)
        
        return waveform
    
    def _stft(self, waveform: np.ndarray) -> tuple:
        """简化的STFT实现"""
        # 这里应该使用scipy.signal.stft
        # 简化实现
        return np.abs(waveform), np.exp(1j * np.angle(waveform))
    
    def _istft(self, spectrogram: np.ndarray) -> np.ndarray:
        """简化的ISTFT实现"""
        # 这里应该使用scipy.signal.istft
        # 简化实现：返回展平的频谱
        return spectrogram.flatten()
    
    @handle_errors(AudioProcessingError, "音频重采样失败")
    def resample(self, waveform: np.ndarray, 
                orig_sr: int, 
                target_sr: int) -> np.ndarray:
        """
        重采样音频
        
        Args:
            waveform: 原始波形
            orig_sr: 原始采样率
            target_sr: 目标采样率
            
        Returns:
            重采样后的波形
        """
        if orig_sr == target_sr:
            return waveform
        
        logger.info(f"重采样: {orig_sr}Hz -> {target_sr}Hz")
        
        # 使用scipy进行重采样
        num_samples = int(len(waveform) * target_sr / orig_sr)
        resampled = scipy.signal.resample(waveform, num_samples)
        
        return resampled.astype(np.float32)
    
    @handle_errors(AudioProcessingError, "WAV文件保存失败")
    @log_operation("保存WAV文件")
    def save_wav(self, waveform: np.ndarray, 
                output_path: Union[str, Path],
                sample_rate: Optional[int] = None) -> Path:
        """
        保存为WAV文件
        
        Args:
            waveform: 音频波形
            output_path: 输出路径
            sample_rate: 采样率（默认使用初始化时的采样率）
            
        Returns:
            输出文件路径
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if sample_rate is None:
            sample_rate = self.sample_rate
        
        # 转换为整数格式
        if self.bit_depth == 16:
            audio_data = (waveform * 32767).astype(np.int16)
        elif self.bit_depth == 24:
            audio_data = (waveform * 8388607).astype(np.int32)
        elif self.bit_depth == 32:
            audio_data = (waveform * 2147483647).astype(np.int32)
        else:
            raise AudioProcessingError(f"不支持的位深度: {self.bit_depth}")
        
        # 写入WAV文件
        with wave.open(str(output_path), 'wb') as wav_file:
            wav_file.setnchannels(1)  # 单声道
            wav_file.setsampwidth(self.bit_depth // 8)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())
        
        file_size = output_path.stat().st_size
        duration = len(waveform) / sample_rate
        
        logger.info(
            f"WAV文件保存成功: {output_path.name}, "
            f"大小: {file_size/1024:.1f}KB, "
            f"时长: {duration:.2f}秒"
        )
        
        return output_path
    
    @handle_errors(AudioProcessingError, "MP3文件保存失败")
    @log_operation("保存MP3文件")
    def save_mp3(self, waveform: np.ndarray,
                output_path: Union[str, Path],
                sample_rate: Optional[int] = None,
                bitrate: str = "128k") -> Path:
        """
        保存为MP3文件（需要ffmpeg）
        
        Args:
            waveform: 音频波形
            output_path: 输出路径
            sample_rate: 采样率
            bitrate: 比特率
            
        Returns:
            输出文件路径
        """
        if not self.check_ffmpeg_available():
            raise AudioProcessingError(
                "ffmpeg未安装，无法保存MP3文件\n"
                "请安装ffmpeg: https://ffmpeg.org/download.html"
            )
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 先保存为临时WAV文件
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_wav_path = Path(tmp_file.name)
        
        try:
            self.save_wav(waveform, tmp_wav_path, sample_rate)
            
            # 使用ffmpeg转换为MP3
            cmd = [
                "ffmpeg",
                "-i", str(tmp_wav_path),
                "-codec:a", "libmp3lame",
                "-b:a", bitrate,
                "-y",  # 覆盖输出文件
                str(output_path)
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode != 0:
                raise AudioProcessingError(
                    f"ffmpeg转换失败: {result.stderr}"
                )
            
            file_size = output_path.stat().st_size
            logger.info(
                f"MP3文件保存成功: {output_path.name}, "
                f"大小: {file_size/1024:.1f}KB"
            )
            
            return output_path
            
        finally:
            # 清理临时文件
            if tmp_wav_path.exists():
                tmp_wav_path.unlink()
    
    def save_audio(self, waveform: np.ndarray,
                  output_path: Union[str, Path],
                  format: Optional[str] = None,
                  **kwargs) -> Path:
        """
        保存音频文件（自动检测格式）
        
        Args:
            waveform: 音频波形
            output_path: 输出路径
            format: 格式（如果为None则从文件扩展名推断）
            **kwargs: 额外参数
            
        Returns:
            输出文件路径
        """
        output_path = Path(output_path)
        
        # 推断格式
        if format is None:
            format = output_path.suffix.lower().lstrip('.')
        
        # 根据格式调用相应的保存方法
        if format == "wav":
            return self.save_wav(waveform, output_path, **kwargs)
        elif format == "mp3":
            return self.save_mp3(waveform, output_path, **kwargs)
        else:
            raise AudioProcessingError(
                f"不支持的音频格式: {format}\n"
                f"支持的格式: wav, mp3"
            )
    
    def __repr__(self) -> str:
        """字符串表示"""
        return (
            f"AudioProcessor(sample_rate={self.sample_rate}Hz, "
            f"bit_depth={self.bit_depth}bit)"
        )
