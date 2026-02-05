#!/usr/bin/env python3
"""
Coqui TTS ONNX 导出模块

将 Coqui TTS 训练的模型导出为 ONNX 格式，以便与现有的 ONNX TTS 推理系统集成。
"""

import json
import logging
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ExportResult:
    """导出结果"""
    success: bool = False
    onnx_model_path: Optional[Path] = None
    config_path: Optional[Path] = None
    model_size: int = 0
    error_message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "success": self.success,
            "onnx_model_path": str(self.onnx_model_path) if self.onnx_model_path else None,
            "config_path": str(self.config_path) if self.config_path else None,
            "model_size": self.model_size,
            "error_message": self.error_message
        }


class CoquiExporter:
    """
    Coqui TTS ONNX 导出器
    
    支持将 Coqui TTS 训练的 PyTorch 模型导出为 ONNX 格式，
    并生成与现有 ONNX TTS 推理系统兼容的配置文件。
    """
    
    def __init__(self, verbose: bool = False):
        """
        初始化导出器
        
        Args:
            verbose: 是否输出详细日志
        """
        self.verbose = verbose
    
    def _log(self, message: str) -> None:
        """记录日志"""
        if self.verbose:
            print(f"[CoquiExporter] {message}")
        logger.info(message)
    
    def export_to_onnx(
        self,
        model_path: Path,
        output_dir: Path,
        config_path: Optional[Path] = None,
        model_name: str = "tts_model",
        opset_version: int = 15
    ) -> ExportResult:
        """
        将 Coqui TTS 模型导出为 ONNX 格式
        
        Args:
            model_path: PyTorch 模型路径 (.pth 文件)
            output_dir: 输出目录
            config_path: 模型配置文件路径（可选）
            model_name: 输出模型名称
            opset_version: ONNX opset 版本
            
        Returns:
            ExportResult: 导出结果
        """
        result = ExportResult()
        
        try:
            self._log(f"开始导出 ONNX 模型: {model_path}")
            
            # 确保输出目录存在
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 检查模型文件是否存在
            if not model_path.exists():
                result.error_message = f"模型文件不存在: {model_path}"
                return result
            
            # 检查 Coqui TTS 是否可用
            if not self._is_coqui_available():
                self._log("Coqui TTS 未安装，使用模拟导出")
                return self._simulated_export(model_path, output_dir, model_name)
            
            # 执行真实导出
            result = self._real_export(
                model_path, output_dir, config_path, model_name, opset_version
            )
            
        except Exception as e:
            self._log(f"导出失败: {e}")
            import traceback
            self._log(traceback.format_exc())
            result.success = False
            result.error_message = str(e)
        
        return result
    
    def _is_coqui_available(self) -> bool:
        """检查 Coqui TTS 是否可用"""
        try:
            import TTS
            import torch
            return True
        except ImportError:
            return False
    
    def _real_export(
        self,
        model_path: Path,
        output_dir: Path,
        config_path: Optional[Path],
        model_name: str,
        opset_version: int
    ) -> ExportResult:
        """
        执行真实的 ONNX 导出
        
        Args:
            model_path: PyTorch 模型路径
            output_dir: 输出目录
            config_path: 配置文件路径
            model_name: 模型名称
            opset_version: ONNX opset 版本
            
        Returns:
            ExportResult: 导出结果
        """
        result = ExportResult()
        
        try:
            import torch
            import torch.onnx
            from TTS.utils.synthesizer import Synthesizer
            from TTS.tts.utils.synthesis import synthesis
            
            self._log("加载 Coqui TTS 模型...")
            
            # 确定配置文件路径
            if config_path is None:
                # 尝试在模型目录中查找配置文件
                model_dir = model_path.parent
                possible_configs = list(model_dir.glob("config.json"))
                if possible_configs:
                    config_path = possible_configs[0]
                    self._log(f"使用配置文件: {config_path}")
            
            if config_path is None or not config_path.exists():
                result.error_message = "未找到模型配置文件"
                return result
            
            # 加载配置
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            
            # 使用 Synthesizer 加载模型
            synthesizer = Synthesizer(
                tts_checkpoint=str(model_path),
                tts_config_path=str(config_path),
                use_cuda=False
            )
            
            model = synthesizer.tts_model
            model.eval()
            
            # 准备导出
            onnx_path = output_dir / f"{model_name}.onnx"
            
            self._log(f"导出 ONNX 模型到: {onnx_path}")
            
            # 创建虚拟输入
            # VITS 模型的输入: text_ids, text_lengths, (speaker_ids, language_ids)
            batch_size = 1
            max_text_len = 100
            
            dummy_text = torch.randint(0, 100, (batch_size, max_text_len), dtype=torch.long)
            dummy_text_lengths = torch.tensor([max_text_len], dtype=torch.long)
            
            input_names = ["text", "text_lengths"]
            output_names = ["audio"]
            dynamic_axes = {
                "text": {0: "batch_size", 1: "text_length"},
                "text_lengths": {0: "batch_size"},
                "audio": {0: "batch_size", 1: "audio_length"}
            }
            
            # 导出 ONNX
            try:
                with torch.no_grad():
                    torch.onnx.export(
                        model.inference,
                        (dummy_text, dummy_text_lengths),
                        str(onnx_path),
                        input_names=input_names,
                        output_names=output_names,
                        dynamic_axes=dynamic_axes,
                        opset_version=opset_version,
                        do_constant_folding=True,
                        verbose=self.verbose
                    )
            except Exception as export_error:
                # 尝试使用 Coqui TTS 内置的导出方法
                self._log(f"标准导出失败，尝试使用 TTS 内置导出: {export_error}")
                return self._export_with_tts_api(model_path, config_path, output_dir, model_name)
            
            # 验证 ONNX 模型
            if onnx_path.exists():
                self._log("验证 ONNX 模型...")
                is_valid, validate_msg = self._validate_onnx_model(onnx_path)
                
                if not is_valid:
                    self._log(f"ONNX 模型验证失败: {validate_msg}")
                    result.error_message = validate_msg
                    return result
            else:
                result.error_message = "ONNX 模型文件未生成"
                return result
            
            # 生成配置文件
            output_config_path = output_dir / f"{model_name}_config.json"
            self._generate_inference_config(config_dict, output_config_path)
            
            # 设置结果
            result.success = True
            result.onnx_model_path = onnx_path
            result.config_path = output_config_path
            result.model_size = onnx_path.stat().st_size
            
            self._log(f"ONNX 导出成功: {onnx_path} ({result.model_size / (1024*1024):.2f} MB)")
            
        except Exception as e:
            self._log(f"导出过程发生错误: {e}")
            import traceback
            self._log(traceback.format_exc())
            result.error_message = str(e)
        
        return result
    
    def _export_with_tts_api(
        self,
        model_path: Path,
        config_path: Path,
        output_dir: Path,
        model_name: str
    ) -> ExportResult:
        """
        使用 Coqui TTS 内置 API 导出 ONNX
        
        某些模型架构可能需要使用 TTS 内置的导出方法。
        
        Args:
            model_path: 模型路径
            config_path: 配置文件路径
            output_dir: 输出目录
            model_name: 模型名称
            
        Returns:
            ExportResult: 导出结果
        """
        result = ExportResult()
        
        try:
            from TTS.api import TTS
            
            self._log("使用 TTS API 进行导出...")
            
            # TTS 内置的 ONNX 导出（如果支持）
            # 注意：Coqui TTS 的某些版本可能不直接支持 ONNX 导出
            # 这里提供一个备选方案
            
            tts = TTS(
                model_path=str(model_path),
                config_path=str(config_path),
                gpu=False
            )
            
            onnx_path = output_dir / f"{model_name}.onnx"
            
            # 尝试调用导出方法
            if hasattr(tts, 'export_onnx'):
                tts.export_onnx(str(onnx_path))
                
                if onnx_path.exists():
                    result.success = True
                    result.onnx_model_path = onnx_path
                    result.model_size = onnx_path.stat().st_size
                    
                    # 生成配置
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config_dict = json.load(f)
                    output_config_path = output_dir / f"{model_name}_config.json"
                    self._generate_inference_config(config_dict, output_config_path)
                    result.config_path = output_config_path
            else:
                result.error_message = "当前 TTS 版本不支持 ONNX 导出"
            
        except Exception as e:
            self._log(f"TTS API 导出失败: {e}")
            result.error_message = str(e)
        
        return result
    
    def _simulated_export(
        self,
        model_path: Path,
        output_dir: Path,
        model_name: str
    ) -> ExportResult:
        """
        模拟导出（当 Coqui TTS 未安装时）
        
        Args:
            model_path: 模型路径
            output_dir: 输出目录
            model_name: 模型名称
            
        Returns:
            ExportResult: 导出结果
        """
        result = ExportResult()
        
        self._log("=" * 60)
        self._log("⚠️ 警告: Coqui TTS 未安装，无法导出 ONNX 模型！")
        self._log("=" * 60)
        self._log("模拟导出仅用于测试流程，不会生成有效的 ONNX 模型。")
        self._log("")
        self._log("如需导出真正的 ONNX 模型，请先安装 Coqui TTS:")
        self._log("  pip install TTS>=0.22.0")
        self._log("=" * 60)
        
        # 创建模拟的导出信息文件
        info_path = output_dir / f"{model_name}_export_info.json"
        
        export_info = {
            "warning": "⚠️ 这是模拟导出生成的信息文件，不是有效的 ONNX 模型！",
            "source_model": str(model_path),
            "solution": "请安装 Coqui TTS 后重新导出: pip install TTS>=0.22.0"
        }
        
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(export_info, f, indent=2, ensure_ascii=False)
        
        result.success = False
        result.error_message = "Coqui TTS 未安装，无法导出 ONNX 模型"
        result.config_path = info_path
        
        return result
    
    def _validate_onnx_model(self, onnx_path: Path) -> Tuple[bool, str]:
        """
        验证 ONNX 模型的有效性
        
        Args:
            onnx_path: ONNX 模型路径
            
        Returns:
            Tuple[bool, str]: (是否有效, 消息)
        """
        try:
            import onnx
            
            # 加载并检查模型
            model = onnx.load(str(onnx_path))
            onnx.checker.check_model(model)
            
            # 获取模型信息
            graph = model.graph
            inputs = [inp.name for inp in graph.input]
            outputs = [out.name for out in graph.output]
            
            self._log(f"ONNX 模型输入: {inputs}")
            self._log(f"ONNX 模型输出: {outputs}")
            
            return True, "模型验证通过"
            
        except ImportError:
            self._log("未安装 onnx 包，跳过验证")
            return True, "跳过验证（onnx 未安装）"
        except Exception as e:
            return False, str(e)
    
    def _generate_inference_config(
        self,
        train_config: Dict,
        output_path: Path
    ) -> None:
        """
        生成与 ONNX TTS 推理系统兼容的配置文件
        
        Args:
            train_config: 训练配置
            output_path: 输出路径
        """
        # 提取音频配置
        audio_config = train_config.get("audio", {})
        
        # 生成推理配置
        inference_config = {
            "model_type": train_config.get("model", "vits"),
            "sample_rate": audio_config.get("sample_rate", 22050),
            "hop_length": audio_config.get("hop_length", 256),
            "win_length": audio_config.get("win_length", 1024),
            "fft_size": audio_config.get("fft_size", 1024),
            "num_mels": audio_config.get("num_mels", 80),
            "mel_fmin": audio_config.get("mel_fmin", 0),
            "mel_fmax": audio_config.get("mel_fmax", None),
            
            # 文本处理配置
            "text_cleaner": train_config.get("text_cleaner", "basic_cleaners"),
            "use_phonemes": train_config.get("use_phonemes", False),
            "phoneme_language": train_config.get("phoneme_language", "zh-cn"),
            
            # 元数据
            "language": train_config.get("language", "zh-cn"),
            "speaker_name": train_config.get("speaker_name", "default"),
            "version": "1.0",
            "framework": "coqui-tts",
            
            # 推理参数
            "inference": {
                "noise_scale": 0.667,
                "noise_scale_w": 0.8,
                "length_scale": 1.0
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(inference_config, f, indent=2, ensure_ascii=False)
        
        self._log(f"推理配置已保存: {output_path}")
    
    def convert_checkpoint_to_inference(
        self,
        checkpoint_path: Path,
        output_dir: Path,
        model_name: str = "tts_model"
    ) -> ExportResult:
        """
        将训练检查点转换为推理模型
        
        Args:
            checkpoint_path: 检查点路径
            output_dir: 输出目录
            model_name: 模型名称
            
        Returns:
            ExportResult: 导出结果
        """
        result = ExportResult()
        
        try:
            self._log(f"转换检查点: {checkpoint_path}")
            
            if not checkpoint_path.exists():
                result.error_message = f"检查点不存在: {checkpoint_path}"
                return result
            
            # 查找对应的配置文件
            checkpoint_dir = checkpoint_path.parent
            config_files = list(checkpoint_dir.glob("config.json"))
            
            if not config_files:
                # 向上查找
                config_files = list(checkpoint_dir.parent.glob("config.json"))
            
            config_path = config_files[0] if config_files else None
            
            # 执行导出
            result = self.export_to_onnx(
                checkpoint_path,
                output_dir,
                config_path,
                model_name
            )
            
        except Exception as e:
            self._log(f"检查点转换失败: {e}")
            result.error_message = str(e)
        
        return result


def export_model(
    model_path: Path,
    output_dir: Path,
    config_path: Optional[Path] = None,
    model_name: str = "tts_model",
    verbose: bool = False
) -> ExportResult:
    """
    导出模型的便捷函数
    
    Args:
        model_path: 模型路径
        output_dir: 输出目录
        config_path: 配置文件路径
        model_name: 模型名称
        verbose: 是否输出详细日志
        
    Returns:
        ExportResult: 导出结果
    """
    exporter = CoquiExporter(verbose=verbose)
    return exporter.export_to_onnx(model_path, output_dir, config_path, model_name)


if __name__ == "__main__":
    import sys
    
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if len(sys.argv) < 3:
        print("用法: python coqui_exporter.py <模型路径> <输出目录> [配置文件路径]")
        print("示例: python coqui_exporter.py ./model/best_model.pth ./onnx_output")
        sys.exit(1)
    
    model_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    config_path = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    
    result = export_model(
        model_path,
        output_dir,
        config_path,
        verbose=True
    )
    
    print(f"\n导出结果:")
    print(f"  成功: {result.success}")
    if result.success:
        print(f"  ONNX 模型: {result.onnx_model_path}")
        print(f"  配置文件: {result.config_path}")
        print(f"  模型大小: {result.model_size / (1024*1024):.2f} MB")
    else:
        print(f"  错误: {result.error_message}")
