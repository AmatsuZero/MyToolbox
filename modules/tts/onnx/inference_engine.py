"""
ONNX推理引擎

负责执行ONNX模型推理，处理输入输出数据，提供进度跟踪
"""

import time
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass

from .model_loader import ONNXModelLoader
from .text_processor import TextProcessor
from .exceptions import InferenceError
from .logger import logger, handle_errors, log_operation

# 延迟导入
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None


@dataclass
class InferenceProgress:
    """推理进度信息"""
    stage: str  # 当前阶段
    progress: float  # 进度百分比 (0-100)
    message: str  # 状态消息
    current_segment: int = 0  # 当前段落
    total_segments: int = 1  # 总段落数


class ONNXInferenceEngine:
    """
    ONNX推理引擎
    
    负责执行模型推理，处理输入输出，支持进度跟踪和分段处理
    """
    
    def __init__(self, 
                 model_loader: ONNXModelLoader,
                 text_processor: Optional[TextProcessor] = None,
                 progress_callback: Optional[Callable[[InferenceProgress], None]] = None):
        """
        初始化推理引擎
        
        Args:
            model_loader: 模型加载器实例
            text_processor: 文本预处理器实例（可选）
            progress_callback: 进度回调函数（可选）
        """
        if not NUMPY_AVAILABLE:
            raise ImportError(
                "numpy未安装。请运行: pip install mytoolbox[onnx]"
            )
        
        self.model_loader = model_loader
        self.text_processor = text_processor or TextProcessor()
        self.progress_callback = progress_callback
        
        # 推理统计
        self.inference_count = 0
        self.total_inference_time = 0.0
        
        logger.info("ONNX推理引擎初始化完成")
    
    def _report_progress(self, stage: str, progress: float, 
                        message: str, **kwargs):
        """
        报告进度
        
        Args:
            stage: 当前阶段
            progress: 进度百分比
            message: 状态消息
            **kwargs: 额外参数
        """
        if self.progress_callback:
            progress_info = InferenceProgress(
                stage=stage,
                progress=progress,
                message=message,
                **kwargs
            )
            self.progress_callback(progress_info)
        
        logger.info(f"[{stage}] {progress:.1f}% - {message}")
    
    @handle_errors(InferenceError, "输入数据准备失败")
    def prepare_input(self, text: str, 
                     input_format: str = "text") -> Dict[str, np.ndarray]:
        """
        准备模型输入数据
        
        Args:
            text: 输入文本
            input_format: 输入格式类型
            
        Returns:
            模型输入字典
        """
        self._report_progress("prepare", 10, "准备输入数据...")
        
        # 处理文本
        processed_data = self.text_processor.process(text, input_format)
        
        # 获取模型输入节点名称
        input_names = self.model_loader.get_input_names()
        
        if not input_names:
            raise InferenceError("模型没有输入节点")
        
        # 构建输入字典
        # 简化版本：假设第一个输入是文本数据
        input_dict = {}
        
        # 主输入节点
        main_input_name = input_names[0]
        
        # 确保数据维度正确（添加batch维度）
        if processed_data.ndim == 1:
            processed_data = np.expand_dims(processed_data, axis=0)
        
        input_dict[main_input_name] = processed_data
        
        # 处理其他可能的输入（如长度信息）
        if len(input_names) > 1:
            # 如果有第二个输入，通常是序列长度
            length_input_name = input_names[1]
            sequence_length = np.array([processed_data.shape[1]], dtype=np.int64)
            input_dict[length_input_name] = sequence_length
        
        logger.debug(f"输入数据准备完成: {list(input_dict.keys())}")
        
        return input_dict
    
    @handle_errors(InferenceError, "模型推理失败")
    @log_operation("执行模型推理")
    def run_inference(self, input_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        执行模型推理
        
        Args:
            input_dict: 模型输入字典
            
        Returns:
            模型输出字典
        """
        self._report_progress("inference", 30, "执行模型推理...")
        
        # 验证输入
        self.model_loader.validate_input(input_dict)
        
        # 获取输出节点名称
        output_names = self.model_loader.get_output_names()
        
        # 执行推理
        start_time = time.time()
        
        try:
            outputs = self.model_loader.session.run(
                output_names,
                input_dict
            )
        except Exception as e:
            raise InferenceError(
                f"推理执行失败: {str(e)}\n"
                f"可能的原因:\n"
                f"  • 输入数据格式不匹配\n"
                f"  • 输入数据维度错误\n"
                f"  • 模型内部错误\n"
                f"  • 内存不足"
            )
        
        inference_time = time.time() - start_time
        
        # 更新统计
        self.inference_count += 1
        self.total_inference_time += inference_time
        
        # 构建输出字典
        output_dict = {
            name: output 
            for name, output in zip(output_names, outputs)
        }
        
        logger.info(
            f"推理完成: 耗时 {inference_time:.3f}秒, "
            f"输出节点: {list(output_dict.keys())}"
        )
        
        self._report_progress("inference", 70, f"推理完成 ({inference_time:.2f}秒)")
        
        return output_dict
    
    def infer_text(self, text: str, 
                   input_format: str = "text") -> Dict[str, np.ndarray]:
        """
        对单个文本执行完整的推理流程
        
        Args:
            text: 输入文本
            input_format: 输入格式类型
            
        Returns:
            模型输出字典
        """
        self._report_progress("start", 0, "开始推理...")
        
        # 准备输入
        input_dict = self.prepare_input(text, input_format)
        
        # 执行推理
        output_dict = self.run_inference(input_dict)
        
        self._report_progress("complete", 100, "推理完成")
        
        return output_dict
    
    def infer_batch(self, texts: List[str],
                   input_format: str = "text") -> List[Dict[str, np.ndarray]]:
        """
        批量推理多个文本
        
        Args:
            texts: 文本列表
            input_format: 输入格式类型
            
        Returns:
            输出字典列表
        """
        results = []
        total = len(texts)
        
        self._report_progress(
            "batch", 0, f"开始批量推理 (共{total}个文本)...",
            total_segments=total
        )
        
        for i, text in enumerate(texts):
            logger.info(f"处理文本 {i+1}/{total}")
            
            # 更新进度
            progress = (i / total) * 100
            self._report_progress(
                "batch", progress, f"处理文本 {i+1}/{total}",
                current_segment=i+1,
                total_segments=total
            )
            
            # 执行推理
            output = self.infer_text(text, input_format)
            results.append(output)
        
        self._report_progress(
            "batch", 100, f"批量推理完成 (共{total}个文本)",
            current_segment=total,
            total_segments=total
        )
        
        return results
    
    def infer_long_text(self, text: str,
                       input_format: str = "text",
                       max_length: Optional[int] = None) -> List[Dict[str, np.ndarray]]:
        """
        推理长文本（自动分段）
        
        Args:
            text: 长文本
            input_format: 输入格式类型
            max_length: 最大长度
            
        Returns:
            输出字典列表（每段一个）
        """
        # 分段
        segments = self.text_processor.split_long_text(text, max_length)
        
        if len(segments) == 1:
            # 不需要分段
            return [self.infer_text(text, input_format)]
        
        logger.info(f"长文本分为 {len(segments)} 段处理")
        
        # 批量处理
        return self.infer_batch(segments, input_format)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取推理统计信息
        
        Returns:
            统计信息字典
        """
        avg_time = (
            self.total_inference_time / self.inference_count 
            if self.inference_count > 0 
            else 0
        )
        
        return {
            "inference_count": self.inference_count,
            "total_time": self.total_inference_time,
            "average_time": avg_time,
            "device": self.model_loader.device,
        }
    
    def reset_statistics(self):
        """重置统计信息"""
        self.inference_count = 0
        self.total_inference_time = 0.0
        logger.info("推理统计已重置")
    
    def __repr__(self) -> str:
        """字符串表示"""
        stats = self.get_statistics()
        return (
            f"ONNXInferenceEngine("
            f"device={stats['device']}, "
            f"inferences={stats['inference_count']}, "
            f"avg_time={stats['average_time']:.3f}s)"
        )
