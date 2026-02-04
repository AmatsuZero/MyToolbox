"""
ONNX模型加载器

负责加载、验证和管理ONNX模型
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging

from .exceptions import ModelLoadError, ModelValidationError, DependencyError
from .logger import logger, handle_errors, log_operation

# 延迟导入ONNX Runtime
try:
    import onnxruntime as ort
    import numpy as np
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    ort = None
    np = None


class ONNXModelLoader:
    """
    ONNX模型加载器
    
    负责加载ONNX模型文件，验证模型有效性，解析模型输入输出签名，
    并提供模型实例缓存功能。
    """
    
    def __init__(self, model_path: str, device: str = "auto"):
        """
        初始化模型加载器
        
        Args:
            model_path: ONNX模型文件路径
            device: 运行设备 ("auto", "cpu", "cuda")
            
        Raises:
            DependencyError: ONNX Runtime未安装
            ModelLoadError: 模型加载失败
        """
        if not ONNX_AVAILABLE:
            raise DependencyError(
                "ONNX Runtime未安装。请运行: pip install mytoolbox[onnx]"
            )
        
        self.model_path = Path(model_path)
        self.device = self._detect_device(device)
        self.session: Optional[ort.InferenceSession] = None
        self.input_metadata: Dict[str, Any] = {}
        self.output_metadata: Dict[str, Any] = {}
        
        # 加载模型
        self._load_model()
    
    def _detect_device(self, device: str) -> str:
        """
        检测并选择运行设备
        
        Args:
            device: 用户指定的设备
            
        Returns:
            实际使用的设备
        """
        if device == "auto":
            # 自动检测CUDA是否可用
            providers = ort.get_available_providers()
            if "CUDAExecutionProvider" in providers:
                logger.info("检测到CUDA支持，将使用GPU加速")
                return "cuda"
            else:
                logger.info("未检测到CUDA支持，将使用CPU")
                return "cpu"
        return device
    
    @handle_errors(ModelLoadError, "模型加载失败")
    @log_operation("加载ONNX模型")
    def _load_model(self):
        """
        加载ONNX模型文件
        
        Raises:
            ModelLoadError: 模型加载失败
        """
        # 验证文件存在
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"模型文件不存在: {self.model_path}\n"
                f"请检查路径是否正确"
            )
        
        # 验证文件格式
        if self.model_path.suffix.lower() != ".onnx":
            raise ModelValidationError(
                f"不支持的文件格式: {self.model_path.suffix}\n"
                f"请提供.onnx格式的模型文件"
            )
        
        # 检查文件大小（空文件或过小的文件可能无效）
        file_size = self.model_path.stat().st_size
        if file_size == 0:
            raise ModelValidationError(
                f"模型文件为空: {self.model_path}\n"
                f"请检查文件是否完整上传"
            )
        
        if file_size < 1024:  # 小于1KB的文件可能不是有效的ONNX模型
            logger.warning(f"模型文件非常小 ({file_size} 字节)，可能不是有效的ONNX模型")
        
        # 简单验证文件头（ONNX文件是protobuf格式）
        try:
            with open(self.model_path, 'rb') as f:
                header = f.read(8)
                # ONNX protobuf文件通常以特定字节开头
                # 如果文件以常见的非protobuf签名开头，则可能是错误的文件类型
                if header.startswith(b'PK'):  # ZIP文件签名
                    raise ModelValidationError(
                        f"文件似乎是ZIP压缩包而非ONNX模型\n"
                        f"如果这是压缩的模型，请先解压后再上传"
                    )
                if header.startswith(b'%PDF'):  # PDF文件签名
                    raise ModelValidationError(
                        f"文件似乎是PDF文档而非ONNX模型"
                    )
                if header.startswith(b'\x89PNG'):  # PNG文件签名
                    raise ModelValidationError(
                        f"文件似乎是PNG图片而非ONNX模型"
                    )
        except ModelValidationError:
            raise
        except Exception as e:
            logger.warning(f"文件头检查失败: {str(e)}")
        
        # 配置执行提供者
        providers = self._get_providers()
        
        # 创建推理会话
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        try:
            self.session = ort.InferenceSession(
                str(self.model_path),
                sess_options=sess_options,
                providers=providers
            )
            logger.info(f"模型加载成功: {self.model_path}")
            logger.info(f"使用设备: {self.device}")
            logger.info(f"执行提供者: {self.session.get_providers()}")
        except Exception as e:
            error_msg = str(e)
            
            # 根据错误类型提供更具体的诊断信息
            if "INVALID_PROTOBUF" in error_msg or "Protobuf parsing failed" in error_msg:
                raise ModelLoadError(
                    f"无法加载ONNX模型: 文件格式无效\n\n"
                    f"详细错误: {error_msg}\n\n"
                    f"可能的原因:\n"
                    f"  • 文件不是有效的ONNX模型（可能只是重命名为.onnx的其他文件）\n"
                    f"  • 文件在传输过程中损坏\n"
                    f"  • 文件下载不完整\n"
                    f"  • ONNX模型版本过高，当前ONNX Runtime不支持\n\n"
                    f"解决方案:\n"
                    f"  1. 确认文件是从可靠来源下载的有效ONNX模型\n"
                    f"  2. 尝试重新下载或导出模型\n"
                    f"  3. 使用 'python -c \"import onnx; onnx.checker.check_model(onnx.load(\\\"模型路径\\\"))\"' 验证模型"
                )
            elif "unsupported" in error_msg.lower() or "不支持" in error_msg:
                raise ModelLoadError(
                    f"无法加载ONNX模型: 模型包含不支持的算子\n\n"
                    f"详细错误: {error_msg}\n\n"
                    f"解决方案:\n"
                    f"  • 更新ONNX Runtime: pip install --upgrade onnxruntime\n"
                    f"  • 使用兼容版本重新导出模型"
                )
            else:
                raise ModelLoadError(
                    f"无法加载ONNX模型: {error_msg}\n\n"
                    f"可能的原因:\n"
                    f"  • 模型文件损坏\n"
                    f"  • ONNX Runtime版本不兼容\n"
                    f"  • 模型使用了不支持的算子\n"
                    f"  • 内存不足"
                )
        
        # 解析模型签名
        self._parse_model_signature()
    
    def _get_providers(self) -> List[str]:
        """
        获取执行提供者列表
        
        Returns:
            执行提供者列表
        """
        if self.device == "cuda":
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            return ["CPUExecutionProvider"]
    
    @handle_errors(ModelValidationError, "模型签名解析失败")
    def _parse_model_signature(self):
        """
        解析模型的输入输出签名
        
        提取输入输出节点的名称、形状和数据类型信息
        """
        # 解析输入
        for input_meta in self.session.get_inputs():
            self.input_metadata[input_meta.name] = {
                "name": input_meta.name,
                "shape": input_meta.shape,
                "dtype": input_meta.type,
            }
            logger.debug(
                f"输入节点: {input_meta.name}, "
                f"形状: {input_meta.shape}, "
                f"类型: {input_meta.type}"
            )
        
        # 解析输出
        for output_meta in self.session.get_outputs():
            self.output_metadata[output_meta.name] = {
                "name": output_meta.name,
                "shape": output_meta.shape,
                "dtype": output_meta.type,
            }
            logger.debug(
                f"输出节点: {output_meta.name}, "
                f"形状: {output_meta.shape}, "
                f"类型: {output_meta.type}"
            )
        
        logger.info(
            f"模型签名解析完成: "
            f"{len(self.input_metadata)}个输入, "
            f"{len(self.output_metadata)}个输出"
        )
    
    def get_input_names(self) -> List[str]:
        """
        获取模型输入节点名称列表
        
        Returns:
            输入节点名称列表
        """
        return list(self.input_metadata.keys())
    
    def get_output_names(self) -> List[str]:
        """
        获取模型输出节点名称列表
        
        Returns:
            输出节点名称列表
        """
        return list(self.output_metadata.keys())
    
    def get_input_info(self, name: Optional[str] = None) -> Dict[str, Any]:
        """
        获取输入节点信息
        
        Args:
            name: 节点名称，如果为None则返回所有输入信息
            
        Returns:
            输入节点信息字典
        """
        if name is None:
            return self.input_metadata
        return self.input_metadata.get(name, {})
    
    def get_output_info(self, name: Optional[str] = None) -> Dict[str, Any]:
        """
        获取输出节点信息
        
        Args:
            name: 节点名称，如果为None则返回所有输出信息
            
        Returns:
            输出节点信息字典
        """
        if name is None:
            return self.output_metadata
        return self.output_metadata.get(name, {})
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """
        验证输入数据是否符合模型要求
        
        Args:
            input_data: 输入数据字典
            
        Returns:
            是否有效
            
        Raises:
            ModelValidationError: 输入数据无效
        """
        # 检查所有必需的输入是否存在
        for input_name in self.input_metadata.keys():
            if input_name not in input_data:
                raise ModelValidationError(
                    f"缺少必需的输入: {input_name}\n"
                    f"模型需要的输入: {list(self.input_metadata.keys())}"
                )
        
        # 检查数据类型和形状（基本验证）
        for input_name, data in input_data.items():
            if input_name not in self.input_metadata:
                logger.warning(f"输入 {input_name} 不在模型签名中，将被忽略")
                continue
            
            expected_info = self.input_metadata[input_name]
            if hasattr(data, 'dtype'):
                logger.debug(
                    f"输入 {input_name}: "
                    f"形状={getattr(data, 'shape', 'unknown')}, "
                    f"类型={data.dtype}"
                )
        
        return True
    
    def __repr__(self) -> str:
        """字符串表示"""
        return (
            f"ONNXModelLoader(model={self.model_path.name}, "
            f"device={self.device}, "
            f"inputs={len(self.input_metadata)}, "
            f"outputs={len(self.output_metadata)})"
        )
