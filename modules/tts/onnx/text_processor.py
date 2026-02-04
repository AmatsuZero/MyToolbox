"""
文本预处理和输入转换模块

负责文本验证、预处理和转换为模型所需的输入格式
"""

import re
import json
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
import logging

from .exceptions import TextProcessingError, ConfigurationError
from .logger import logger, handle_errors, log_operation

# 延迟导入numpy
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None


class TextProcessor:
    """
    文本预处理器
    
    负责文本验证、清理、分词和转换为模型输入格式
    """
    
    # 默认配置
    DEFAULT_CONFIG = {
        "max_length": 1000,  # 最大文本长度
        "min_length": 1,     # 最小文本长度
        "encoding": "utf-8",
        "normalize_whitespace": True,
        "remove_special_chars": False,
        "allowed_chars": None,  # None表示允许所有字符
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 config_file: Optional[str] = None):
        """
        初始化文本预处理器
        
        Args:
            config: 配置字典
            config_file: 配置文件路径（JSON或YAML）
            
        Raises:
            ConfigurationError: 配置加载失败
        """
        if not NUMPY_AVAILABLE:
            raise ImportError(
                "numpy未安装。请运行: pip install mytoolbox[onnx]"
            )
        
        # 加载配置
        self.config = self.DEFAULT_CONFIG.copy()
        
        if config_file:
            self._load_config_file(config_file)
        
        if config:
            self.config.update(config)
        
        # 初始化转换器
        self.converters: Dict[str, Callable] = {
            "text": self._convert_to_text,
            "char_ids": self._convert_to_char_ids,
            "phoneme_ids": self._convert_to_phoneme_ids,
        }
        
        logger.info(f"文本预处理器初始化完成，配置: {self.config}")
    
    @handle_errors(ConfigurationError, "配置文件加载失败")
    def _load_config_file(self, config_file: str):
        """
        从文件加载配置
        
        Args:
            config_file: 配置文件路径
        """
        config_path = Path(config_file)
        
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_file}")
        
        # 根据文件扩展名选择解析器
        if config_path.suffix.lower() in [".yaml", ".yml"]:
            with open(config_path, "r", encoding="utf-8") as f:
                file_config = yaml.safe_load(f)
        elif config_path.suffix.lower() == ".json":
            with open(config_path, "r", encoding="utf-8") as f:
                file_config = json.load(f)
        else:
            raise ConfigurationError(
                f"不支持的配置文件格式: {config_path.suffix}\n"
                f"支持的格式: .yaml, .yml, .json"
            )
        
        if file_config:
            self.config.update(file_config)
            logger.info(f"从 {config_file} 加载配置成功")
    
    @handle_errors(TextProcessingError, "文本验证失败")
    def validate_text(self, text: str) -> bool:
        """
        验证文本是否有效
        
        Args:
            text: 待验证的文本
            
        Returns:
            是否有效
            
        Raises:
            TextProcessingError: 文本无效
        """
        # 检查空值
        if not text or not text.strip():
            raise TextProcessingError(
                "文本内容为空\n"
                "请提供有效的文本内容"
            )
        
        # 检查长度
        text_length = len(text)
        if text_length < self.config["min_length"]:
            raise TextProcessingError(
                f"文本长度过短: {text_length} < {self.config['min_length']}"
            )
        
        if text_length > self.config["max_length"]:
            raise TextProcessingError(
                f"文本长度过长: {text_length} > {self.config['max_length']}\n"
                f"建议:\n"
                f"  • 将文本分段处理\n"
                f"  • 或增加max_length配置"
            )
        
        # 检查不支持的字符
        if self.config.get("allowed_chars"):
            allowed_chars = set(self.config["allowed_chars"])
            unsupported_chars = set(text) - allowed_chars
            if unsupported_chars:
                raise TextProcessingError(
                    f"文本包含不支持的字符: {unsupported_chars}\n"
                    f"请移除这些字符或更新allowed_chars配置"
                )
        
        return True
    
    @handle_errors(TextProcessingError, "文本预处理失败")
    @log_operation("预处理文本")
    def preprocess(self, text: str) -> str:
        """
        预处理文本
        
        Args:
            text: 原始文本
            
        Returns:
            预处理后的文本
        """
        # 验证文本
        self.validate_text(text)
        
        processed_text = text
        
        # 标准化空白字符
        if self.config.get("normalize_whitespace", True):
            processed_text = re.sub(r'\s+', ' ', processed_text)
            processed_text = processed_text.strip()
        
        # 移除特殊字符
        if self.config.get("remove_special_chars", False):
            processed_text = re.sub(r'[^\w\s]', '', processed_text)
        
        logger.debug(f"预处理完成: '{text[:50]}...' -> '{processed_text[:50]}...'")
        
        return processed_text
    
    def _convert_to_text(self, text: str) -> np.ndarray:
        """
        转换为文本数组（UTF-8编码）
        
        Args:
            text: 文本
            
        Returns:
            numpy数组
        """
        # 编码为字节
        encoded = text.encode(self.config.get("encoding", "utf-8"))
        return np.frombuffer(encoded, dtype=np.uint8)
    
    def _convert_to_char_ids(self, text: str) -> np.ndarray:
        """
        转换为字符ID序列
        
        Args:
            text: 文本
            
        Returns:
            字符ID数组
        """
        # 获取字符到ID的映射
        char_to_id = self.config.get("char_to_id", {})
        
        if not char_to_id:
            # 如果没有提供映射，使用Unicode码点
            char_ids = [ord(c) for c in text]
        else:
            # 使用提供的映射
            char_ids = []
            for c in text:
                if c in char_to_id:
                    char_ids.append(char_to_id[c])
                else:
                    # 未知字符使用特殊ID
                    unknown_id = char_to_id.get("<UNK>", 0)
                    char_ids.append(unknown_id)
                    logger.warning(f"未知字符: '{c}'，使用ID {unknown_id}")
        
        return np.array(char_ids, dtype=np.int64)
    
    def _convert_to_phoneme_ids(self, text: str) -> np.ndarray:
        """
        转换为音素ID序列
        
        Args:
            text: 文本
            
        Returns:
            音素ID数组
        """
        # 获取音素转换器
        phoneme_converter = self.config.get("phoneme_converter")
        
        if not phoneme_converter:
            raise TextProcessingError(
                "音素转换需要提供phoneme_converter配置\n"
                "请在配置文件中指定音素转换函数或映射表"
            )
        
        # 如果是字典映射
        if isinstance(phoneme_converter, dict):
            phoneme_ids = []
            for c in text:
                if c in phoneme_converter:
                    phoneme_ids.append(phoneme_converter[c])
                else:
                    unknown_id = phoneme_converter.get("<UNK>", 0)
                    phoneme_ids.append(unknown_id)
            return np.array(phoneme_ids, dtype=np.int64)
        
        # 如果是可调用函数
        elif callable(phoneme_converter):
            result = phoneme_converter(text)
            if isinstance(result, np.ndarray):
                return result
            return np.array(result, dtype=np.int64)
        
        else:
            raise TextProcessingError(
                f"不支持的phoneme_converter类型: {type(phoneme_converter)}"
            )
    
    @handle_errors(TextProcessingError, "文本转换失败")
    def convert(self, text: str, 
                format_type: str = "text",
                **kwargs) -> np.ndarray:
        """
        将文本转换为模型输入格式
        
        Args:
            text: 预处理后的文本
            format_type: 转换格式类型 ("text", "char_ids", "phoneme_ids")
            **kwargs: 额外参数
            
        Returns:
            转换后的numpy数组
            
        Raises:
            TextProcessingError: 转换失败
        """
        if format_type not in self.converters:
            raise TextProcessingError(
                f"不支持的转换格式: {format_type}\n"
                f"支持的格式: {list(self.converters.keys())}"
            )
        
        converter = self.converters[format_type]
        result = converter(text)
        
        logger.debug(
            f"文本转换完成: format={format_type}, "
            f"shape={result.shape}, dtype={result.dtype}"
        )
        
        return result
    
    def process(self, text: str, 
                format_type: str = "text",
                **kwargs) -> np.ndarray:
        """
        完整的文本处理流程：验证 -> 预处理 -> 转换
        
        Args:
            text: 原始文本
            format_type: 转换格式类型
            **kwargs: 额外参数
            
        Returns:
            处理后的numpy数组
        """
        # 预处理
        processed_text = self.preprocess(text)
        
        # 转换
        result = self.convert(processed_text, format_type, **kwargs)
        
        return result
    
    def split_long_text(self, text: str, 
                       max_length: Optional[int] = None) -> List[str]:
        """
        将长文本分段
        
        Args:
            text: 原始文本
            max_length: 最大长度，默认使用配置中的max_length
            
        Returns:
            文本段落列表
        """
        if max_length is None:
            max_length = self.config["max_length"]
        
        if len(text) <= max_length:
            return [text]
        
        # 按句子分割
        sentences = re.split(r'([。！？.!?])', text)
        
        segments = []
        current_segment = ""
        
        for i in range(0, len(sentences), 2):
            sentence = sentences[i]
            punctuation = sentences[i + 1] if i + 1 < len(sentences) else ""
            full_sentence = sentence + punctuation
            
            if len(current_segment) + len(full_sentence) <= max_length:
                current_segment += full_sentence
            else:
                if current_segment:
                    segments.append(current_segment)
                current_segment = full_sentence
        
        if current_segment:
            segments.append(current_segment)
        
        logger.info(f"长文本分段: {len(text)}字符 -> {len(segments)}段")
        
        return segments
    
    def __repr__(self) -> str:
        """字符串表示"""
        return f"TextProcessor(max_length={self.config['max_length']})"
