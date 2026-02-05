"""
Coqui TTS 依赖检查模块

该模块提供 Coqui TTS 工具的安装检测、版本兼容性验证、GPU/CUDA 检测和安装指引功能。
"""

import subprocess
import sys
import shutil
from dataclasses import dataclass
from typing import Optional, Tuple
from packaging import version


@dataclass
class CoquiCheckResult:
    """Coqui TTS 检查结果"""
    is_installed: bool
    version: Optional[str] = None
    is_compatible: bool = False
    error_message: Optional[str] = None
    install_path: Optional[str] = None
    # GPU/CUDA 相关
    cuda_available: bool = False
    cuda_version: Optional[str] = None
    gpu_name: Optional[str] = None
    gpu_memory: Optional[str] = None
    # MPS (Apple Metal) 相关
    mps_available: bool = False
    device_type: str = "cpu"  # "cuda", "mps", 或 "cpu"


class CoquiChecker:
    """Coqui TTS 依赖检查器"""

    # 支持的 Coqui TTS 版本范围
    MIN_VERSION = "0.17.0"
    MAX_VERSION = "1.0.0"  # 不包含
    RECOMMENDED_VERSION = "0.22.0"

    # Coqui TTS 命令行工具名称
    COQUI_COMMANDS = ["tts", "tts-server"]

    def __init__(self):
        self._check_result: Optional[CoquiCheckResult] = None

    def check(self, force_recheck: bool = False) -> CoquiCheckResult:
        """
        检查 Coqui TTS 是否已安装及其兼容性

        Args:
            force_recheck: 是否强制重新检查

        Returns:
            CoquiCheckResult: 检查结果
        """
        if self._check_result is not None and not force_recheck:
            return self._check_result

        # 检查是否安装
        is_installed, install_path = self._check_installation()

        if not is_installed:
            self._check_result = CoquiCheckResult(
                is_installed=False,
                error_message="Coqui TTS 未安装。请使用以下命令安装：\n"
                              f"  pip install TTS>={self.RECOMMENDED_VERSION}"
            )
            return self._check_result

        # 获取版本
        installed_version = self._get_version()

        if installed_version is None:
            self._check_result = CoquiCheckResult(
                is_installed=True,
                install_path=install_path,
                is_compatible=False,
                error_message="无法获取 Coqui TTS 版本信息。请尝试重新安装。"
            )
            return self._check_result

        # 检查版本兼容性
        is_compatible, compat_message = self._check_version_compatibility(installed_version)

        # 检查 GPU/CUDA/MPS
        cuda_available, cuda_version, gpu_name, gpu_memory = self._check_cuda()
        mps_available = self._check_mps()
        
        # 确定设备类型
        if cuda_available:
            device_type = "cuda"
        elif mps_available:
            device_type = "mps"
        else:
            device_type = "cpu"

        self._check_result = CoquiCheckResult(
            is_installed=True,
            version=installed_version,
            install_path=install_path,
            is_compatible=is_compatible,
            error_message=compat_message if not is_compatible else None,
            cuda_available=cuda_available,
            cuda_version=cuda_version,
            gpu_name=gpu_name,
            gpu_memory=gpu_memory,
            mps_available=mps_available,
            device_type=device_type
        )

        return self._check_result

    def _check_installation(self) -> Tuple[bool, Optional[str]]:
        """
        检查 Coqui TTS 是否已安装

        Returns:
            Tuple[bool, Optional[str]]: (是否安装, 安装路径)
        """
        # 首先尝试查找命令行工具
        for cmd in self.COQUI_COMMANDS:
            path = shutil.which(cmd)
            if path:
                return True, path

        # 尝试通过 Python 包检查
        try:
            import TTS
            return True, TTS.__file__
        except ImportError:
            pass

        # 尝试通过 pip 检查
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "show", "TTS"],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                # 解析安装位置
                for line in result.stdout.split("\n"):
                    if line.startswith("Location:"):
                        return True, line.split(":", 1)[1].strip()
                return True, None
        except (subprocess.TimeoutExpired, Exception):
            pass

        return False, None

    def _get_version(self) -> Optional[str]:
        """
        获取 Coqui TTS 版本

        Returns:
            Optional[str]: 版本字符串，获取失败返回 None
        """
        # 尝试从 Python 包获取版本
        try:
            import TTS
            if hasattr(TTS, "__version__"):
                return TTS.__version__
            # 尝试从 VERSION 属性获取
            if hasattr(TTS, "VERSION"):
                return TTS.VERSION
        except ImportError:
            pass

        # 尝试通过 pip 获取版本
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "show", "TTS"],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    if line.startswith("Version:"):
                        return line.split(":", 1)[1].strip()
        except (subprocess.TimeoutExpired, Exception):
            pass

        # 尝试通过命令行获取版本
        try:
            result = subprocess.run(
                ["tts", "--version"],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                # 解析版本输出
                output = result.stdout.strip()
                if output:
                    # 尝试提取版本号
                    parts = output.split()
                    for part in parts:
                        if part[0].isdigit():
                            return part
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            pass

        return None

    def _check_version_compatibility(self, installed_version: str) -> Tuple[bool, Optional[str]]:
        """
        检查版本兼容性

        Args:
            installed_version: 已安装的版本

        Returns:
            Tuple[bool, Optional[str]]: (是否兼容, 不兼容时的错误信息)
        """
        try:
            ver = version.parse(installed_version)
            min_ver = version.parse(self.MIN_VERSION)
            max_ver = version.parse(self.MAX_VERSION)

            if ver < min_ver:
                return False, (
                    f"Coqui TTS 版本 {installed_version} 过低。\n"
                    f"要求版本: >= {self.MIN_VERSION}, < {self.MAX_VERSION}\n"
                    f"建议升级到版本 {self.RECOMMENDED_VERSION}：\n"
                    f"  pip install --upgrade TTS>={self.RECOMMENDED_VERSION}"
                )

            if ver >= max_ver:
                return False, (
                    f"Coqui TTS 版本 {installed_version} 过高，可能存在兼容性问题。\n"
                    f"要求版本: >= {self.MIN_VERSION}, < {self.MAX_VERSION}\n"
                    f"建议降级到版本 {self.RECOMMENDED_VERSION}：\n"
                    f"  pip install TTS=={self.RECOMMENDED_VERSION}"
                )

            return True, None

        except Exception as e:
            return False, f"版本解析失败: {e}"

    def _check_cuda(self) -> Tuple[bool, Optional[str], Optional[str], Optional[str]]:
        """
        检查 CUDA/GPU 可用性

        Returns:
            Tuple[bool, Optional[str], Optional[str], Optional[str]]: 
                (CUDA可用, CUDA版本, GPU名称, GPU显存)
        """
        try:
            import torch
            
            if not torch.cuda.is_available():
                return False, None, None, None
            
            cuda_version = torch.version.cuda
            gpu_name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else None
            
            # 获取 GPU 显存
            if torch.cuda.device_count() > 0:
                total_memory = torch.cuda.get_device_properties(0).total_memory
                gpu_memory = f"{total_memory / (1024**3):.1f} GB"
            else:
                gpu_memory = None
            
            return True, cuda_version, gpu_name, gpu_memory
            
        except ImportError:
            return False, None, None, None
        except Exception:
            return False, None, None, None
    
    def _check_mps(self) -> bool:
        """
        检查 Apple Metal (MPS) 可用性
        
        MPS (Metal Performance Shaders) 是 Apple Silicon 的 GPU 加速后端，
        需要 macOS >= 12.3 和 PyTorch >= 1.12

        Returns:
            bool: MPS 是否可用
        """
        try:
            import torch
            
            # 检查 PyTorch 是否支持 MPS
            if not hasattr(torch.backends, 'mps'):
                return False
            
            # 检查 MPS 是否可用
            if not torch.backends.mps.is_available():
                return False
            
            # 检查 MPS 是否已构建到当前 PyTorch 中
            if not torch.backends.mps.is_built():
                return False
            
            return True
            
        except ImportError:
            return False
        except Exception:
            return False
    
    def _get_mps_info(self) -> Tuple[Optional[str], Optional[str]]:
        """
        获取 Apple Silicon GPU 信息
        
        Returns:
            Tuple[Optional[str], Optional[str]]: (GPU名称, 统一内存大小)
        """
        try:
            import platform
            import subprocess
            
            # 获取芯片名称
            chip_name = None
            try:
                result = subprocess.run(
                    ['sysctl', '-n', 'machdep.cpu.brand_string'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    chip_name = result.stdout.strip()
            except Exception:
                # 使用 platform 作为后备
                chip_name = platform.processor() or "Apple Silicon"
            
            # 获取统一内存大小
            memory_size = None
            try:
                result = subprocess.run(
                    ['sysctl', '-n', 'hw.memsize'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    mem_bytes = int(result.stdout.strip())
                    memory_size = f"{mem_bytes / (1024**3):.1f} GB (统一内存)"
            except Exception:
                pass
            
            return chip_name, memory_size
            
        except Exception:
            return None, None

    def ensure_available(self) -> bool:
        """
        确保 Coqui TTS 可用，不可用时打印错误信息

        Returns:
            bool: Coqui TTS 是否可用
        """
        result = self.check()

        if not result.is_installed:
            print(f"\n❌ 错误: {result.error_message}\n")
            self.print_install_guide()
            return False

        if not result.is_compatible:
            print(f"\n⚠️  警告: {result.error_message}\n")
            return False

        return True

    def print_install_guide(self):
        """打印 Coqui TTS 安装指南"""
        guide = """
╔══════════════════════════════════════════════════════════════════╗
║                     Coqui TTS 安装指南                            ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  Coqui TTS 是一个深度学习语音合成框架，支持多种模型架构。        ║
║                                                                  ║
║  安装步骤:                                                       ║
║  ─────────                                                       ║
║  1. 使用 uv 安装 (推荐):                                         ║
║     uv sync --extra training                                     ║
║                                                                  ║
║  2. 使用 uv pip 安装:                                            ║
║     uv pip install TTS>=0.22.0                                   ║
║                                                                  ║
║  3. 验证安装:                                                    ║
║     uv run tts --list_models                                     ║
║                                                                  ║
║  GPU 训练要求 (推荐):                                            ║
║  ─────────────────                                               ║
║  • NVIDIA GPU (8GB+ 显存推荐)                                    ║
║  • CUDA >= 11.7                                                  ║
║  • cuDNN >= 8.5                                                  ║
║                                                                  ║
║  Apple Silicon (M1/M2/M3/M4):                                    ║
║  ───────────────────────────                                     ║
║  • 支持 Metal (MPS) GPU 加速                                     ║
║  • 需要 macOS >= 12.3 和 PyTorch >= 1.12                         ║
║  • 部分操作可能回退到 CPU                                        ║
║                                                                  ║
║  CPU 训练:                                                       ║
║  ─────────                                                       ║
║  • 支持但训练速度较慢                                            ║
║  • 建议减小 batch_size                                           ║
║                                                                  ║
║  系统要求:                                                       ║
║  ─────────                                                       ║
║  • Python >= 3.9                                                 ║
║  • 推荐使用虚拟环境                                              ║
║  • 足够的磁盘空间 (模型约 500MB - 2GB)                           ║
║                                                                  ║
║  更多信息:                                                       ║
║  https://github.com/coqui-ai/TTS                                 ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""
        print(guide)

    def get_status_summary(self) -> str:
        """
        获取状态摘要信息

        Returns:
            str: 状态摘要字符串
        """
        result = self.check()

        if not result.is_installed:
            return "Coqui TTS: 未安装 ❌"

        status_icon = "✅" if result.is_compatible else "⚠️"
        compat_text = "兼容" if result.is_compatible else "版本不兼容"

        summary = f"Coqui TTS: v{result.version} ({compat_text}) {status_icon}"
        
        if result.cuda_available:
            summary += f"\n  GPU: {result.gpu_name} ({result.gpu_memory}), CUDA {result.cuda_version} ✅"
        elif result.mps_available:
            mps_name, mps_memory = self._get_mps_info()
            gpu_info = mps_name or "Apple Silicon GPU"
            if mps_memory:
                summary += f"\n  GPU: {gpu_info} ({mps_memory}), Metal (MPS) ✅"
            else:
                summary += f"\n  GPU: {gpu_info}, Metal (MPS) ✅"
        else:
            summary += "\n  GPU: 不可用 (将使用 CPU 训练) ⚠️"
        
        return summary

    def get_device(self) -> str:
        """
        获取推荐的训练设备

        Returns:
            str: "cuda", "mps" 或 "cpu"
        """
        result = self.check()
        return result.device_type


# 全局检查器实例
_checker: Optional[CoquiChecker] = None


def get_checker() -> CoquiChecker:
    """获取全局 Coqui TTS 检查器实例"""
    global _checker
    if _checker is None:
        _checker = CoquiChecker()
    return _checker


def check_coqui() -> CoquiCheckResult:
    """
    便捷函数：检查 Coqui TTS 状态

    Returns:
        CoquiCheckResult: 检查结果
    """
    return get_checker().check()


def ensure_coqui_available() -> bool:
    """
    便捷函数：确保 Coqui TTS 可用

    Returns:
        bool: Coqui TTS 是否可用
    """
    return get_checker().ensure_available()


def get_training_device() -> str:
    """
    便捷函数：获取训练设备

    Returns:
        str: "cuda", "mps" 或 "cpu"
    """
    return get_checker().get_device()


if __name__ == "__main__":
    # 测试检查功能
    checker = CoquiChecker()
    result = checker.check()

    print(f"\n{checker.get_status_summary()}\n")

    if result.is_installed:
        print(f"安装路径: {result.install_path}")
        print(f"版本: {result.version}")
        print(f"兼容性: {'是' if result.is_compatible else '否'}")
        if result.error_message:
            print(f"问题: {result.error_message}")
        print(f"\n推荐训练设备: {checker.get_device()}")
    else:
        checker.print_install_guide()
