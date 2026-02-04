"""
Mimic3 依赖检查模块

该模块提供 Mimic3 TTS 工具的安装检测、版本兼容性验证和安装指引功能。
"""

import subprocess
import sys
import shutil
from dataclasses import dataclass
from typing import Optional, Tuple
from packaging import version


@dataclass
class Mimic3CheckResult:
    """Mimic3 检查结果"""
    is_installed: bool
    version: Optional[str] = None
    is_compatible: bool = False
    error_message: Optional[str] = None
    install_path: Optional[str] = None


class Mimic3Checker:
    """Mimic3 依赖检查器"""

    # 支持的 Mimic3 版本范围
    MIN_VERSION = "0.2.0"
    MAX_VERSION = "1.0.0"  # 不包含
    RECOMMENDED_VERSION = "0.2.4"

    # Mimic3 命令行工具名称
    MIMIC3_COMMANDS = ["mimic3", "mimic3-train"]

    def __init__(self):
        self._check_result: Optional[Mimic3CheckResult] = None

    def check(self, force_recheck: bool = False) -> Mimic3CheckResult:
        """
        检查 Mimic3 是否已安装及其兼容性

        Args:
            force_recheck: 是否强制重新检查

        Returns:
            Mimic3CheckResult: 检查结果
        """
        if self._check_result is not None and not force_recheck:
            return self._check_result

        # 检查是否安装
        is_installed, install_path = self._check_installation()

        if not is_installed:
            self._check_result = Mimic3CheckResult(
                is_installed=False,
                error_message="Mimic3 未安装。请使用以下命令安装：\n"
                              f"  pip install mycroft-mimic3-tts=={self.RECOMMENDED_VERSION}"
            )
            return self._check_result

        # 获取版本
        installed_version = self._get_version()

        if installed_version is None:
            self._check_result = Mimic3CheckResult(
                is_installed=True,
                install_path=install_path,
                is_compatible=False,
                error_message="无法获取 Mimic3 版本信息。请尝试重新安装。"
            )
            return self._check_result

        # 检查版本兼容性
        is_compatible, compat_message = self._check_version_compatibility(installed_version)

        self._check_result = Mimic3CheckResult(
            is_installed=True,
            version=installed_version,
            install_path=install_path,
            is_compatible=is_compatible,
            error_message=compat_message if not is_compatible else None
        )

        return self._check_result

    def _check_installation(self) -> Tuple[bool, Optional[str]]:
        """
        检查 Mimic3 是否已安装

        Returns:
            Tuple[bool, Optional[str]]: (是否安装, 安装路径)
        """
        # 首先尝试查找命令行工具
        for cmd in self.MIMIC3_COMMANDS:
            path = shutil.which(cmd)
            if path:
                return True, path

        # 尝试通过 Python 包检查
        try:
            import mimic3_tts
            return True, mimic3_tts.__file__
        except ImportError:
            pass

        # 尝试通过 pip 检查
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "show", "mycroft-mimic3-tts"],
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
        获取 Mimic3 版本

        Returns:
            Optional[str]: 版本字符串，获取失败返回 None
        """
        # 尝试从 Python 包获取版本
        try:
            import mimic3_tts
            if hasattr(mimic3_tts, "__version__"):
                return mimic3_tts.__version__
        except ImportError:
            pass

        # 尝试通过 pip 获取版本
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "show", "mycroft-mimic3-tts"],
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
                ["mimic3", "--version"],
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
                    f"Mimic3 版本 {installed_version} 过低。\n"
                    f"要求版本: >= {self.MIN_VERSION}, < {self.MAX_VERSION}\n"
                    f"建议升级到版本 {self.RECOMMENDED_VERSION}：\n"
                    f"  pip install --upgrade mycroft-mimic3-tts=={self.RECOMMENDED_VERSION}"
                )

            if ver >= max_ver:
                return False, (
                    f"Mimic3 版本 {installed_version} 过高，可能存在兼容性问题。\n"
                    f"要求版本: >= {self.MIN_VERSION}, < {self.MAX_VERSION}\n"
                    f"建议降级到版本 {self.RECOMMENDED_VERSION}：\n"
                    f"  pip install mycroft-mimic3-tts=={self.RECOMMENDED_VERSION}"
                )

            return True, None

        except Exception as e:
            return False, f"版本解析失败: {e}"

    def ensure_available(self) -> bool:
        """
        确保 Mimic3 可用，不可用时打印错误信息

        Returns:
            bool: Mimic3 是否可用
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
        """打印 Mimic3 安装指南"""
        guide = """
╔══════════════════════════════════════════════════════════════════╗
║                       Mimic3 安装指南                            ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  Mimic3 是 Mycroft AI 开发的开源神经网络语音合成系统。          ║
║                                                                  ║
║  安装步骤:                                                       ║
║  ─────────                                                       ║
║  1. 使用 uv 安装 (推荐):                                         ║
║     uv sync --extra tts                                          ║
║                                                                  ║
║  2. 使用 uv pip 安装:                                            ║
║     uv pip install mycroft-mimic3-tts==0.2.4                     ║
║                                                                  ║
║  3. 验证安装:                                                    ║
║     uv run mimic3 --version                                      ║
║                                                                  ║
║  系统要求:                                                       ║
║  ─────────                                                       ║
║  • Python >= 3.7                                                 ║
║  • 推荐使用虚拟环境                                              ║
║  • 足够的磁盘空间用于存储语音模型                                ║
║                                                                  ║
║  更多信息:                                                       ║
║  https://github.com/MycroftAI/mimic3                             ║
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
            return "Mimic3: 未安装 ❌"

        status_icon = "✅" if result.is_compatible else "⚠️"
        compat_text = "兼容" if result.is_compatible else "版本不兼容"

        return f"Mimic3: v{result.version} ({compat_text}) {status_icon}"


# 全局检查器实例
_checker: Optional[Mimic3Checker] = None


def get_checker() -> Mimic3Checker:
    """获取全局 Mimic3 检查器实例"""
    global _checker
    if _checker is None:
        _checker = Mimic3Checker()
    return _checker


def check_mimic3() -> Mimic3CheckResult:
    """
    便捷函数：检查 Mimic3 状态

    Returns:
        Mimic3CheckResult: 检查结果
    """
    return get_checker().check()


def ensure_mimic3_available() -> bool:
    """
    便捷函数：确保 Mimic3 可用

    Returns:
        bool: Mimic3 是否可用
    """
    return get_checker().ensure_available()


if __name__ == "__main__":
    # 测试检查功能
    checker = Mimic3Checker()
    result = checker.check()

    print(f"\n{checker.get_status_summary()}\n")

    if result.is_installed:
        print(f"安装路径: {result.install_path}")
        print(f"版本: {result.version}")
        print(f"兼容性: {'是' if result.is_compatible else '否'}")
        if result.error_message:
            print(f"问题: {result.error_message}")
    else:
        checker.print_install_guide()
