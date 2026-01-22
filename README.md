# 语音转文字工具 (Speech-to-Text Toolbox)

基于 OpenAI Whisper 的语音识别系统，支持多种音视频格式的语音转文字和字幕生成。

## 功能特性

### 🎯 核心功能
- **多格式支持**: 支持 MP3、WAV、FLAC、AAC、OGG 等音频格式
- **视频处理**: 支持从 MP4、WebM、MKV、AVI、MOV 等视频文件中提取音频
- **高精度识别**: 基于 OpenAI Whisper 模型，支持多种语言识别
- **批量处理**: 支持批量处理多个文件，提高效率
- **实时监控**: 实时监控处理进度和性能指标

### 📊 输出格式
- **字幕格式**: SRT、VTT、ASS、TXT、JSON、CSV
- **文本格式**: 纯文本、带时间戳文本、结构化数据
- **多语言支持**: 自动检测语言或指定目标语言

### ⚡ 性能优化
- **GPU 加速**: 支持 CUDA 和 Apple Silicon GPU 加速
- **内存优化**: 智能内存管理和垃圾回收
- **并行处理**: 多进程并行处理提高效率
- **缓存机制**: 模型和结果缓存避免重复计算

### 🔧 错误处理
- **智能重试**: 指数退避重试机制
- **错误日志**: 详细的错误记录和报告
- **性能监控**: 实时监控系统资源使用情况
- **优雅降级**: 在资源不足时自动调整策略

## 快速开始

### 安装依赖

```bash
# 使用 uv 包管理器安装依赖
uv sync

# 或使用 pip
pip install -r requirements.txt
```

### 基本使用

```bash
# 扫描 downloads 目录并处理所有音视频文件
python main.py --input downloads --output subtitles

# 指定模型大小和语言
python main.py --input downloads --model medium --language zh

# 批量处理并生成多种字幕格式
python main.py --input downloads --format all --batch-size 4
```

### 命令行参数

```bash
# 输入参数
--input, -i         输入文件或目录路径（必需）
--recursive         递归扫描子目录
--file-pattern      文件匹配模式（例如：*.mp4,*.wav）

# 模型参数
--model, -m         模型大小（tiny, base, small, medium, large，默认：base）
--language, -l       目标语言（默认：auto，自动检测）
--device            计算设备（auto, cpu, cuda, mps，默认：auto）

# 输出参数
--output, -o        输出目录路径（默认：output）
--format            输出字幕格式（srt, vtt, ass, txt, json, csv, all，默认：all）
--encoding          输出文件编码（默认：utf-8）

# 处理参数
--batch-size        批处理大小（默认：1）
--max-workers       最大工作进程数（默认：1）
--chunk-length      音频分块长度（秒，默认：30）
--temperature       采样温度（默认：0.0）

# 性能参数
--optimize-memory   启用内存优化
--enable-gpu        启用GPU加速
--max-memory        最大内存使用量（MB）
--timeout           处理超时时间（秒，默认：3600）

# 配置参数
--config            配置文件路径
--save-config       保存配置到文件
--verbose           详细输出模式
--quiet             静默模式
--dry-run           试运行模式（不实际处理文件）
```

## 项目结构

```
MyToolbox/
├── main.py                 # 主程序入口
├── cli_interface.py        # 命令行接口
├── config.py              # 配置管理
├── file_scanner.py         # 文件扫描器
├── format_converter.py     # 格式转换器
├── whisper_integration.py  # Whisper模型集成
├── subtitle_generator.py   # 字幕生成器
├── error_handler.py        # 错误处理器
├── pyproject.toml          # 项目配置
├── uv.lock                 # 依赖锁定文件
├── .gitignore             # Git忽略文件
├── README.md              # 项目说明
└── config/                # 配置文件目录
    ├── default_config.json # 默认配置
    └── user_config.json    # 用户配置
```

## 模块说明

### 1. 文件扫描器 (file_scanner.py)
- 扫描指定目录中的音视频文件
- 支持文件类型识别和大小过滤
- 提供文件统计信息和预览功能

### 2. 格式转换器 (format_converter.py)
- 音视频格式转换和预处理
- 音频提取和标准化处理
- 支持多种编解码器和参数配置

### 3. Whisper 集成 (whisper_integration.py)
- OpenAI Whisper 模型加载和管理
- 语音识别和转录功能
- 多语言支持和参数优化

### 4. 字幕生成器 (subtitle_generator.py)
- 多种字幕格式生成
- 时间戳对齐和格式转换
- 字幕样式和布局配置

### 5. 错误处理器 (error_handler.py)
- 异常处理和错误恢复
- 性能监控和资源管理
- 日志记录和报告生成

### 6. 配置管理 (config.py)
- 系统配置加载和保存
- 预设配置和自定义配置
- 配置验证和优化建议

## 配置说明

### 默认配置

系统提供多种预设配置，可根据需求选择：

- **fast**: 快速模式，使用 tiny 模型，适合短音频处理
- **accurate**: 精确模式，使用 medium 模型，适合高质量转录
- **balanced**: 平衡模式，使用 base 模型，兼顾速度和精度
- **low_memory**: 低内存模式，适合资源受限环境
- **high_quality**: 高质量模式，使用 large 模型，适合专业用途

### 自定义配置

创建自定义配置文件 `config/user_config.json`：

```json
{
  "project_name": "语音转文字工具",
  "version": "1.0.0",
  "file_scanner": {
    "scan_directory": "downloads",
    "recursive": true,
    "max_file_size_mb": 1024
  },
  "whisper": {
    "model_size": "base",
    "language": "auto",
    "device": "auto"
  },
  "subtitle_generator": {
    "output_directory": "subtitles",
    "supported_formats": ["srt", "vtt", "txt"]
  }
}
```

## 使用示例

### 示例 1：基本使用

```bash
# 处理单个音频文件
python main.py --input audio.wav --output subtitles

# 处理整个目录
python main.py --input downloads --output subtitles --recursive
```

### 示例 2：高级配置

```bash
# 使用高质量模型处理中文音频
python main.py --input audio.mp3 --model medium --language zh --format all

# 批量处理并启用GPU加速
python main.py --input videos --batch-size 4 --enable-gpu --max-workers 2
```

### 示例 3：性能优化

```bash
# 低内存环境使用
python main.py --input audio --model tiny --optimize-memory --max-memory 2048

# 高性能处理
python main.py --input large_files --model large --enable-gpu --batch-size 1
```

## 输出文件说明

处理完成后，系统会生成以下文件：

### 字幕文件
- `*.srt`: SubRip 字幕格式，兼容性最好
- `*.vtt`: WebVTT 格式，适合网页播放器
- `*.ass`: Advanced SubStation Alpha 格式，支持高级样式
- `*.txt`: 纯文本格式，便于阅读和编辑
- `*.json`: 结构化数据格式，便于程序处理
- `*.csv`: 表格格式，便于数据分析和导入

### 报告文件
- `processing_summary.json`: 处理摘要报告
- `error_log.json`: 错误日志记录
- `performance_report.json`: 性能分析报告

## 故障排除

### 常见问题

1. **模型加载失败**
   - 检查网络连接
   - 确保有足够的磁盘空间（模型文件较大）
   - 尝试使用较小的模型（tiny 或 base）

2. **内存不足**
   - 使用 `--optimize-memory` 参数
   - 减小批处理大小 `--batch-size 1`
   - 使用较小的模型 `--model tiny`

3. **处理速度慢**
   - 启用 GPU 加速 `--enable-gpu`
   - 增加工作进程数 `--max-workers 4`
   - 使用较小的模型 `--model base`

4. **文件格式不支持**
   - 检查文件是否损坏
   - 尝试使用其他格式
   - 使用格式转换工具预处理

### 性能优化建议

- **硬件要求**: 建议 8GB 以上内存，支持 CUDA 的 GPU
- **模型选择**: 根据需求平衡速度和精度
- **批处理**: 适当调整批处理大小提高效率
- **内存管理**: 定期清理缓存和临时文件

## 开发指南

### 扩展功能

系统采用模块化设计，便于功能扩展：

1. **添加新格式支持**: 在 `format_converter.py` 中添加新的编解码器
2. **自定义输出格式**: 在 `subtitle_generator.py` 中实现新的输出格式
3. **集成新模型**: 在 `whisper_integration.py` 中集成其他语音识别模型

### API 使用

系统提供 Python API，可在其他项目中调用：

```python
from main import SpeechToTextSystem
from config import ConfigManager

# 加载配置
config_manager = ConfigManager()
config = config_manager.load_config()

# 创建系统实例
system = SpeechToTextSystem(config)

# 处理文件
results = system.process_file("audio.wav")

# 获取结果
if results["success"]:
    print(f"转录文本: {results['transcription_result']['text']}")
    print(f"字幕文件: {results['subtitle_files']}")
```

## 许可证

本项目采用 MIT 许可证，详见 LICENSE 文件。

## 贡献指南

欢迎提交 Issue 和 Pull Request 来改进项目：

1. Fork 项目仓库
2. 创建功能分支
3. 提交代码变更
4. 创建 Pull Request

## 更新日志

### v1.0.0 (2026-01-22)
- 初始版本发布
- 支持多种音视频格式
- 集成 OpenAI Whisper 模型
- 提供多种字幕输出格式
- 实现批量处理和性能优化

## 联系方式

如有问题或建议，请通过以下方式联系：

- 项目仓库: https://github.com/jiangzhenhua/MyToolbox
- 邮箱: jiangzhenhua@example.com

---

**注意**: 本项目基于 OpenAI Whisper，请遵守相关使用条款和许可证要求。