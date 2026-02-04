# ONNX TTS 模块

基于ONNX Runtime的文本到语音合成推理模块，支持命令行和Web图形界面两种使用方式。

## 功能特性

- ✅ 支持ONNX格式的语音合成模型
- ✅ 灵活的文本预处理和输入格式转换
- ✅ 高性能推理引擎，支持CPU和GPU加速
- ✅ 多种音频格式输出（WAV、MP3）
- ✅ 命令行接口，方便脚本集成
- ✅ Web图形界面，用户友好
- ✅ 实时进度反馈
- ✅ 完善的错误处理和日志记录

## 安装

### 1. 安装ONNX Runtime依赖

```bash
# 安装ONNX TTS模块及其依赖
pip install mytoolbox[onnx]

# 或手动安装依赖
pip install onnxruntime>=1.16.0 numpy>=1.24.0 scipy>=1.11.0
```

### 2. （可选）安装GPU支持

如果需要使用GPU加速：

```bash
# 安装CUDA版本的ONNX Runtime
pip install onnxruntime-gpu>=1.16.0
```

### 3. （可选）安装MP3支持

如果需要输出MP3格式，需要安装ffmpeg：

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Windows
# 从 https://ffmpeg.org/download.html 下载并安装
```

## 快速开始

### 命令行模式

#### 基本用法

```bash
# 使用ONNX模型合成语音
python cli_interface.py --onnx-tts \
    --onnx-model path/to/model.onnx \
    --text "你好，这是一个测试。" \
    --onnx-output output.wav
```

#### 从文件读取文本

```bash
python cli_interface.py --onnx-tts \
    --onnx-model path/to/model.onnx \
    --text-file input.txt \
    --onnx-output output.wav
```

#### 指定输出格式和采样率

```bash
python cli_interface.py --onnx-tts \
    --onnx-model path/to/model.onnx \
    --text "你好，世界！" \
    --onnx-format mp3 \
    --onnx-sample-rate 44100 \
    --onnx-output output.mp3
```

#### 使用配置文件

```bash
python cli_interface.py --onnx-tts \
    --onnx-model path/to/model.onnx \
    --text "测试文本" \
    --onnx-config config.yaml
```

#### 指定设备

```bash
# 使用CPU
python cli_interface.py --onnx-tts \
    --onnx-model path/to/model.onnx \
    --text "测试" \
    --onnx-device cpu

# 使用GPU（需要CUDA支持）
python cli_interface.py --onnx-tts \
    --onnx-model path/to/model.onnx \
    --text "测试" \
    --onnx-device cuda
```

### Web界面模式

启动Web服务器：

```bash
python cli_interface.py --gui --onnx-tts
```

然后在浏览器中访问 `http://localhost:5000`

#### Web界面功能

1. **上传模型**：拖拽或点击上传ONNX模型文件
2. **输入文本**：在文本框中输入待合成的文本
3. **设置选项**：选择输出格式和采样率
4. **生成语音**：点击按钮开始合成
5. **播放和下载**：在线播放或下载生成的音频

## 配置文件

创建一个YAML配置文件来自定义模块行为：

```yaml
# config.yaml
text_processing:
  max_length: 1000
  normalize_whitespace: true

audio_processing:
  sample_rate: 22050
  bit_depth: 16

model:
  device: "auto"
  input_format: "text"
  output_type: "waveform"
```

使用配置文件：

```bash
python cli_interface.py --onnx-tts \
    --onnx-model model.onnx \
    --text "测试" \
    --onnx-config config.yaml
```

## API使用

### Python API

```python
from modules.tts.onnx import (
    ONNXModelLoader,
    ONNXInferenceEngine,
    TextProcessor,
    AudioProcessor
)

# 1. 加载模型
model_loader = ONNXModelLoader(
    model_path="path/to/model.onnx",
    device="auto"
)

# 2. 初始化组件
text_processor = TextProcessor()
inference_engine = ONNXInferenceEngine(
    model_loader=model_loader,
    text_processor=text_processor
)
audio_processor = AudioProcessor(sample_rate=22050)

# 3. 执行推理
text = "你好，这是一个测试。"
output_dict = inference_engine.infer_text(text)

# 4. 处理音频
waveform = audio_processor.process_model_output(output_dict)

# 5. 保存音频
audio_processor.save_wav(waveform, "output.wav")
```

### 进度回调

```python
from modules.tts.onnx import InferenceProgress

def progress_callback(progress: InferenceProgress):
    print(f"[{progress.stage}] {progress.progress:.0f}% - {progress.message}")

inference_engine = ONNXInferenceEngine(
    model_loader=model_loader,
    text_processor=text_processor,
    progress_callback=progress_callback
)
```

## 命令行参数

### ONNX TTS模式参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--onnx-model` | ONNX模型文件路径 | 必需 |
| `--text` | 待合成的文本内容 | - |
| `--text-file` | 包含文本的文件路径 | - |
| `--onnx-output` | 输出音频文件路径 | 自动生成 |
| `--onnx-format` | 输出格式（wav/mp3） | wav |
| `--onnx-sample-rate` | 采样率 | 22050 |
| `--onnx-device` | 设备（auto/cpu/cuda） | auto |
| `--onnx-config` | 配置文件路径 | - |
| `--input-format` | 输入格式类型 | text |
| `--output-type` | 模型输出类型 | waveform |

## 故障排查

### 常见问题

#### 1. ONNX Runtime未安装

**错误信息**：
```
ONNX Runtime依赖未安装
```

**解决方案**：
```bash
pip install mytoolbox[onnx]
```

#### 2. 模型加载失败

**错误信息**：
```
无法加载ONNX模型
```

**可能原因**：
- 模型文件损坏
- ONNX Runtime版本不兼容
- 模型使用了不支持的算子

**解决方案**：
- 检查模型文件完整性
- 更新ONNX Runtime：`pip install --upgrade onnxruntime`
- 使用兼容的模型版本

#### 3. GPU不可用

**错误信息**：
```
未检测到CUDA支持，将使用CPU
```

**解决方案**：
- 安装CUDA工具包
- 安装GPU版本的ONNX Runtime：`pip install onnxruntime-gpu`

#### 4. MP3转换失败

**错误信息**：
```
ffmpeg未安装，无法保存MP3文件
```

**解决方案**：
安装ffmpeg（见安装章节）

#### 5. 内存不足

**错误信息**：
```
内存不足
```

**解决方案**：
- 使用更小的模型
- 减少文本长度
- 使用CPU模式

## 性能优化

### 1. 使用GPU加速

```bash
python cli_interface.py --onnx-tts \
    --onnx-model model.onnx \
    --text "测试" \
    --onnx-device cuda
```

### 2. 批量处理

对于多个文本，使用Python API进行批量处理：

```python
texts = ["文本1", "文本2", "文本3"]
results = inference_engine.infer_batch(texts)
```

### 3. 模型优化

- 使用量化模型减少内存占用
- 使用优化的ONNX模型（如使用onnx-simplifier）

## 开发指南

### 添加自定义文本处理器

```python
from modules.tts.onnx import TextProcessor

class CustomTextProcessor(TextProcessor):
    def preprocess(self, text: str) -> str:
        # 自定义预处理逻辑
        text = super().preprocess(text)
        # 添加额外处理
        return text
```

### 添加自定义音频后处理

```python
from modules.tts.onnx import AudioProcessor

class CustomAudioProcessor(AudioProcessor):
    def process_model_output(self, output_dict, output_type="waveform"):
        waveform = super().process_model_output(output_dict, output_type)
        # 添加自定义后处理
        return waveform
```

## 许可证

本模块遵循项目主许可证。

## 贡献

欢迎提交Issue和Pull Request！

## 更新日志

### v0.1.0 (2024-02-04)
- 初始版本发布
- 支持ONNX模型推理
- 命令行和Web界面
- WAV和MP3输出
- CPU和GPU支持
