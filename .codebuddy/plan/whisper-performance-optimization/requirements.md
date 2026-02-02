# Whisper 性能优化需求文档

## 引言

本文档旨在分析和规划 MyToolbox 项目中 Whisper 语音转文字功能的性能优化方案。用户在使用 `large` 模型转换约 1.5 小时的视频时遇到转换时间过长的问题。

**最终方案决策**：采用**双引擎架构**，在 macOS (Apple Silicon) 上优先使用 WhisperKit 获得最佳性能，在其他平台使用 faster-whisper 作为后备引擎。

---

## 当前项目实现状况分析

### 已实现的优化措施

| 优化措施 | 实现状态 | 说明 |
|---------|---------|------|
| 自动设备检测 | ✅ 已实现 | 支持 CPU、CUDA、MPS 自动选择 |
| MPS 后端支持 | ✅ 已实现 | 针对 Apple Silicon 的基础 MPS 支持 |
| Float64 兼容性修补 | ✅ 已实现 | 通过 monkey patch 解决 MPS 不支持 float64 的问题 |
| 语言特定优化 | ✅ 已实现 | 针对中文、英文、日文、韩文的参数优化 |

### 当前存在的问题

1. **使用原生 OpenAI Whisper**：性能较差，未使用高性能替代方案
2. **MPS 执行效率低**：为避免 float64 问题，在转录时将模型移回 CPU 执行
3. **未启用量化**：未使用 INT8/FP16 量化技术
4. **单引擎架构**：无法针对不同平台选择最优方案

---

## 双引擎架构方案

### 架构概述

```
┌─────────────────────────────────────────────────────────────┐
│                    WhisperEngine（抽象基类）                  │
│  - transcribe(audio_path, language, ...)                   │
│  - get_supported_models()                                   │
│  - get_device_info()                                        │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────────┐
│   WhisperKitEngine      │     │   FasterWhisperEngine       │
│   (macOS Apple Silicon) │     │   (跨平台后备方案)            │
├─────────────────────────┤     ├─────────────────────────────┤
│ - Core ML + ANE 加速    │     │ - CTranslate2 后端           │
│ - 最高性能（快 2-3x）   │     │ - INT8/FP16 量化             │
│ - 仅支持 macOS          │     │ - 支持 CPU/CUDA              │
└─────────────────────────┘     └─────────────────────────────┘
```

### 引擎选择策略

| 平台 | 硬件 | 优先引擎 | 后备引擎 |
|-----|------|---------|---------|
| macOS | Apple Silicon (M1/M2/M3) | WhisperKit | faster-whisper (CPU) |
| macOS | Intel | faster-whisper (CPU) | openai-whisper |
| Linux | CUDA GPU | faster-whisper (CUDA) | openai-whisper |
| Linux | CPU only | faster-whisper (CPU) | openai-whisper |
| Windows | CUDA GPU | faster-whisper (CUDA) | openai-whisper |

### 预期性能对比

| 场景（1.5h 视频 + large 模型） | 当前耗时 | WhisperKit | faster-whisper |
|------------------------------|---------|-----------|----------------|
| Apple Silicon (M2 Pro) | ~3-4 小时 | **15-25 分钟** | 30-45 分钟 |
| CUDA GPU (RTX 3080) | ~1-1.5 小时 | N/A | **15-20 分钟** |
| CPU only (Intel i7) | ~4-6 小时 | N/A | ~1-1.5 小时 |

---

## 综合需求规格

### 需求 1：实现引擎抽象层

**用户故事：** 作为一名开发者，我希望有一个统一的引擎抽象层，以便在不同平台上无缝切换转录引擎

#### 验收标准

1. WHEN 系统初始化 THEN 系统 SHALL 创建 `WhisperEngine` 抽象基类定义统一接口
2. IF 需要添加新引擎 THEN 系统 SHALL 只需继承基类并实现接口方法
3. WHEN 调用转录方法 THEN 系统 SHALL 返回统一格式的 `TranscriptionResult`
4. IF 引擎初始化失败 THEN 系统 SHALL 自动尝试下一个可用引擎

---

### 需求 2：集成 WhisperKit 引擎（macOS 优先方案）

**用户故事：** 作为一名 macOS 用户，我希望使用 WhisperKit 获得 Apple Silicon 上的最佳性能

#### 验收标准

1. WHEN 在 Apple Silicon Mac 上运行 THEN 系统 SHALL 优先尝试加载 WhisperKit 引擎
2. IF WhisperKit 可用 THEN 系统 SHALL 使用 Core ML 和 ANE（神经网络引擎）进行硬件加速
3. WHEN 使用 WhisperKit THEN 系统 SHALL 支持 tiny、base、small、medium、large 模型
4. IF WhisperKit 模型未下载 THEN 系统 SHALL 自动从 Hugging Face 下载对应模型
5. WHEN WhisperKit 初始化失败 THEN 系统 SHALL 记录错误日志并回退到 faster-whisper
6. IF 在非 Apple Silicon 设备上 THEN 系统 SHALL 跳过 WhisperKit 引擎

#### 技术实现方案

**方案 A：使用 whisperkit Python 绑定（推荐）**
```python
# 安装：pip install whisperkit
from whisperkit import WhisperKit

kit = WhisperKit(model="large-v3")
result = kit.transcribe("audio.mp3")
```

**方案 B：通过 subprocess 调用 Swift CLI**
```python
# 使用 whisperkit-cli（需预先安装）
import subprocess
result = subprocess.run(["whisperkit-cli", "transcribe", "--model", "large-v3", "audio.mp3"])
```

**推荐方案 A**：使用 `whisperkit` Python 包，保持代码一致性。

---

### 需求 3：集成 faster-whisper 引擎（跨平台后备方案）

**用户故事：** 作为一名跨平台用户，我希望使用 faster-whisper 获得良好的性能和广泛的兼容性

#### 验收标准

1. WHEN WhisperKit 不可用 THEN 系统 SHALL 使用 faster-whisper 作为后备引擎
2. IF 设备为 CUDA GPU THEN 系统 SHALL 使用 faster-whisper CUDA 后端 + FP16 量化
3. IF 设备为 CPU THEN 系统 SHALL 使用 faster-whisper CPU 后端 + INT8 量化
4. WHEN 初始化 faster-whisper THEN 系统 SHALL 设置最优线程数（物理核心数 - 2，最少 4）
5. IF faster-whisper 不可用 THEN 系统 SHALL 回退到 openai-whisper

---

### 需求 4：实现智能引擎选择器

**用户故事：** 作为一名用户，我希望系统自动选择最佳引擎，无需手动配置

#### 验收标准

1. WHEN 系统启动时 THEN 系统 SHALL 检测当前平台和硬件能力
2. IF 用户未指定引擎 THEN 系统 SHALL 按优先级自动选择最佳可用引擎
3. WHEN 用户指定 `--engine` 参数 THEN 系统 SHALL 使用用户指定的引擎
4. IF 指定引擎不可用 THEN 系统 SHALL 提示错误并列出可用引擎
5. WHEN 引擎选择完成 THEN 系统 SHALL 记录所选引擎和原因到日志

#### 引擎优先级

```python
ENGINE_PRIORITY = {
    "darwin_arm64": ["whisperkit", "faster-whisper", "openai-whisper"],
    "darwin_x86_64": ["faster-whisper", "openai-whisper"],
    "linux_cuda": ["faster-whisper", "openai-whisper"],
    "linux_cpu": ["faster-whisper", "openai-whisper"],
    "windows_cuda": ["faster-whisper", "openai-whisper"],
    "windows_cpu": ["faster-whisper", "openai-whisper"],
}
```

---

### 需求 5：实现量化策略配置

**用户故事：** 作为一名用户，我希望系统自动选择最佳量化配置，在性能和精度之间取得平衡

#### 验收标准

1. WHEN 使用 faster-whisper + CPU THEN 系统 SHALL 默认使用 INT8 量化
2. WHEN 使用 faster-whisper + CUDA THEN 系统 SHALL 默认使用 FP16 量化
3. IF 用户指定 `--compute-type` 参数 THEN 系统 SHALL 使用用户指定的量化类型
4. WHEN 量化类型不被支持 THEN 系统 SHALL 自动降级到兼容的量化类型

#### 支持的量化类型

| 量化类型 | CPU | CUDA | 说明 |
|---------|-----|------|------|
| int8 | ✅ | ✅ | 最小内存，最快速度 |
| int8_float16 | ❌ | ✅ | 平衡方案 |
| float16 | ❌ | ✅ | 高精度 |
| float32 | ✅ | ✅ | 最高精度，最慢 |

---

### 需求 6：实现转录进度反馈

**用户故事：** 作为一名用户，我希望能看到转录进度和预估剩余时间

#### 验收标准

1. WHEN 转录任务开始 THEN 系统 SHALL 显示音频总时长和使用的引擎
2. WHEN 转录进行中 THEN 系统 SHALL 每 5 秒更新一次进度百分比
3. IF 引擎支持回调 THEN 系统 SHALL 显示实时转录文本预览
4. WHEN 转录完成 THEN 系统 SHALL 显示总耗时和实时率（RTF）

---

### 需求 7：保持向后兼容性

**用户故事：** 作为一名现有用户，我希望升级后现有的调用方式仍然有效

#### 验收标准

1. WHEN 调用现有 `WhisperIntegration.transcribe_audio()` THEN 系统 SHALL 正常工作
2. IF 未安装任何优化引擎 THEN 系统 SHALL 回退到 openai-whisper
3. WHEN 使用旧版配置文件 THEN 系统 SHALL 兼容解析并使用默认引擎
4. IF 模型格式不兼容 THEN 系统 SHALL 自动下载正确格式的模型

---

## 依赖变更

### 新增依赖

```txt
# requirements.txt 新增
whisperkit>=0.1.0; sys_platform == "darwin" and platform_machine == "arm64"
faster-whisper>=1.0.0
```

### 可选依赖

```txt
# 高性能可选依赖
ctranslate2>=4.0.0  # faster-whisper 后端
```

---

## 技术限制和注意事项

1. **WhisperKit 平台限制**：仅支持 macOS 14+ 和 Apple Silicon
2. **模型格式差异**：
   - WhisperKit：使用 Core ML 格式模型（`.mlmodelc`）
   - faster-whisper：使用 CTranslate2 格式模型
   - openai-whisper：使用 PyTorch 格式模型
3. **首次运行**：首次使用新引擎时需要下载对应格式的模型，可能需要几分钟
4. **内存需求**：
   - WhisperKit large 模型：~2GB
   - faster-whisper large INT8：~3GB
   - openai-whisper large：~6GB

---

## 实施优先级

1. **P0（必须）**：实现引擎抽象层 + WhisperKit 集成 + faster-whisper 集成
2. **P1（重要）**：智能引擎选择器 + 量化策略配置
3. **P2（可选）**：进度反馈 + 向后兼容性完善

---

## 预期成果

完成双引擎架构后：

| 指标 | 当前 | 目标 |
|-----|-----|-----|
| 1.5h 视频转录时间（Apple Silicon） | 3-4 小时 | **15-25 分钟** |
| 1.5h 视频转录时间（CUDA） | 1-1.5 小时 | **15-20 分钟** |
| 内存占用（large 模型） | ~6GB | ~2-3GB |
| 支持平台 | macOS/Linux/Windows | macOS/Linux/Windows |
