# 需求文档

## 引言

本项目旨在将现有的 Mimic3 语音训练模块迁移到 Coqui TTS 框架，以实现定制化语音模型的训练功能。由于 Mimic3 项目已停止维护且与 Python 3.12+ 存在兼容性问题，需要采用更活跃、功能更强大的 Coqui TTS 作为替代方案。

### 项目背景

- **现有系统**：MyToolbox 语音处理工具集，包含 Whisper 字幕提取、Mimic3 训练和 ONNX TTS 推理模块
- **迁移原因**：Mimic3 项目已不再维护，`mimic-train` 包与 Python 3.12+ 不兼容
- **目标框架**：Coqui TTS - 一个活跃维护的开源 TTS 框架，支持多种模型架构（VITS、Tacotron2、FastSpeech2 等）

### 技术约束

- Python 版本：>= 3.12
- 现有 ONNX 推理模块需保持兼容
- Web GUI 和 CLI 接口需保持一致性
- 训练结果需能导出为 ONNX 格式

---

## 需求

### 需求 1：Coqui TTS 训练模块核心实现

**用户故事：** 作为一名语音开发者，我希望使用 Coqui TTS 训练自定义语音模型，以便生成具有特定说话人风格的高质量语音合成模型。

#### 验收标准

1. WHEN 用户提供音频数据集和对应的文本标注 THEN 系统 SHALL 能够使用 Coqui TTS 训练框架进行模型训练
2. WHEN 训练开始 THEN 系统 SHALL 自动检测并验证 Coqui TTS 依赖是否已正确安装
3. IF Coqui TTS 未安装 THEN 系统 SHALL 提供清晰的安装指引和错误提示
4. WHEN 训练配置创建 THEN 系统 SHALL 支持以下模型架构选择：
   - VITS (推荐，支持端到端训练)
   - FastSpeech2
   - Tacotron2
5. WHEN 用户指定训练参数 THEN 系统 SHALL 支持以下可配置项：
   - 训练轮数 (epochs)
   - 批处理大小 (batch_size)
   - 学习率 (learning_rate)
   - 音频采样率 (sample_rate)
   - 说话人名称 (speaker_name)
   - 模型架构类型 (model_type)
6. WHEN 训练数据格式不符合要求 THEN 系统 SHALL 自动进行数据预处理和格式转换

---

### 需求 2：训练数据预处理与对齐

**用户故事：** 作为一名语音开发者，我希望系统能自动处理和对齐我的训练数据，以便无需手动进行复杂的数据准备工作。

#### 验收标准

1. WHEN 用户提供原始音频和字幕文件 THEN 系统 SHALL 自动将数据转换为 Coqui TTS 所需的 LJSpeech 格式
2. WHEN 音频文件格式不是 WAV THEN 系统 SHALL 自动转换为指定采样率的 WAV 格式
3. WHEN 字幕文件为 SRT/VTT 格式 THEN 系统 SHALL 解析时间戳并切割对应的音频片段
4. WHEN 数据集准备完成 THEN 系统 SHALL 生成以下文件结构：
   - `wavs/` 目录包含处理后的音频片段
   - `metadata.csv` 包含音频文件名和对应的文本内容
5. IF 输入目录只有音频文件无字幕 THEN 系统 SHALL 使用 Whisper 模块自动生成字幕
6. WHEN 数据预处理完成 THEN 系统 SHALL 生成数据集统计报告，包含：
   - 总音频时长
   - 音频片段数量
   - 平均片段时长
   - 文本词汇量统计

---

### 需求 3：训练进度监控与状态反馈

**用户故事：** 作为一名语音开发者，我希望实时监控训练进度和状态，以便了解训练情况并在需要时进行调整。

#### 验收标准

1. WHEN 训练进行中 THEN 系统 SHALL 实时报告以下信息：
   - 当前训练轮数 / 总轮数
   - 当前步数 / 总步数
   - 当前损失值 (loss)
   - 学习率
   - 已用时间
   - 预估剩余时间
2. WHEN 训练状态变化 THEN 系统 SHALL 触发进度回调通知
3. WHEN 用户请求停止训练 THEN 系统 SHALL 安全地中断训练并保存当前检查点
4. IF 训练被中断 THEN 系统 SHALL 支持从最近的检查点恢复训练
5. WHEN 训练完成或失败 THEN 系统 SHALL 生成详细的训练报告，包含：
   - 最终损失值
   - 总训练时间
   - 模型文件路径和大小
   - 训练日志摘要

---

### 需求 4：模型导出与 ONNX 转换

**用户故事：** 作为一名语音开发者，我希望将训练好的模型导出为 ONNX 格式，以便与现有的 ONNX TTS 推理系统无缝集成。

#### 验收标准

1. WHEN 训练成功完成 THEN 系统 SHALL 自动将模型导出为 ONNX 格式
2. WHEN 导出 ONNX 模型 THEN 系统 SHALL 同时生成以下配置文件：
   - 音频配置 (采样率、mel 参数等)
   - 文本处理配置 (字符集、音素配置等)
   - 模型元数据 (版本、说话人信息等)
3. WHEN ONNX 导出完成 THEN 系统 SHALL 验证导出模型的有效性
4. IF ONNX 导出失败 THEN 系统 SHALL 保留原始 PyTorch 模型并提供手动导出指引
5. WHEN 用户请求手动导出 THEN 系统 SHALL 提供独立的模型导出命令和 API

---

### 需求 5：CLI 命令行接口集成

**用户故事：** 作为一名用户，我希望通过命令行参数启动和控制训练流程，以便在服务器环境或自动化脚本中使用。

#### 验收标准

1. WHEN 用户执行 `python main.py --train` THEN 系统 SHALL 启动 Coqui TTS 训练模式
2. WHEN 用户指定训练参数 THEN 系统 SHALL 支持以下命令行参数：
   - `--train-input`：训练数据输入目录
   - `--train-output`：模型输出目录
   - `--train-speaker-name`：说话人名称
   - `--train-epochs`：训练轮数
   - `--train-batch-size`：批处理大小
   - `--train-sample-rate`：音频采样率
   - `--train-model-type`：模型架构类型 (新增)
   - `--train-config`：训练配置文件路径
3. WHEN 训练完成 THEN 系统 SHALL 返回适当的退出码 (0 表示成功，非 0 表示失败)
4. WHEN 使用 `--verbose` 参数 THEN 系统 SHALL 输出详细的训练日志
5. IF 命令行参数缺失必要信息 THEN 系统 SHALL 显示帮助信息和使用示例

---

### 需求 6：Web GUI 界面集成

**用户故事：** 作为一名用户，我希望通过 Web 界面直观地配置和监控训练过程，以便获得更好的用户体验。

#### 验收标准

1. WHEN 用户访问训练页面 THEN 系统 SHALL 显示训练配置表单，包含所有可配置参数
2. WHEN 训练进行中 THEN Web 界面 SHALL 实时显示训练进度和状态
3. WHEN 用户点击"停止训练"按钮 THEN 系统 SHALL 安全地中断当前训练
4. WHEN 训练完成 THEN Web 界面 SHALL 显示训练结果摘要和模型下载链接
5. IF 训练过程发生错误 THEN Web 界面 SHALL 显示详细的错误信息和解决建议
6. WHEN 用户上传训练数据 THEN Web 界面 SHALL 显示数据验证结果和预处理进度

---

### 需求 7：与现有 ONNX TTS 推理系统的兼容性

**用户故事：** 作为一名用户，我希望训练好的模型能够直接在现有的 ONNX TTS 推理系统中使用，以便无需额外配置即可进行语音合成。

#### 验收标准

1. WHEN 训练完成并导出 ONNX 模型 THEN 系统 SHALL 将模型自动注册到 ONNX TTS 模型列表
2. WHEN 用户在 ONNX TTS 界面选择训练好的模型 THEN 系统 SHALL 能够正常加载和使用该模型
3. WHEN ONNX 模型加载失败 THEN 系统 SHALL 提供详细的错误诊断信息
4. IF 模型配置与推理引擎不兼容 THEN 系统 SHALL 提供配置调整建议
5. WHEN 训练模型导出时 THEN 系统 SHALL 生成与现有推理引擎兼容的配置文件格式

---

### 需求 8：错误处理与日志记录

**用户故事：** 作为一名用户，我希望系统能够妥善处理各种错误情况并提供详细的日志，以便快速定位和解决问题。

#### 验收标准

1. WHEN 训练过程发生错误 THEN 系统 SHALL 记录详细的错误堆栈和上下文信息
2. WHEN 依赖检查失败 THEN 系统 SHALL 提供清晰的安装指引
3. WHEN 数据验证失败 THEN 系统 SHALL 列出具体的问题文件和修复建议
4. WHEN GPU 内存不足 THEN 系统 SHALL 自动回退到 CPU 模式或建议减小批处理大小
5. WHEN 训练被意外中断 THEN 系统 SHALL 能够从最近的检查点恢复
6. WHEN 日志文件超过指定大小 THEN 系统 SHALL 自动进行日志轮转

---

## 技术规格

### 依赖要求

```python
# pyproject.toml 新增依赖
training = [
    "TTS>=0.22.0",           # Coqui TTS 主包
    "torch>=2.0.0",          # PyTorch (训练必需)
    "torchaudio>=2.0.0",     # 音频处理
    "numpy>=1.24.0",         # 数值计算
    "scipy>=1.11.0",         # 科学计算
    "librosa>=0.10.0",       # 音频分析
    "soundfile>=0.12.0",     # 音频文件读写
]
```

### 文件结构

```
modules/training/
├── __init__.py
├── coqui_trainer.py         # Coqui TTS 训练器 (替代 mimic3_trainer.py)
├── coqui_checker.py         # Coqui TTS 依赖检查器
├── coqui_exporter.py        # ONNX 导出器
├── train_config.py          # 训练配置 (更新)
├── train_controller.py      # 训练控制器 (更新)
├── train_data_manager.py    # 数据管理器 (更新)
└── train_reporter.py        # 训练报告器
```

### 数据集格式

Coqui TTS 支持多种数据集格式，推荐使用 LJSpeech 格式：

```
dataset/
├── wavs/
│   ├── audio_001.wav
│   ├── audio_002.wav
│   └── ...
└── metadata.csv
```

`metadata.csv` 格式：
```
audio_001|文本内容1|文本内容1
audio_002|文本内容2|文本内容2
```

---

## 成功标准

1. ✅ 完整替代 Mimic3 训练功能，无功能缺失
2. ✅ 与 Python 3.12+ 完全兼容
3. ✅ 训练完成后能自动导出可用的 ONNX 模型
4. ✅ CLI 和 Web GUI 接口保持一致性
5. ✅ 现有 ONNX TTS 推理流程无需修改
6. ✅ 完善的错误处理和用户反馈机制

---

## 边界情况

1. **大规模数据集**：当数据集超过 100 小时时，需要支持分批加载和增量训练
2. **多说话人训练**：虽然当前需求为单说话人，但架构应预留多说话人扩展能力
3. **GPU 不可用**：支持 CPU 训练模式，但需提示用户训练时间会显著增加
4. **训练中断恢复**：检查点应包含完整的训练状态，确保恢复后训练效果一致
5. **磁盘空间不足**：在训练开始前检查可用空间，避免训练中途失败

---

## 风险评估

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| Coqui TTS 版本兼容性 | 中等 | 锁定测试通过的版本号 |
| ONNX 导出失败 | 高 | 提供手动导出工具和备用方案 |
| 训练资源消耗 | 中等 | 提供资源预估和低配置模式 |
| 数据隐私 | 低 | 所有处理本地完成，不上传外部服务器 |
