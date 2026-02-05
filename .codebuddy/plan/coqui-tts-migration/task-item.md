# 实施计划

## Coqui TTS 训练模块迁移任务清单

本任务清单基于需求文档，将 Mimic3 训练模块迁移到 Coqui TTS 框架。

---

- [ ] 1. 更新项目依赖配置
   - 修改 `pyproject.toml`，将 `mimic-train` 替换为 Coqui TTS 相关依赖
   - 添加训练所需的 `TTS>=0.22.0`、`torch>=2.0.0`、`torchaudio>=2.0.0` 等依赖
   - 更新可选依赖组，确保训练依赖可单独安装
   - _需求：1.2, 1.3_

- [ ] 2. 实现 Coqui TTS 依赖检查器
   - 创建 `modules/training/coqui_checker.py`，替代 `mimic3_checker.py`
   - 实现 `CoquiChecker` 类，检测 Coqui TTS 是否已安装及版本兼容性
   - 提供清晰的安装指引和错误提示信息
   - 支持检测 GPU/CUDA 可用性
   - _需求：1.2, 1.3, 8.2, 8.4_

- [ ] 3. 实现 Coqui TTS 训练器核心模块
   - 创建 `modules/training/coqui_trainer.py`，替代 `mimic3_trainer.py`
   - 实现 `CoquiTrainer` 类，封装 Coqui TTS 训练 API
   - 支持 VITS、FastSpeech2、Tacotron2 等模型架构选择
   - 实现训练参数配置（epochs、batch_size、learning_rate、sample_rate 等）
   - 保持与现有 `TrainingProgress`、`TrainingResult`、`TrainingStatus` 数据结构兼容
   - _需求：1.1, 1.4, 1.5, 1.6_

- [ ] 4. 更新训练配置模块
   - 修改 `modules/training/train_config.py`，添加 Coqui TTS 特有配置项
   - 新增 `model_type` 参数（VITS/FastSpeech2/Tacotron2）
   - 添加 Coqui TTS 特定的音频配置（mel 参数、hop_size 等）
   - 保持与现有配置的向后兼容性
   - _需求：1.4, 1.5_

- [ ] 5. 更新数据预处理模块
   - 修改 `modules/training/train_data_manager.py`，支持生成 LJSpeech 格式数据集
   - 实现 `metadata.csv` 生成逻辑（Coqui TTS 所需格式）
   - 添加音频格式转换功能（自动转为指定采样率的 WAV 格式）
   - 实现数据集统计报告生成（总时长、片段数、词汇量等）
   - _需求：2.1, 2.2, 2.3, 2.4, 2.6_

- [ ] 6. 实现训练进度监控与回调机制
   - 在 `CoquiTrainer` 中实现 Coqui TTS 训练回调接口
   - 实时报告训练轮数、步数、损失值、学习率、已用时间、预估剩余时间
   - 实现训练中断和检查点保存功能
   - 支持从检查点恢复训练
   - _需求：3.1, 3.2, 3.3, 3.4, 3.5_

- [ ] 7. 实现 ONNX 模型导出模块
   - 创建 `modules/training/coqui_exporter.py`
   - 实现将 Coqui TTS 训练模型导出为 ONNX 格式的功能
   - 生成配套的音频配置、文本处理配置、模型元数据文件
   - 实现导出模型的有效性验证
   - 提供手动导出的命令行接口
   - _需求：4.1, 4.2, 4.3, 4.4, 4.5_

- [ ] 8. 更新训练控制器
   - 修改 `modules/training/train_controller.py`
   - 将 `Mimic3Trainer` 替换为 `CoquiTrainer`
   - 将 `Mimic3Checker` 替换为 `CoquiChecker`
   - 集成 ONNX 导出流程，训练完成后自动导出
   - 保持现有的训练流程阶段（检查依赖→扫描数据→匹配文件→生成字幕→对齐数据→训练→导出）
   - _需求：1.1, 4.1, 7.1_

- [ ] 9. 更新 CLI 命令行接口
   - 修改 `main.py` 中的训练相关参数
   - 新增 `--train-model-type` 参数，支持选择模型架构
   - 确保所有训练参数（`--train-input`、`--train-output`、`--train-epochs` 等）正常工作
   - 添加 `--verbose` 参数的详细日志输出
   - 实现适当的退出码（0 成功，非 0 失败）
   - _需求：5.1, 5.2, 5.3, 5.4, 5.5_

- [ ] 10. 更新 Web GUI 界面集成
   - 修改训练相关的 HTML 模板和 API 端点
   - 在训练配置表单中添加模型架构选择（VITS/FastSpeech2/Tacotron2）
   - 确保实时训练进度显示正常工作
   - 实现训练停止按钮功能
   - 添加训练结果摘要和模型下载链接
   - _需求：6.1, 6.2, 6.3, 6.4, 6.5, 6.6_

- [ ] 11. 实现与 ONNX TTS 推理系统的兼容集成
   - 确保导出的 ONNX 模型格式与现有推理引擎兼容
   - 实现训练完成后自动将模型注册到 ONNX TTS 模型列表
   - 生成与现有推理引擎兼容的配置文件（config.json）
   - 添加模型加载失败时的错误诊断信息
   - _需求：7.1, 7.2, 7.3, 7.4, 7.5_

- [ ] 12. 完善错误处理与日志记录
   - 在各模块中添加详细的异常捕获和错误日志
   - 实现 GPU 内存不足时自动回退到 CPU 模式
   - 添加数据验证失败时的具体问题文件列表和修复建议
   - 实现训练意外中断时的检查点恢复机制
   - 添加日志轮转功能
   - _需求：8.1, 8.2, 8.3, 8.4, 8.5, 8.6_

---

## 任务依赖关系

```
1 (依赖配置)
    ↓
2 (依赖检查器)
    ↓
3 (训练器核心) ← 4 (训练配置)
    ↓
5 (数据预处理)
    ↓
6 (进度监控)
    ↓
7 (ONNX导出)
    ↓
8 (训练控制器)
    ↓
┌───┴───┐
9       10
(CLI)   (Web GUI)
    ↓
11 (推理系统集成)
    ↓
12 (错误处理完善)
```

## 预估工作量

| 任务 | 预估时间 | 复杂度 |
|------|---------|--------|
| 1. 依赖配置 | 0.5h | 低 |
| 2. 依赖检查器 | 1h | 低 |
| 3. 训练器核心 | 4h | 高 |
| 4. 训练配置 | 1h | 低 |
| 5. 数据预处理 | 2h | 中 |
| 6. 进度监控 | 2h | 中 |
| 7. ONNX导出 | 3h | 高 |
| 8. 训练控制器 | 2h | 中 |
| 9. CLI接口 | 1h | 低 |
| 10. Web GUI | 2h | 中 |
| 11. 推理集成 | 2h | 中 |
| 12. 错误处理 | 1.5h | 低 |
| **总计** | **22h** | - |
