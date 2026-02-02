# Whisper 性能优化 - 实施计划

基于需求文档，本实施计划采用双引擎架构方案，在 macOS Apple Silicon 上优先使用 WhisperKit，其他平台使用 faster-whisper 作为后备引擎。

---

## 任务清单

- [ ] 1. 创建引擎抽象层和数据模型
   - 创建 `whisper_engines/` 目录结构
   - 定义 `WhisperEngine` 抽象基类，包含 `transcribe()`、`load_model()`、`get_device_info()` 等接口
   - 定义统一的 `EngineTranscriptionResult` 数据类，确保所有引擎返回格式一致
   - 实现引擎注册机制，支持动态加载可用引擎
   - _需求：1.1, 1.2, 1.3_

- [ ] 2. 实现 WhisperKit 引擎（macOS 优先方案）
   - 创建 `whisper_engines/whisperkit_engine.py`
   - 实现 Apple Silicon 检测逻辑（`platform.machine() == 'arm64'`）
   - 集成 whisperkit Python 包，实现 `transcribe()` 方法
   - 实现模型自动下载和缓存管理
   - 添加 Core ML / ANE 硬件加速配置
   - 实现初始化失败时的异常处理和日志记录
   - _需求：2.1, 2.2, 2.3, 2.4, 2.5, 2.6_

- [ ] 3. 实现 faster-whisper 引擎（跨平台后备方案）
   - 创建 `whisper_engines/faster_whisper_engine.py`
   - 集成 faster-whisper 包，实现 `transcribe()` 方法
   - 实现 CUDA/CPU 设备自动检测
   - 实现量化策略：CUDA 使用 FP16，CPU 使用 INT8
   - 实现最优线程数配置（物理核心数 - 2，最少 4）
   - 将转录结果转换为统一的 `EngineTranscriptionResult` 格式
   - _需求：3.1, 3.2, 3.3, 3.4, 3.5_

- [ ] 4. 实现 OpenAI Whisper 引擎（兼容后备方案）
   - 创建 `whisper_engines/openai_whisper_engine.py`
   - 将现有 `whisper_integration.py` 中的逻辑迁移到新引擎类
   - 保留现有的 MPS 兼容性修复代码
   - 实现统一接口适配
   - _需求：7.2_

- [ ] 5. 实现智能引擎选择器
   - 创建 `whisper_engines/engine_selector.py`
   - 实现平台检测逻辑（darwin_arm64、darwin_x86_64、linux_cuda、linux_cpu、windows_*）
   - 实现引擎优先级配置和自动选择逻辑
   - 实现引擎可用性检测（检查依赖包是否安装）
   - 实现引擎回退机制：首选引擎失败时自动尝试下一个
   - 添加 `--engine` 命令行参数支持
   - _需求：4.1, 4.2, 4.3, 4.4, 4.5_

- [ ] 6. 实现量化策略配置
   - 在 `FasterWhisperEngine` 中实现量化类型自动选择
   - 添加 `--compute-type` 命令行参数支持
   - 实现量化类型兼容性检测和自动降级
   - 支持 int8、int8_float16、float16、float32 四种量化类型
   - _需求：5.1, 5.2, 5.3, 5.4_

- [ ] 7. 重构 WhisperIntegration 类
   - 修改 `whisper_integration.py`，使用引擎选择器获取最佳引擎
   - 保持 `transcribe_audio()` 方法签名不变，确保向后兼容
   - 将引擎初始化逻辑委托给选择器
   - 添加引擎切换日志记录
   - _需求：7.1, 7.3_

- [ ] 8. 实现转录进度反馈
   - 定义进度回调接口 `ProgressCallback`
   - 在各引擎中实现进度回调支持
   - 实现转录开始时显示音频时长和引擎信息
   - 实现转录完成时显示总耗时和实时率（RTF）
   - _需求：6.1, 6.2, 6.3, 6.4_

- [ ] 9. 更新 CLI 和依赖配置
   - 在 `cli_interface.py` 中添加 `--engine` 和 `--compute-type` 参数
   - 更新 `requirements.txt`，添加条件依赖（whisperkit、faster-whisper）
   - 更新项目 README，说明新增的引擎选项和性能优化
   - _需求：4.3, 5.3_

- [ ] 10. 集成测试和验证
   - 编写引擎抽象层单元测试
   - 编写各引擎的集成测试（可用性检测、转录功能）
   - 在 Apple Silicon 设备上验证 WhisperKit 性能
   - 验证引擎回退机制正常工作
   - 验证现有功能的向后兼容性
   - _需求：1.4, 2.5, 3.5, 7.1, 7.2_

---

## 文件结构预览

```
MyToolbox/
├── whisper_engines/
│   ├── __init__.py              # 引擎模块入口
│   ├── base_engine.py           # 抽象基类定义
│   ├── engine_selector.py       # 智能引擎选择器
│   ├── whisperkit_engine.py     # WhisperKit 引擎
│   ├── faster_whisper_engine.py # faster-whisper 引擎
│   └── openai_whisper_engine.py # OpenAI Whisper 引擎
├── whisper_integration.py       # 重构后的集成层（保持接口兼容）
├── cli_interface.py             # 更新 CLI 参数
└── requirements.txt             # 更新依赖
```

---

## 任务依赖关系

```
任务1 (抽象层) 
    ├── 任务2 (WhisperKit)
    ├── 任务3 (faster-whisper)
    └── 任务4 (OpenAI Whisper)
            │
            ▼
        任务5 (引擎选择器)
            │
            ├── 任务6 (量化配置)
            └── 任务7 (重构集成层)
                    │
                    ├── 任务8 (进度反馈)
                    └── 任务9 (CLI更新)
                            │
                            ▼
                        任务10 (测试验证)
```

---

## 预计工作量

| 任务 | 预计耗时 | 优先级 |
|-----|---------|-------|
| 任务1: 引擎抽象层 | 2h | P0 |
| 任务2: WhisperKit 引擎 | 4h | P0 |
| 任务3: faster-whisper 引擎 | 3h | P0 |
| 任务4: OpenAI Whisper 引擎 | 2h | P0 |
| 任务5: 引擎选择器 | 2h | P1 |
| 任务6: 量化配置 | 1h | P1 |
| 任务7: 重构集成层 | 2h | P1 |
| 任务8: 进度反馈 | 2h | P2 |
| 任务9: CLI 更新 | 1h | P1 |
| 任务10: 测试验证 | 3h | P1 |
| **总计** | **~22h** | - |
