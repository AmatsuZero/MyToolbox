# 需求文档：GUI Web 界面支持

## 引言

本功能旨在为现有的语音处理命令行工具（MyToolbox）添加基于 Web 的图形用户界面（GUI）支持。该项目当前支持两种命令行模式：

1. **Whisper 字幕提取模式**：使用 OpenAI Whisper 从音视频文件中提取字幕
2. **Mimic3 语音训练模式**：训练自定义语音模型

通过新增 `--gui` 参数，用户可以选择启动一个基于 Flask 的 Web 服务，提供直观的网页界面来操作这些功能。Web 界面将支持文件拖拽上传、实时进度显示，并保持与现有命令行功能的完全兼容。

### 技术背景

- **现有框架**：Python 3.12+, argparse 命令行解析, 模块化架构
- **目标框架**：Flask Web 框架
- **现有功能模块**：
  - `core/cli/cli_interface.py`：命令行接口
  - `modules/whisper/`：Whisper 语音识别模块
  - `modules/training/`：语音训练模块

---

## 需求

### 需求 1：命令行参数扩展

**用户故事：** 作为一名用户，我希望通过添加 `--gui` 参数来启动 Web 界面，以便在不修改现有命令行使用习惯的情况下获得图形化操作体验。

#### 验收标准

1. WHEN 用户执行 `python main.py --gui` THEN 系统 SHALL 启动 Flask Web 服务并打开主页面（homepage）
2. WHEN 用户执行 `python main.py --whisper --gui` THEN 系统 SHALL 启动 Flask Web 服务并直接跳转到 Whisper 字幕提取功能页面
3. WHEN 用户执行 `python main.py --train --gui` THEN 系统 SHALL 启动 Flask Web 服务并直接跳转到语音训练功能页面
4. WHEN `--gui` 参数未指定 THEN 系统 SHALL 保持现有的命令行交互行为不变
5. WHEN Flask 服务启动成功 THEN 系统 SHALL 在控制台输出服务访问地址（默认 http://localhost:5000）
6. IF 用户指定了端口参数 `--port <port>` THEN 系统 SHALL 使用指定端口启动 Web 服务

---

### 需求 2：Web 服务架构

**用户故事：** 作为一名开发者，我希望 Web 服务采用模块化架构设计，以便后续维护和功能扩展。

#### 验收标准

1. WHEN Web 服务初始化时 THEN 系统 SHALL 创建独立的 Flask 应用实例，与现有业务逻辑解耦
2. WHEN 系统启动 Web 模式 THEN 系统 SHALL 加载所有必要的路由和静态资源
3. WHEN 处理 API 请求 THEN 系统 SHALL 复用现有的 `SpeechToTextSystem` 和 `TrainController` 等核心业务类
4. IF 后端处理发生错误 THEN 系统 SHALL 返回结构化的 JSON 错误响应，包含错误代码和描述
5. WHEN 系统提供 API 接口 THEN 系统 SHALL 遵循 RESTful 设计原则
6. WHEN Web 服务运行时 THEN 系统 SHALL 支持 CORS 跨域请求（可配置）

---

### 需求 3：主页面（Homepage）设计

**用户故事：** 作为一名用户，我希望主页面提供清晰的功能导航入口，以便快速选择所需功能。

#### 验收标准

1. WHEN 用户访问主页面 THEN 系统 SHALL 显示两个主要功能入口卡片：「字幕提取」和「语音训练」
2. WHEN 用户点击功能入口卡片 THEN 系统 SHALL 导航到对应的功能页面
3. WHEN 用户访问主页面 THEN 系统 SHALL 显示简洁的应用标题和功能说明
4. IF 用户首次访问 THEN 系统 SHALL 提供简要的使用引导或帮助提示
5. WHEN 页面加载完成 THEN 系统 SHALL 在 2 秒内完成渲染（无网络延迟情况下）

---

### 需求 4：Whisper 字幕提取功能页面

**用户故事：** 作为一名用户，我希望通过 Web 界面上传音视频文件并提取字幕，以便获得比命令行更直观的操作体验。

#### 验收标准

1. WHEN 用户访问 Whisper 功能页面 THEN 系统 SHALL 显示文件上传区域和参数配置面板
2. WHEN 用户将文件拖拽到上传区域 THEN 系统 SHALL 显示文件预览信息（文件名、大小、格式）
3. WHEN 用户点击上传区域 THEN 系统 SHALL 打开系统文件选择对话框
4. IF 用户上传不支持的文件格式 THEN 系统 SHALL 显示格式错误提示并拒绝上传
5. WHEN 用户配置参数 THEN 系统 SHALL 提供以下选项：
   - Whisper 模型大小（tiny/base/small/medium/large）
   - 目标语言（支持自动检测）
   - 输出字幕格式（SRT/VTT/TXT/JSON/CSV/全部）
6. WHEN 用户点击「开始处理」按钮 THEN 系统 SHALL 将文件上传到服务端并启动处理任务
7. WHEN 处理任务进行中 THEN 系统 SHALL 实时显示进度条和当前处理阶段（上传中/音频提取/语音识别/生成字幕）
8. WHEN 处理完成 THEN 系统 SHALL 显示结果摘要并提供字幕文件下载链接
9. IF 处理过程中发生错误 THEN 系统 SHALL 显示友好的错误提示并允许用户重试
10. WHEN 用户上传多个文件 THEN 系统 SHALL 支持批量处理并显示每个文件的处理状态

---

### 需求 5：语音训练功能页面

**用户故事：** 作为一名用户，我希望通过 Web 界面配置和启动语音训练任务，以便更直观地管理训练过程。

#### 验收标准

1. WHEN 用户访问训练功能页面 THEN 系统 SHALL 显示训练数据上传区域和训练参数配置面板
2. WHEN 用户上传训练数据 THEN 系统 SHALL 支持音频文件和对应字幕文件的上传
3. WHEN 用户配置训练参数 THEN 系统 SHALL 提供以下选项：
   - 说话人名称
   - 训练轮数（epochs）
   - 批处理大小
   - 音频采样率
4. WHEN 用户点击「开始训练」按钮 THEN 系统 SHALL 启动训练任务并进入训练监控状态
5. WHEN 训练进行中 THEN 系统 SHALL 实时显示训练进度、当前轮次和损失值变化
6. WHEN 训练完成 THEN 系统 SHALL 显示训练结果摘要并提供模型下载链接
7. IF 训练过程中发生错误 THEN 系统 SHALL 保存当前进度并显示错误信息

---

### 需求 6：文件拖拽上传功能

**用户故事：** 作为一名用户，我希望通过拖拽方式上传文件，以便获得流畅的操作体验。

#### 验收标准

1. WHEN 用户将文件拖入上传区域 THEN 系统 SHALL 高亮显示上传区域边框
2. WHEN 用户释放文件 THEN 系统 SHALL 立即开始文件预检查（格式、大小）
3. IF 文件大小超过限制（默认 2GB）THEN 系统 SHALL 显示文件过大错误提示
4. WHEN 文件通过预检查 THEN 系统 SHALL 显示文件信息并等待用户确认上传
5. WHEN 用户拖拽多个文件 THEN 系统 SHALL 支持批量添加到上传队列
6. IF 浏览器不支持拖拽上传 THEN 系统 SHALL 降级为传统点击上传方式
7. WHEN 文件上传进行中 THEN 系统 SHALL 显示上传进度条和上传速度

---

### 需求 7：实时进度显示

**用户故事：** 作为一名用户，我希望在处理过程中看到实时进度反馈，以便了解任务执行状态。

#### 验收标准

1. WHEN 任务开始执行 THEN 系统 SHALL 显示任务进度条（0-100%）
2. WHEN 任务处于不同阶段 THEN 系统 SHALL 显示当前阶段名称和阶段进度
3. WHEN 后端处理状态更新 THEN 系统 SHALL 通过 WebSocket 或轮询机制实时更新前端显示
4. IF 任务处理时间超过预期 THEN 系统 SHALL 显示预估剩余时间
5. WHEN 任务完成 THEN 系统 SHALL 显示完成动画和结果摘要
6. IF 用户刷新页面 THEN 系统 SHALL 恢复显示当前任务的进度状态

---

### 需求 8：界面设计与用户体验

**用户故事：** 作为一名用户，我希望 Web 界面美观易用，以便高效完成任务。

#### 验收标准

1. WHEN 系统渲染页面 THEN 系统 SHALL 采用响应式设计，支持桌面和平板设备
2. WHEN 用户与界面交互 THEN 系统 SHALL 提供即时的视觉反馈（按钮状态变化、加载动画）
3. WHEN 页面加载 THEN 系统 SHALL 使用现代化的 UI 组件库（如 Bootstrap 或 Tailwind CSS）
4. WHEN 用户执行操作 THEN 系统 SHALL 通过 Toast 通知显示操作结果
5. IF 用户处于暗色模式系统环境 THEN 系统 SHALL 支持自动切换暗色主题（可选）
6. WHEN 用户需要帮助 THEN 系统 SHALL 在关键操作处提供工具提示（Tooltip）

---

### 需求 9：后端 API 设计

**用户故事：** 作为一名开发者，我希望后端提供清晰的 API 接口，以便前后端分离开发和未来扩展。

#### 验收标准

1. WHEN 前端请求文件上传 THEN 系统 SHALL 提供 `POST /api/upload` 接口
2. WHEN 前端请求启动 Whisper 任务 THEN 系统 SHALL 提供 `POST /api/whisper/start` 接口
3. WHEN 前端查询任务状态 THEN 系统 SHALL 提供 `GET /api/task/<task_id>/status` 接口
4. WHEN 前端下载结果文件 THEN 系统 SHALL 提供 `GET /api/download/<file_id>` 接口
5. WHEN 前端请求启动训练任务 THEN 系统 SHALL 提供 `POST /api/train/start` 接口
6. WHEN API 返回响应 THEN 系统 SHALL 使用统一的 JSON 响应格式：
   ```json
   {
     "success": true/false,
     "data": {...},
     "error": {"code": "...", "message": "..."}
   }
   ```
7. WHEN 处理长时间运行的任务 THEN 系统 SHALL 返回任务 ID 供后续状态查询

---

### 需求 10：配置与部署

**用户故事：** 作为一名运维人员，我希望能够灵活配置 Web 服务参数，以便在不同环境中部署。

#### 验收标准

1. WHEN 系统启动 GUI 模式 THEN 系统 SHALL 支持通过命令行参数配置端口号（`--port`）
2. WHEN 系统启动 GUI 模式 THEN 系统 SHALL 支持通过命令行参数配置绑定地址（`--host`）
3. IF 未指定端口 THEN 系统 SHALL 默认使用 5000 端口
4. IF 未指定绑定地址 THEN 系统 SHALL 默认绑定 127.0.0.1（仅本地访问）
5. WHEN 用户指定 `--host 0.0.0.0` THEN 系统 SHALL 允许外部网络访问（需用户明确指定）
6. WHEN Web 服务启动 THEN 系统 SHALL 自动打开默认浏览器访问界面（可通过 `--no-browser` 禁用）
7. WHEN 配置文件中包含 GUI 相关配置 THEN 系统 SHALL 从配置文件加载默认值

---

## 技术约束

1. **兼容性**：新增功能必须保持与现有命令行模式的完全兼容
2. **依赖管理**：Flask 及相关依赖应作为可选依赖，不影响纯命令行模式的使用
3. **性能要求**：Web 界面的响应时间不应超过 500ms（不包含实际任务处理时间）
4. **安全考虑**：默认仅绑定本地地址，外部访问需要用户明确配置
5. **代码组织**：Web 相关代码应放置在独立的模块目录（如 `web/` 或 `gui/`）

---

## 项目结构建议

```
MyToolbox/
├── main.py                    # 主入口（修改以支持 --gui 参数）
├── core/
│   ├── cli/
│   │   └── cli_interface.py   # 命令行接口（修改以支持 --gui 参数）
│   └── ...
├── web/                       # 新增 Web 模块
│   ├── __init__.py
│   ├── app.py                 # Flask 应用工厂
│   ├── routes/                # 路由模块
│   │   ├── __init__.py
│   │   ├── main.py            # 主页面路由
│   │   ├── whisper.py         # Whisper API 路由
│   │   └── train.py           # 训练 API 路由
│   ├── static/                # 静态资源
│   │   ├── css/
│   │   ├── js/
│   │   └── images/
│   └── templates/             # HTML 模板
│       ├── base.html
│       ├── index.html
│       ├── whisper.html
│       └── train.html
└── ...
```

---

## 成功标准

1. 用户能够通过 `--gui` 参数成功启动 Web 界面
2. 所有现有的命令行功能在 Web 界面中都能正常使用
3. 文件拖拽上传功能在主流浏览器（Chrome、Firefox、Safari、Edge）中正常工作
4. 进度条能够准确反映任务执行进度
5. Web 界面响应迅速，用户体验流畅
