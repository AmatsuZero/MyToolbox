# CODEBUDDY.md

This file provides guidance to CodeBuddy Code when working with code in this repository.

## Project Overview

MyToolbox is a Python-based speech-to-text system built on OpenAI Whisper. It provides both CLI and web interfaces for transcribing audio/video files and generating subtitles in multiple formats. The system supports multiple Whisper engine implementations with automatic platform-aware selection.

## Development Setup

### Environment Setup
```bash
# The project uses Python 3.12+ with uv package manager
uv sync

# Alternative: use pip with locked dependencies
pip install -r requirements.txt

# Install optional dependencies for web interface
uv sync --extra gui

# Install optional dependencies for TTS training
uv sync --extra tts
```

### Running the Application

#### CLI Interface
```bash
# Basic transcription
python main.py --input <audio_file> --output <output_dir>

# With specific model and language
python main.py --input <path> --model medium --language zh

# Batch processing with GPU acceleration
python main.py --input <directory> --recursive --enable-gpu --batch-size 4

# Dry run to preview what will be processed
python main.py --input <path> --dry-run
```

#### Web Interface
```bash
# Start the Flask web server
python web/app.py

# Or with custom host/port
python web/app.py --host 0.0.0.0 --port 8080
```

### Testing
```bash
# Run all tests (tests/ directory contains unit tests)
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_specific_module.py

# Run with verbose output
python -m pytest -v tests/
```

## Architecture Overview

### High-Level Component Structure

```
SpeechToTextSystem (main.py)
├── MediaFileScanner (core/utils/file_scanner.py)
├── FormatConverter (core/utils/format_converter.py)
├── WhisperIntegration (modules/whisper/whisper_integration.py)
│   └── EngineSelector → WhisperEngine (whisper_engines/)
├── SubtitleGenerator (modules/whisper/subtitle_generator.py)
├── ErrorHandler (core/utils/error_handler.py)
└── ConfigManager (core/config/config.py)
```

### Multi-Engine Architecture

The system implements a **plugin-based engine architecture** for Whisper model execution:

#### Engine Registry Pattern
- **Location**: `whisper_engines/`
- **Registry**: `engine_registry.py` maintains a priority-ordered registry of available engines
- **Selector**: `engine_selector.py` selects the optimal engine based on platform and availability

#### Available Engines (Priority Order by Platform)

1. **WhisperKit Engine** (`whisperkit_engine.py`)
   - **Priority**: 10 (highest)
   - **Platform**: macOS ARM64 only (Apple Silicon)
   - **Requirements**: macOS 14+, WhisperKit CLI installed
   - **Acceleration**: Native Metal via WhisperKit
   - **Implementation**: Subprocess-based CLI execution with JSON report parsing
   - **Note**: Requires `WHISPERKIT_CLI_PATH` environment variable or WhisperKit CLI in PATH

2. **Faster-Whisper Engine** (`faster_whisper_engine.py`)
   - **Priority**: 20
   - **Platform**: Cross-platform (macOS, Linux, Windows)
   - **Acceleration**: CUDA (GPU) or CPU with quantization
   - **Implementation**: Direct library calls with streaming segment processing
   - **Optimization**: Auto CPU thread allocation based on available cores

3. **OpenAI Whisper Engine** (`openai_whisper_engine.py`)
   - **Priority**: 30 (fallback)
   - **Platform**: Universal
   - **Acceleration**: CUDA, MPS (Metal Performance Shaders), or CPU
   - **Implementation**: Direct library calls with device-specific optimizations
   - **Note**: Includes MPS dtype patching for Apple Silicon compatibility

#### Engine Interface (Protocol)

All engines implement the `WhisperEngine` protocol defined in `base_engine.py`:

```python
class WhisperEngine(Protocol):
    name: str
    
    def configure(self, options: EngineOptions) -> None: ...
    def is_available(self) -> bool: ...
    def load_model(self, model_size: str) -> bool: ...
    def get_supported_models(self) -> List[str]: ...
    def transcribe(self, audio_file, language, progress_callback) -> EngineTranscriptionResult: ...
    def get_device_info(self) -> Dict[str, Any]: ...
```

### Processing Pipeline Flow

```
Input File(s)
    ↓
1. File Scanning (MediaFileScanner)
   - Identifies audio/video files
   - Validates format support
   - Returns list of MediaFile objects
    ↓
2. Format Conversion (FormatConverter)
   - Extracts audio from video files
   - Converts to Whisper-optimal format (16kHz WAV)
   - Normalizes audio levels
    ↓
3. Engine Selection (EngineSelector)
   - Detects platform (Darwin/Linux/Windows, ARM64/x86_64)
   - Checks CUDA availability
   - Selects highest-priority available engine
    ↓
4. Model Loading (Selected WhisperEngine)
   - Loads Whisper model to appropriate device
   - Returns success/failure status
    ↓
5. Transcription (Selected WhisperEngine)
   - Processes audio in segments
   - Returns TranscriptionResult with text and timestamps
    ↓
6. Subtitle Generation (SubtitleGenerator)
   - Creates SubtitleSegment objects from transcription
   - Generates multiple output formats (SRT, VTT, ASS, TXT, JSON, CSV)
   - Saves to output directory
    ↓
7. Error Handling & Reporting (ErrorHandler)
   - Logs errors and warnings
   - Generates performance reports
   - Creates summary JSON files
```

### Configuration System

Configuration is managed through a hierarchical system:

1. **Default config**: Built-in presets (fast/accurate/balanced/low_memory/high_quality)
2. **User config**: Custom settings in `config/user_config.json`
3. **CLI arguments**: Runtime parameters that override file configs

**Preset configurations** (`core/config/config.py`):
- `fast`: Uses tiny model for quick processing
- `accurate`: Uses medium model for better quality
- `balanced`: Uses base model (default, good trade-off)
- `low_memory`: Minimal memory footprint
- `high_quality`: Uses large model for best accuracy

### Web Interface Architecture

The web interface uses Flask with a task-based asynchronous architecture:

- **Application Factory**: `web/app.py` defines `create_app()` for flexible configuration
- **Blueprint Structure**:
  - `routes/main.py`: Main HTML interface (`/`)
  - `routes/upload.py`: File upload endpoints (`/api/upload/*`)
  - `routes/whisper.py`: Transcription endpoints (`/api/whisper/*`)
  - `routes/train.py`: TTS training endpoints (`/api/train/*`)
- **Task Manager**: `web/task_manager.py` implements a singleton pattern for managing asynchronous tasks
  - Thread-safe task state management
  - Progress tracking (0-100%)
  - Task status: pending → uploading → processing → completed/failed

**Task Processing Flow**:
```
POST /api/upload → Store file → Create task
                                    ↓
                        Background thread starts
                                    ↓
                        Process transcription
                                    ↓
                        Update task progress
                                    ↓
GET /api/whisper/status/<task_id> ← Client polls for updates
                                    ↓
                        Task completes
                                    ↓
GET /api/whisper/download/<task_id> ← Client downloads results
```

## Key Module Responsibilities

### `main.py` - SpeechToTextSystem
Central orchestrator that coordinates the entire transcription workflow. Maintains component lifecycle and handles batch processing.

### `whisper_engines/` - Engine Implementations
Each engine provides platform-optimized Whisper model execution. Engines are discovered and selected automatically based on availability and platform.

### `core/utils/file_scanner.py` - MediaFileScanner
Scans directories for supported audio/video files. Supports 8 audio formats (mp3, wav, flac, etc.) and 9 video formats (mp4, mkv, etc.).

### `core/utils/format_converter.py` - FormatConverter
Handles all audio/video format conversions using ffmpeg. Converts inputs to Whisper-optimal format (16kHz mono/stereo WAV).

### `core/utils/error_handler.py` - ErrorHandler
Centralized error handling with retry logic, system monitoring, and report generation. Implements exponential backoff for transient failures.

### `modules/whisper/whisper_integration.py` - WhisperIntegration
High-level interface for Whisper transcription. Uses EngineSelector to choose and configure the appropriate engine, then delegates transcription work.

### `modules/whisper/subtitle_generator.py` - SubtitleGenerator
Converts transcription results into multiple subtitle formats. Supports SRT, VTT, ASS, TXT, JSON, and CSV output formats.

### `core/config/config.py` - ConfigManager
Manages configuration loading, validation, and preset application. Provides config merging from multiple sources.

## Important Design Patterns

### Strategy Pattern (Engine Selection)
The `WhisperEngine` protocol defines a common interface, with each engine implementing platform-specific optimizations. This allows runtime selection of the best available engine.

### Registry Pattern (Engine Discovery)
`EngineRegistry` maintains a class-level registry of engines with priority and availability metadata. New engines can be added by inheriting from `WhisperEngine` and registering.

### Singleton Pattern (Task Management)
`TaskManager` uses a singleton pattern with thread-safe operations for managing web interface tasks.

### Component-Based Architecture (Main System)
`SpeechToTextSystem` uses a `_components` dictionary to manage dependencies, making it easy to swap implementations or add new components.

## Critical Integration Points

### Adding a New Whisper Engine
1. Create new engine class in `whisper_engines/` implementing `WhisperEngine` protocol
2. Define `name` class attribute and priority
3. Register in `EngineRegistry` using `@register_engine(name, priority)`
4. Implement `is_available()` to check platform/dependencies
5. Implement `load_model()` and `transcribe()` methods
6. Add platform-specific priority in `engine_selector.py`

### Adding a New Subtitle Format
1. Add format to `SUPPORTED_OUTPUT_FORMATS` in `subtitle_generator.py`
2. Implement `generate_<format>()` method that takes segments and returns formatted string
3. Update `save_subtitle_files()` to handle the new format
4. Add format to CLI help text in `cli_interface.py`

### Modifying Configuration Schema
1. Update dataclass definitions in `core/config/config.py`
2. Update `validate_config()` to check new fields
3. Update preset configurations if needed
4. Update CLI argument parsing in `cli_interface.py`

## Dependencies & Requirements

### Core Dependencies
- **openai-whisper**: OpenAI's Whisper model (fallback engine)
- **faster-whisper**: Optimized Whisper implementation (primary engine)
- **torch**: PyTorch for model execution
- **torchaudio**: Audio processing
- **psutil**: System resource monitoring
- **PyYAML**: Configuration file parsing

### Optional Dependencies
- **flask + flask-cors**: Web interface (`--extra gui`)
- **mycroft-mimic3-tts**: TTS training functionality (`--extra tts`)

### External Tools
- **ffmpeg**: Required for audio/video format conversion
- **WhisperKit CLI**: Optional, for Apple Silicon native acceleration

### Platform-Specific Support
- **macOS ARM64**: Best performance with WhisperKit engine
- **macOS x86_64**: Use Faster-Whisper or OpenAI Whisper
- **Linux with CUDA**: Use Faster-Whisper with GPU acceleration
- **Linux CPU-only**: Use Faster-Whisper with CPU optimization
- **Windows**: Similar to Linux (CUDA or CPU)

## Common Development Tasks

### Running a Single Transcription Test
```bash
# Test with a small audio file using base model
python main.py --input test_audio.mp3 --model base --output test_output

# Check generated files in test_output/
ls -la test_output/
```

### Debugging Engine Selection
```python
# Add this to your code to see which engine was selected
from whisper_engines import EngineSelector, create_engine_options

options = create_engine_options(model_size="base")
engine = EngineSelector.select_engine(options)
print(f"Selected engine: {engine.name}")
print(f"Device info: {engine.get_device_info()}")
```

### Testing Different Engines
```bash
# Force OpenAI Whisper by setting environment variable
export FORCE_OPENAI_WHISPER=1
python main.py --input audio.wav

# Force Faster-Whisper
export FORCE_FASTER_WHISPER=1
python main.py --input audio.wav

# Use WhisperKit on macOS (requires installation)
export WHISPERKIT_CLI_PATH=/path/to/whisperkit
python main.py --input audio.wav
```

### Monitoring System Resources
The system automatically monitors and logs resource usage. Check generated report files:
```bash
# View performance metrics
cat error_log.performance.json

# View error logs
cat error_log.errors.json
```

## File Locations & Outputs

### Input Directories
- Default scan directory: `downloads/`
- Configurable via `--input` CLI argument

### Output Directories
- Default output: `output/` or `subtitles/`
- Generated files: `<basename>.<format>` (e.g., `audio.srt`, `audio.vtt`, etc.)
- Web uploads: `uploads/`
- Converted audio: `converted_audio/`

### Configuration Files
- Default config: `config/default_config.json` (if exists)
- User config: `config/user_config.json` (if exists)

### Log Files
- Performance report: `error_log.performance.json`
- Error report: `error_log.errors.json`

## Troubleshooting Notes

### Model Loading Issues
If a model fails to load, the system will automatically fall back to the next available engine. Check `error_log.errors.json` for details.

### Memory Issues
Use `--optimize-memory` flag or apply the `low_memory` preset configuration. This reduces batch size and enables garbage collection.

### GPU Not Detected
Check PyTorch CUDA availability:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"MPS available: {torch.backends.mps.is_available()}")
```

### WhisperKit Not Found
Set the environment variable:
```bash
export WHISPERKIT_CLI_PATH=/opt/homebrew/bin/whisperkit
# Or add whisperkit to your PATH
```

### FFmpeg Not Found
Install ffmpeg:
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```
