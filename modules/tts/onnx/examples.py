#!/usr/bin/env python3
"""
ONNX TTS 使用示例

演示如何使用ONNX TTS模块进行文本到语音合成
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from modules.tts.onnx import (
    ONNXModelLoader,
    ONNXInferenceEngine,
    TextProcessor,
    AudioProcessor,
    InferenceProgress,
    check_onnx_available
)


def example_basic_usage():
    """示例1：基本用法"""
    print("=" * 50)
    print("示例1：基本用法")
    print("=" * 50)
    
    # 检查依赖
    available, error = check_onnx_available()
    if not available:
        print(f"错误: {error}")
        return
    
    # 模型路径（请替换为实际路径）
    model_path = "path/to/your/model.onnx"
    
    # 1. 加载模型
    print("\n1. 加载模型...")
    model_loader = ONNXModelLoader(
        model_path=model_path,
        device="auto"
    )
    print(f"   模型已加载: {model_loader}")
    
    # 2. 初始化组件
    print("\n2. 初始化组件...")
    text_processor = TextProcessor()
    inference_engine = ONNXInferenceEngine(
        model_loader=model_loader,
        text_processor=text_processor
    )
    audio_processor = AudioProcessor(sample_rate=22050)
    
    # 3. 执行推理
    print("\n3. 执行推理...")
    text = "你好，这是一个测试。"
    output_dict = inference_engine.infer_text(text)
    print(f"   推理完成，输出节点: {list(output_dict.keys())}")
    
    # 4. 处理音频
    print("\n4. 处理音频...")
    waveform = audio_processor.process_model_output(output_dict)
    print(f"   波形shape: {waveform.shape}")
    
    # 5. 保存音频
    print("\n5. 保存音频...")
    output_path = audio_processor.save_wav(waveform, "output/example1.wav")
    print(f"   音频已保存: {output_path}")
    
    print("\n✅ 示例1完成！")


def example_with_progress():
    """示例2：带进度回调"""
    print("\n" + "=" * 50)
    print("示例2：带进度回调")
    print("=" * 50)
    
    model_path = "path/to/your/model.onnx"
    
    # 定义进度回调函数
    def progress_callback(progress: InferenceProgress):
        print(f"   [{progress.stage}] {progress.progress:.0f}% - {progress.message}")
    
    # 加载模型
    print("\n加载模型...")
    model_loader = ONNXModelLoader(model_path, device="auto")
    
    # 初始化推理引擎（带进度回调）
    text_processor = TextProcessor()
    inference_engine = ONNXInferenceEngine(
        model_loader=model_loader,
        text_processor=text_processor,
        progress_callback=progress_callback
    )
    audio_processor = AudioProcessor()
    
    # 执行推理
    print("\n执行推理（带进度显示）...")
    text = "这是一个带进度显示的示例。"
    output_dict = inference_engine.infer_text(text)
    
    # 保存音频
    waveform = audio_processor.process_model_output(output_dict)
    audio_processor.save_wav(waveform, "output/example2.wav")
    
    print("\n✅ 示例2完成！")


def example_batch_processing():
    """示例3：批量处理"""
    print("\n" + "=" * 50)
    print("示例3：批量处理")
    print("=" * 50)
    
    model_path = "path/to/your/model.onnx"
    
    # 加载模型
    print("\n加载模型...")
    model_loader = ONNXModelLoader(model_path)
    text_processor = TextProcessor()
    inference_engine = ONNXInferenceEngine(model_loader, text_processor)
    audio_processor = AudioProcessor()
    
    # 批量文本
    texts = [
        "第一段文本。",
        "第二段文本。",
        "第三段文本。"
    ]
    
    print(f"\n批量处理 {len(texts)} 个文本...")
    results = inference_engine.infer_batch(texts)
    
    # 保存所有音频
    for i, output_dict in enumerate(results):
        waveform = audio_processor.process_model_output(output_dict)
        output_path = f"output/example3_part{i+1}.wav"
        audio_processor.save_wav(waveform, output_path)
        print(f"   已保存: {output_path}")
    
    print("\n✅ 示例3完成！")


def example_with_config():
    """示例4：使用配置文件"""
    print("\n" + "=" * 50)
    print("示例4：使用配置文件")
    print("=" * 50)
    
    model_path = "path/to/your/model.onnx"
    config_file = "modules/tts/onnx/config_template.yaml"
    
    # 使用配置文件初始化文本处理器
    print("\n加载配置文件...")
    text_processor = TextProcessor(config_file=config_file)
    print(f"   配置: {text_processor.config}")
    
    # 加载模型
    model_loader = ONNXModelLoader(model_path)
    inference_engine = ONNXInferenceEngine(model_loader, text_processor)
    audio_processor = AudioProcessor()
    
    # 执行推理
    print("\n执行推理...")
    text = "使用配置文件的示例。"
    output_dict = inference_engine.infer_text(text)
    
    # 保存音频
    waveform = audio_processor.process_model_output(output_dict)
    audio_processor.save_wav(waveform, "output/example4.wav")
    
    print("\n✅ 示例4完成！")


def example_mp3_output():
    """示例5：输出MP3格式"""
    print("\n" + "=" * 50)
    print("示例5：输出MP3格式")
    print("=" * 50)
    
    model_path = "path/to/your/model.onnx"
    
    # 检查ffmpeg
    audio_processor = AudioProcessor()
    if not audio_processor.check_ffmpeg_available():
        print("⚠️  ffmpeg未安装，无法输出MP3格式")
        print("   请安装ffmpeg后再试")
        return
    
    # 加载模型
    print("\n加载模型...")
    model_loader = ONNXModelLoader(model_path)
    text_processor = TextProcessor()
    inference_engine = ONNXInferenceEngine(model_loader, text_processor)
    
    # 执行推理
    print("\n执行推理...")
    text = "这是一个MP3输出示例。"
    output_dict = inference_engine.infer_text(text)
    
    # 保存为MP3
    print("\n保存为MP3...")
    waveform = audio_processor.process_model_output(output_dict)
    audio_processor.save_mp3(
        waveform,
        "output/example5.mp3",
        bitrate="192k"
    )
    
    print("\n✅ 示例5完成！")


def example_statistics():
    """示例6：查看统计信息"""
    print("\n" + "=" * 50)
    print("示例6：查看统计信息")
    print("=" * 50)
    
    model_path = "path/to/your/model.onnx"
    
    # 加载模型
    model_loader = ONNXModelLoader(model_path)
    text_processor = TextProcessor()
    inference_engine = ONNXInferenceEngine(model_loader, text_processor)
    audio_processor = AudioProcessor()
    
    # 执行多次推理
    texts = ["测试1", "测试2", "测试3"]
    
    print("\n执行多次推理...")
    for text in texts:
        output_dict = inference_engine.infer_text(text)
        waveform = audio_processor.process_model_output(output_dict)
    
    # 查看统计信息
    print("\n统计信息:")
    stats = inference_engine.get_statistics()
    print(f"   推理次数: {stats['inference_count']}")
    print(f"   总耗时: {stats['total_time']:.3f} 秒")
    print(f"   平均耗时: {stats['average_time']:.3f} 秒")
    print(f"   使用设备: {stats['device']}")
    
    print("\n✅ 示例6完成！")


def main():
    """运行所有示例"""
    print("ONNX TTS 使用示例")
    print("=" * 50)
    
    # 检查依赖
    available, error = check_onnx_available()
    if not available:
        print(f"\n❌ 错误: {error}")
        return
    
    print("\n✅ ONNX Runtime 可用")
    
    # 创建输出目录
    Path("output").mkdir(exist_ok=True)
    
    # 运行示例
    try:
        # 注意：以下示例需要实际的ONNX模型文件
        # 请将 "path/to/your/model.onnx" 替换为实际路径
        
        print("\n⚠️  注意：请先将示例中的模型路径替换为实际路径")
        print("   模型路径: path/to/your/model.onnx")
        
        # 取消注释以运行示例
        # example_basic_usage()
        # example_with_progress()
        # example_batch_processing()
        # example_with_config()
        # example_mp3_output()
        # example_statistics()
        
    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
