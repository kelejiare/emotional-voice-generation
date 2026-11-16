# 情感可控语音生成与音色迁移

基于 EmoVoice 与 CosyVoice3 的语音生成实验，探索情感提示、音色迁移和自动化评测。该项目基于开源模型进行组合与工程调试，基础模型与论文成果属于原作者。

## 方法

- **英文实验**：使用 EmoVoice 或 EmoVoice-PP 生成带情感的语音，再使用 CosyVoice3 的 voice conversion 迁移音色；脚本也包含单阶段对照。
- **中文实验**：使用 CosyVoice3 的指令生成与 voice conversion 组成两阶段链路，支持长文本分段和音频拼接。
- **评测工具**：包含 Whisper 转写及 WER、语音情绪分类、生成时长和音频统计脚本。各脚本的统计口径不同，使用前需检查输入与实现。

本项目是模型组合与推理流程实验；两阶段处理不等同于证明情感和音色在模型表征中严格解耦。

## 环境与模型

原始实验记录使用 Linux、Python 3.10、PyTorch 2.4.1+cu121 和 NVIDIA V100。`requirements.txt` 来自原工作环境，尚未在干净环境重新验证；安装前请依据 CUDA、PyTorch 及第三方项目的版本要求检查依赖。

需要自行准备 [EmoVoice 权重](https://huggingface.co/yhaha/EmoVoice)、Qwen 模型、CosyVoice 解码器和 [CosyVoice3](https://github.com/FunAudioLLM/CosyVoice) 代码及权重。模型、数据和参考音频均未收录；使用时须遵循各自的许可及授权。

## 代码入口

```text
examples/tts/    EmoVoice 推理和训练代码
src/slam_llm/    上游模型和工具代码
scripts/         英文比较、中文生成和演示脚本
tools/           数据构建、语音生成、评测和音频处理工具
```

英文比较入口为 `scripts/run_lizhen_en_compare_Aplus.sh`；中文演示入口为 `scripts/run_best_emotion_demo_60s_vc.sh`。这些脚本保留原实验的路径和依赖假设，需先检查并替换权重、参考音频和第三方库路径，不能直接视为开箱即用。JSONL 字段以脚本及相关构建工具为准。没有附带原始语音、模型权重或运行结果。

## 结果与限制

本地对比报告包含 4 条英文样本。其 CosyVoice 音色迁移阶段 RTF 均值约 0.267，并不包含前一阶段 EmoVoice 生成耗时，不能当作完整系统端到端 RTF。要比较整体效率，需在相同硬件和输入下重新记录各阶段及端到端耗时。

`tools/eval_full_metrics_v2.py` 中的情绪指标来自文件名而非音频分类，不能作为情绪识别准确率；`tools/ser_eval_superb_offline.py` 使用音频分类模型，但运行路径和标签映射需要使用者核对。主观听感评价需要独立设计并记录评测过程。本仓库未重新验证完整运行和实验指标。

## 来源与许可

本仓库基于 [EmoVoice](https://github.com/yanghaha0908/EmoVoice)，上游基准提交 `5285cb891611cf1ee2d9bd07b931cd3cf967cd64`；其代码又基于 SLAM-LLM 和 CosyVoice 等项目。EmoVoice 上游 README 声明代码采用 MIT、其预训练模型采用 CC-BY-NC；第三方子项目、模型和数据可能另有许可，使用时请分别核对。原项目的引用信息见 [EmoVoice 论文](https://arxiv.org/abs/2504.12867)。
