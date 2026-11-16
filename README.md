# 情感可控语音生成与音色迁移

基于 EmoVoice 与 CosyVoice3，探索情感语音生成、音色迁移和自动化评测。

## 方法

- **英文**：通过 EmoVoice 或 EmoVoice-PP 生成情感语音，再用 CosyVoice3 迁移音色。
- **中文**：使用 CosyVoice3 的指令生成与音色迁移，支持长文本分段和音频拼接。
- **评测**：使用 Whisper 转写和 WER、语音情绪分类及音频统计工具。

## 环境与运行

原实验使用 Linux、Python 3.10、PyTorch 2.4.1+cu121 和 NVIDIA V100。请自行准备 [EmoVoice 权重](https://huggingface.co/yhaha/EmoVoice)、Qwen、CosyVoice 解码器，以及 [CosyVoice3](https://github.com/FunAudioLLM/CosyVoice) 的代码和权重。

- `scripts/run_lizhen_en_compare_Aplus.sh`：英文实验与比较。
- `scripts/run_best_emotion_demo_60s_vc.sh`：中文语音生成演示。
- `tools/`：输入构建、语音处理和评测工具。
- `examples/tts/`、`src/slam_llm/`：模型推理和训练代码。

脚本保留原实验的路径和依赖假设，运行前需检查并替换模型、参考音频及第三方库路径。本仓库不含权重、参考音频和运行结果，也未重新验证完整运行。

## 结果说明

本地 4 条英文样本中，音色迁移阶段的平均 RTF 约为 0.267；该值不包含前一阶段的生成耗时，不能作为端到端系统 RTF。`tools/eval_full_metrics.py` 中从文件名计算的情绪指标并非语音分类准确率，不应用于评估模型。

## 参考与许可

代码基于 [EmoVoice](https://github.com/yanghaha0908/EmoVoice)，并使用 SLAM-LLM、CosyVoice 等实现。上游声明代码采用 MIT、预训练模型采用 CC-BY-NC；使用第三方代码、模型和数据时请核对各自许可。论文见 [EmoVoice](https://arxiv.org/abs/2504.12867)。
