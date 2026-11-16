import os
import argparse
import torch
import torchaudio
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification


# ---------------------------------------------------------
# 1️⃣ 离线加载 SUPERB SER 模型
# ---------------------------------------------------------
def load_ser(device):
    print("[INFO] Loading SUPERB SER model (offline)...")

    model_path = "/data/lizhen/EmoVoice/superbwav2vec2-base-superb-er"

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_path,
        local_files_only=True
    )

    model = AutoModelForAudioClassification.from_pretrained(
        model_path,
        local_files_only=True
    )

    model.to(device)
    model.eval()

    return feature_extractor, model


# ---------------------------------------------------------
# 2️⃣ 标签映射
# SUPERB 官方顺序固定：
# 0: ang
# 1: hap
# 2: neu
# 3: sad
# ---------------------------------------------------------
label_list = ["ang", "hap", "neu", "sad"]

def map_label(label):
    mapping = {
        "ang": "angry",
        "hap": "happy",
        "exc": "happy",
        "sad": "sad",
        "neu": "neutral"
    }
    return mapping.get(label, label)


# ---------------------------------------------------------
# 3️⃣ 从文件名提取 GT
# lizhen_0000_angry.wav
# ---------------------------------------------------------
def get_gt_label(filepath):
    name = os.path.basename(filepath)
    emotion = name.split("_")[-1].replace(".wav", "")
    return emotion.lower()


# ---------------------------------------------------------
# 4️⃣ SER 评测主函数
# ---------------------------------------------------------
def evaluate_folder(wav_dir):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor, model = load_ser(device)

    wav_files = sorted([
        os.path.join(wav_dir, f)
        for f in os.listdir(wav_dir)
        if f.endswith(".wav")
    ])

    if len(wav_files) == 0:
        print("❌ No wav files found.")
        return

    correct = 0
    total = 0

    print(f"[INFO] Evaluating {len(wav_files)} files")

    for wav_path in tqdm(wav_files):

        gt_label = get_gt_label(wav_path)

        # 读取音频
        speech, sr = torchaudio.load(wav_path)

        # 转 mono
        if speech.shape[0] > 1:
            speech = torch.mean(speech, dim=0, keepdim=True)

        # 🔥 重采样到 16k（关键修复）
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            speech = resampler(speech)
            sr = 16000

        # 提取特征
        inputs = extractor(
            speech.squeeze().numpy(),
            sampling_rate=sr,
            return_tensors="pt"
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits

        predicted_id = torch.argmax(logits, dim=-1).item()
        predicted_label = label_list[predicted_id]
        predicted_label = map_label(predicted_label)

        if predicted_label == gt_label:
            correct += 1

        total += 1

    acc = correct / total

    print("\n===== SER RESULT =====")
    print("Total:", total)
    print("Correct:", correct)
    print("Emotion ACC:", acc)


# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_dir", type=str, required=True)
    args = parser.parse_args()

    evaluate_folder(args.wav_dir)