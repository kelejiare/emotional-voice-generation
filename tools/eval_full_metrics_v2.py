import os
import json
import glob
import argparse
import whisper
import torch
import numpy as np
from jiwer import wer
from tqdm import tqdm

# ========= UTMOS =========
try:
    from utmos import UTMOS
    utmos_model = UTMOS()
    HAS_UTMOS = True
except:
    HAS_UTMOS = False
    print("[WARN] UTMOS not available, skipping MOS evaluation")

# ========= Emotion inference =========
def infer_emotion_from_filename(fname):
    fname = fname.lower()
    if "angry" in fname:
        return "angry"
    if "happy" in fname:
        return "happy"
    if "sad" in fname:
        return "sad"
    return "neutral"

# ========= Whisper =========
def load_whisper():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Loading Whisper on {device}")
    model = whisper.load_model("large-v3", device=device)
    return model

# ========= Evaluate =========
def evaluate_model(model_name, audio_dir, gt_dict, whisper_model):
    results = {}
    total_wer = []
    total_utmos = []
    emo_correct = 0
    total = 0

    wavs = sorted(glob.glob(os.path.join(audio_dir, "*.wav")))

    if len(wavs) == 0:
        print(f"[WARN] No wav files found in {audio_dir}")
        return {}

    print(f"[INFO] Evaluating {model_name} ({len(wavs)} files)")

    for wav in tqdm(wavs, desc=model_name):
        fname = os.path.basename(wav)
        key = fname.replace(".wav", "")
        pred_emotion = infer_emotion_from_filename(fname)

        gt_text = gt_dict.get(key, "")
        if gt_text == "":
            print(f"[WARN] GT text missing for {key}")

        # ---- ASR ----
        result = whisper_model.transcribe(wav, language="en")
        pred_text = result["text"].strip()

        sample_wer = wer(gt_text, pred_text)
        total_wer.append(sample_wer)

        # ---- Emotion ACC ----
        if pred_emotion in key:
            emo_correct += 1

        # ---- UTMOS ----
        if HAS_UTMOS:
            try:
                score = utmos_model.score(wav)
                total_utmos.append(score)
            except:
                pass

        total += 1

    results["WER"] = float(np.mean(total_wer))
    results["Emotion_ACC"] = float(emo_correct / total)

    if HAS_UTMOS and len(total_utmos) > 0:
        results["UTMOS"] = float(np.mean(total_utmos))

    return results

# ========= Main =========
def main(args):
    run_dir = args.run_dir

    # ---- 读取 GT ----
    gt_path = os.path.join(run_dir, "emovoice_Aplus", "gt_text")
    if not os.path.exists(gt_path):
        raise FileNotFoundError("gt_text not found")

    gt_dict = {}
    with open(gt_path, "r") as f:
        for line in f:
            key, text = line.strip().split("\t")
            gt_dict[key] = text

    whisper_model = load_whisper()

    report = {}

    # 1️⃣ cosyvoice3_instruct
    report["cosyvoice3_instruct"] = evaluate_model(
        "cosyvoice3_instruct",
        os.path.join(run_dir, "cosyvoice3_instruct", "pred_audio"),
        gt_dict,
        whisper_model
    )

    # 2️⃣ emovoice_Aplus
    report["emovoice_Aplus"] = evaluate_model(
        "emovoice_Aplus",
        os.path.join(run_dir, "emovoice_Aplus", "pred_audio", "neutral_prompt_speech"),
        gt_dict,
        whisper_model
    )

    # 3️⃣ emovoicePP_Aplus（未VC）
    report["emovoicePP_Aplus"] = evaluate_model(
        "emovoicePP_Aplus",
        os.path.join(run_dir, "emovoicePP_Aplus", "pred_audio", "neutral_prompt_speech"),
        gt_dict,
        whisper_model
    )

    # 4️⃣ emovoicePP_Aplus_vc（VC后）
    report["emovoicePP_Aplus_vc"] = evaluate_model(
        "emovoicePP_Aplus_vc",
        os.path.join(run_dir, "emovoicePP_Aplus", "pred_audio_vc", "neutral_prompt_speech"),
        gt_dict,
        whisper_model
    )

    out_path = os.path.join(run_dir, "compare_full_metrics_v2.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=4)

    print("\n========== FINAL REPORT ==========")
    print(json.dumps(report, indent=4))
    print(f"\nSaved to: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    args = parser.parse_args()
    main(args)