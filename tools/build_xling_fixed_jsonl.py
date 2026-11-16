import argparse
import json
import re
from pathlib import Path

PUNCTS = "，。！？；："

def norm_emo(e: str) -> str:
    e = (e or "neutral").strip().lower()
    alias = {"joy": "happy", "happiness": "happy", "anger": "angry", "sadness": "sad", "sorrow": "sad"}
    return alias.get(e, e)

def emo_prompt_en(emo: str) -> str:
    # dataset 的 prompt 模板是英文：Say this sentence with emotion of {emotion_text_prompt}
    table = {
        "angry":  "controlled anger, higher energy, sharper attacks, slightly faster pace, tense but NOT shouting",
        "happy":  "genuine happiness, brighter tone, slightly higher pitch, lively rhythm, clear articulation",
        "sad":    "sadness, lower energy, softer volume, slower pace, longer pauses, clear articulation",
        "neutral":"calm neutral tone, steady pace, natural intonation, clear articulation",
    }
    base = table.get(emo, table["neutral"])
    return f"{base}. Speak fluent Mandarin Chinese. Pause naturally at punctuation and phrase boundaries."

def normalize_zh_punct(s: str) -> str:
    s = (s or "").strip()
    s = s.replace(",", "，").replace(".", "。").replace("!", "！").replace("?", "？").replace(";", "；").replace(":", "：")
    s = re.sub(r"\s+", "，", s)
    if not re.search(r"[。！？]$", s):
        s += "。"
    return s

def split_text(s: str, max_chars: int) -> list[str]:
    """
    先按中文标点切，再合并到 <= max_chars 的段落，保证每段不要太长。
    """
    s = normalize_zh_punct(s)
    # 以标点做切分，但保留标点
    parts = []
    buf = ""
    for ch in s:
        buf += ch
        if ch in PUNCTS:
            parts.append(buf.strip())
            buf = ""
    if buf.strip():
        parts.append(buf.strip())

    merged = []
    cur = ""
    for p in parts:
        if len(cur) + len(p) <= max_chars:
            cur += p
        else:
            if cur:
                merged.append(cur)
            cur = p
    if cur:
        merged.append(cur)

    # 兜底：极长无标点的情况
    final = []
    for seg in merged:
        if len(seg) <= max_chars:
            final.append(seg)
        else:
            for i in range(0, len(seg), max_chars):
                chunk = seg[i:i+max_chars]
                if not re.search(r"[。！？]$", chunk):
                    chunk += "。"
                final.append(chunk)
    return final

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--emotion", default="neutral")
    ap.add_argument("--ref_wav", required=True)
    ap.add_argument("--out_fixed", required=True)
    ap.add_argument("--max_chars", type=int, default=28)  # 关键：控制每段长度，稳定
    ap.add_argument("--key_prefix", default="xling")
    args = ap.parse_args()

    emo = norm_emo(args.emotion)
    text = args.text
    segs = split_text(text, args.max_chars)

    out_fixed = Path(args.out_fixed)
    out_fixed.parent.mkdir(parents=True, exist_ok=True)

    prompt = emo_prompt_en(emo)
    with out_fixed.open("w", encoding="utf-8") as fout:
        for i, seg in enumerate(segs):
            d = {
                "key": f"{args.key_prefix}_{i:04d}_{emo}",
                "emotion": emo,
                "emotion_text_prompt": prompt,
                "neutral_speaker_wav": args.ref_wav,   # Stage1 也用目标 speaker wav，尽量贴音色
                "source_text": seg,
                "target_text": seg,
                # inference_mode 仍会读这些字段，必须补齐（PDF 里也写过这个坑）:
                "answer_cosyvoice_speech_token": [],
                "target_len": 0,
                "source_len": max(1, len(seg)),
            }
            fout.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"[INFO] wrote fixed jsonl: {out_fixed} (n={len(segs)})")

if __name__ == "__main__":
    main()