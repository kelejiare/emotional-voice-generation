import argparse, json
from pathlib import Path

# 6 段情绪转变（大约 60s），你也可以自行改文案
DEFAULT_SEGMENTS = [
    ("neutral",
     "今天我想分享一个小故事。清晨出门时，空气很清新，我的心情也很平静。"),
    ("happy",
     "走到路口时，阳光突然洒下来，我忍不住笑了。今天好像什么都很顺利。"),
    ("surprised",
     "正当我抬头看天，手机响了。消息里写着一个好消息，我一下子愣住了，然后兴奋得加快了脚步。"),
    ("sad",
     "可走着走着，我又想起一件遗憾的事。那种失落慢慢涌上来，声音也不自觉放轻了。"),
    ("angry",
     "更糟的是，我看到有人随手把垃圾扔在地上。那一刻我有点生气，语气变得更硬、更急。"),
    ("calm",
     "我深呼吸，告诉自己别被情绪带走。最后我把垃圾捡起来，慢慢走向公园，心情重新安定下来。"),
]

EMO_PROMPT_ZH = {
    "neutral": "用自然、平静的语气朗读，语速正常，吐字清晰，按标点自然停顿。",
    "happy": "用真诚的开心朗读：音色更明亮、语调更上扬，节奏更活泼，连读顺畅。",
    "surprised": "用惊喜与兴奋朗读：语速略快，语调起伏更明显，但不要尖叫，保持清晰。",
    "sad": "用难过的情绪朗读：语速略慢，音量更轻，尾音更下沉，停顿稍长。",
    "angry": "用克制的愤怒朗读：能量更高、咬字更硬、语速略快，但不要吼叫。",
    "calm": "用放松和释然的语气朗读：语速稍慢，语调平稳，呼吸更自然。",
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref_wav", required=True, help="target speaker reference wav (Chinese, ~5s)")
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--key_prefix", default="story")
    args = ap.parse_args()

    outp = Path(args.out_jsonl)
    outp.parent.mkdir(parents=True, exist_ok=True)

    with outp.open("w", encoding="utf-8") as f:
        for i, (emo, text) in enumerate(DEFAULT_SEGMENTS):
            d = {
                "key": f"{args.key_prefix}_{i:04d}_{emo}",
                "emotion": emo,
                # 关键：情绪提示只放这里，绝不要写进 source_text/target_text
                "emotion_text_prompt": (
                    EMO_PROMPT_ZH.get(emo, EMO_PROMPT_ZH["neutral"])
                    + " 只输出中文，不要输出英文或字母；不要朗读任何提示词。"
                ),
                # 关键：Stage1/Stage2 都用同一个中文 ref wav，尽量降低英文腔
                "neutral_speaker_wav": args.ref_wav,
                "source_text": text,
                "target_text": text,
                # dataset/infer 兼容字段
                "answer_cosyvoice_speech_token": [],
                "target_len": 0,
                "source_len": max(1, len(text)),
            }
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"[INFO] wrote: {outp}")

if __name__ == "__main__":
    main()