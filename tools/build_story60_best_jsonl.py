import argparse, json
from pathlib import Path

SEGMENTS = [
    ("neutral",
     "现在我想用一段一分钟的小故事，展示情绪语音的连续变化。我会像平时说话那样，自然地讲出来。"),
    ("happy",
     "今天早上，我推开窗子，阳光一下子落在桌面上。我忍不住笑了，心里像被轻轻点亮了一样。"),
    ("surprised",
     "突然手机响了一声，我看到消息的时候愣住了——原来一直等的结果，真的来了。我连呼吸都快了半拍。"),
    ("sad",
     "可下一秒，我又想起曾经错过的人和事。那种酸涩慢慢涌上来，声音也不自觉放轻了，停顿更长了一点。"),
    ("angry",
     "更糟的是，我看到有人把垃圾随手扔在地上。我压着火气，语气变得更硬、更急，却努力不让自己失控。"),
    ("calm",
     "我深呼吸，告诉自己把注意力放回当下。我把垃圾捡起来，慢慢往前走，情绪也重新安定下来。"),
]

PROMPT = {
    "neutral": "情绪：平静自然。语速正常，吐字清晰，按标点自然停顿。不要朗读提示词。 <|endofprompt|>",
    "happy": "情绪：真诚开心。音色更明亮，语调更上扬，节奏更活泼，连读顺畅。不要朗读提示词。 <|endofprompt|>",
    "surprised": "情绪：惊喜兴奋。语速略快，语调起伏更明显，但不尖叫，保持清晰。不要朗读提示词。 <|endofprompt|>",
    "sad": "情绪：低落难过。语速稍慢，音量更轻，尾音下沉，停顿略长。不要朗读提示词。 <|endofprompt|>",
    "angry": "情绪：克制愤怒。能量更高，咬字更硬，语速略快，但不吼叫。不要朗读提示词。 <|endofprompt|>",
    "calm": "情绪：放松释然。语速稍慢，语调平稳，呼吸自然，结尾收住。不要朗读提示词。 <|endofprompt|>",
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref_wav", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--key_prefix", default="demo")
    args = ap.parse_args()

    outp = Path(args.out_jsonl)
    outp.parent.mkdir(parents=True, exist_ok=True)

    with outp.open("w", encoding="utf-8") as f:
        for i, (emo, text) in enumerate(SEGMENTS):
            d = {
                "key": f"{args.key_prefix}_{i:04d}_{emo}",
                "emotion": emo,
                "emotion_text_prompt": PROMPT[emo],
                "neutral_speaker_wav": args.ref_wav,
                "source_text": text,
                "target_text": text,
                "answer_cosyvoice_speech_token": [],
                "target_len": 0,
                "source_len": max(1, len(text)),
            }
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"[INFO] wrote: {outp} (n={len(SEGMENTS)})")

if __name__ == "__main__":
    main()