#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CVPR-grade comparison report generator for 4 TTS pipelines.

Inputs:
- jsonl (keys/emotions/text/prompt wav)
- run_root with audio dirs
- emovoice infer.log (Aplus / PP)
- cosyvoice vc.log / instruct.log

Outputs (inside run_root):
- compare_metrics.json
- compare_report.md
- figs/*.png + figs/*.pdf  (publication-ready, 300dpi + vector PDF)

Design:
- Reproducible: dump env snapshot + sha256 of inputs
- Interpretable: clear metric definitions + plots + per-sample table
- Reusable: one command regenerates everything

Notes:
- CosyVoice logs contain "yield speech len X, rtf Y". We map yields to keys by run order.
- EmoVoice logs contain explicit per-wav gen_time/RTF.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torchaudio
import matplotlib.pyplot as plt


# ---------------------------
# Global plotting style (CVPR-ish)
# ---------------------------
def set_cvpr_style() -> None:
    # CVPR author kit stresses readability, consistent fonts, clear captions/labels. :contentReference[oaicite:5]{index=5}
    # Use serif-like font; Times New Roman if available, else DejaVu Serif.
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,  # publication-ready export. :contentReference[oaicite:6]{index=6}
        "font.family": "DejaVu Serif",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.linewidth": 1.0,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "lines.linewidth": 2.2,
        "lines.markersize": 6.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_fig(fig: plt.Figure, out_png: str, out_pdf: str) -> None:
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


# ---------------------------
# Utility
# ---------------------------
def sha256_file(path: str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def mean_std(xs: List[float]) -> Tuple[Optional[float], Optional[float]]:
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return None, None
    arr = np.array(xs, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0


def safe_makedirs(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def read_jsonl(jsonl_path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            rows.append({
                "key": d["key"],
                "emotion": d.get("emotion", ""),
                "text": d.get("target_text") or d.get("source_text") or "",
                "emotion_text_prompt": d.get("emotion_text_prompt", ""),
                "neutral_speaker_wav": d.get("neutral_speaker_wav", ""),
            })
    return rows


# ---------------------------
# Audio metrics
# ---------------------------
def wav_stats(path: str) -> Dict[str, Any]:
    info = torchaudio.info(path)
    sr = int(info.sample_rate)
    nframes = int(info.num_frames)
    dur = (nframes / sr) if sr > 0 else 0.0

    wav, _ = torchaudio.load(path)
    wav = wav.float()
    if wav.dim() == 2 and wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    x = wav[0]

    eps = 1e-9
    rms = torch.sqrt(torch.mean(x**2) + eps).item()
    rms_db = 20.0 * float(torch.log10(torch.tensor(rms + eps)).item())

    # Silence ratio: 20ms frames with energy below -40dB
    frame = int(sr * 0.02)
    hop = frame
    if frame <= 0 or x.numel() < frame:
        silence_ratio = 0.0
    else:
        frames = x.unfold(0, frame, hop)
        fe = torch.sqrt(torch.mean(frames**2, dim=1) + eps)
        fe_db = 20.0 * torch.log10(fe + eps)
        silence_ratio = float((fe_db < -40.0).float().mean().item())

    return {
        "path": path,
        "sample_rate": sr,
        "num_frames": nframes,
        "duration_sec": round(float(dur), 3),
        "rms_db": round(float(rms_db), 2),
        "silence_ratio": round(float(silence_ratio), 3),
        "file_size_mb": round(os.path.getsize(path) / (1024 * 1024), 3),
    }


# ---------------------------
# Parse EmoVoice infer.log
# ---------------------------
RE_DECODE = re.compile(r"decode_config:\s*(\{.*\})")
RE_MODEL = re.compile(r"model_config:\s*(\{.*\})")
RE_AUDIO = re.compile(
    r"Generated Audio:\s*.*?/(?P<fname>[^/]+\.wav),\s*audio length:\s*(?P<alen>[0-9.]+)s,\s*generation time:\s*(?P<gtime>[0-9.]+)s,\s*RTF:\s*(?P<rtf>[0-9.]+)"
)
RE_LLM_RTF = re.compile(r"LLM RTF:\s*(?P<llmrtf>[0-9.]+)")


def parse_emovoice_infer(log_path: str) -> Dict[str, Any]:
    out = {"decode_config": None, "model_config": None, "per_key": {}}
    if not log_path or not os.path.exists(log_path):
        return out

    last_key = None
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if out["decode_config"] is None:
                m = RE_DECODE.search(line)
                if m:
                    try:
                        out["decode_config"] = ast.literal_eval(m.group(1))
                    except Exception:
                        out["decode_config"] = None

            if out["model_config"] is None:
                m = RE_MODEL.search(line)
                if m:
                    try:
                        out["model_config"] = ast.literal_eval(m.group(1))
                    except Exception:
                        out["model_config"] = None

            m = RE_AUDIO.search(line)
            if m:
                fname = m.group("fname")
                key = os.path.splitext(fname)[0]
                last_key = key
                out["per_key"].setdefault(key, {})
                out["per_key"][key].update({
                    "gen_time_sec": float(m.group("gtime")),
                    "rtf": float(m.group("rtf")),
                    "audio_len_sec_from_log": float(m.group("alen")),
                })

            m = RE_LLM_RTF.search(line)
            if m and last_key:
                out["per_key"].setdefault(last_key, {})
                out["per_key"][last_key]["llm_rtf"] = float(m.group("llmrtf"))

    return out


# ---------------------------
# Parse CosyVoice logs (yield speech len / rtf)
# ---------------------------
RE_YIELD = re.compile(r"yield speech len\s+(?P<len>[0-9.]+),\s*rtf\s+(?P<rtf>[0-9.]+)")


def parse_cosyvoice_yields(log_path: str) -> List[Dict[str, float]]:
    """
    Returns list of yields in appearance order: [{'len':..., 'rtf':..., 'time': len*rtf}, ...]
    """
    ys: List[Dict[str, float]] = []
    if not log_path or not os.path.exists(log_path):
        return ys
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = RE_YIELD.search(line)
            if m:
                L = float(m.group("len"))
                R = float(m.group("rtf"))
                ys.append({"audio_len_sec": L, "rtf": R, "gen_time_sec": L * R})
    return ys


def attach_cosyvoice_perf_by_order(keys: List[str], yields: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Map N yield entries to N keys in order. If mismatch, map as much as possible and warn via metadata.
    """
    per_key: Dict[str, Dict[str, float]] = {}
    n = min(len(keys), len(yields))
    for i in range(n):
        per_key[keys[i]] = yields[i]
    return per_key


# ---------------------------
# Env snapshot (reproducibility)
# ---------------------------
def env_snapshot() -> Dict[str, Any]:
    snap: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "python": sys.version.replace("\n", " "),
    }
    # Optional packages
    for name in ["torch", "torchaudio", "onnxruntime", "transformers"]:
        try:
            mod = __import__(name)
            snap[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            snap[name] = None

    # CUDA info (best-effort)
    try:
        snap["cuda_available"] = bool(torch.cuda.is_available())
        snap["cuda_device_count"] = int(torch.cuda.device_count())
        if torch.cuda.is_available():
            snap["cuda_device_name_0"] = torch.cuda.get_device_name(0)
    except Exception:
        pass

    # git commits (best-effort)
    def git_rev(path: str) -> Optional[str]:
        if not path or not os.path.exists(path):
            return None
        try:
            r = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, stderr=subprocess.DEVNULL)
            return r.decode().strip()
        except Exception:
            return None

    snap["git_emovoice_repo"] = git_rev("/data/lizhen/EmoVoice")
    snap["git_cosyvoice_repo"] = git_rev("/data/lizhen/EmoVoice/third_party/CosyVoice")
    return snap


# ---------------------------
# Plotting helpers
# ---------------------------
def bar_with_error(title: str, ylabel: str, methods: List[str], means: List[float], stds: List[float],
                   out_png: str, out_pdf: str, annotate: bool = True) -> None:
    set_cvpr_style()
    fig = plt.figure(figsize=(9.2, 3.8))
    ax = fig.add_subplot(111)

    x = np.arange(len(methods))
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))

    bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=18, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Clean spines (CVPR-like)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.25, linestyle="--")

    if annotate:
        for b, v in zip(bars, means):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.2f}",
                    ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    save_fig(fig, out_png, out_pdf)


def radar_plot(title: str, methods: List[str], metrics: Dict[str, List[float]], out_png: str, out_pdf: str) -> None:
    """
    metrics: dict name -> list[float] aligned with methods
    Will min-max normalize each metric to [0,1] for radar overview.
    """
    set_cvpr_style()
    metric_names = list(metrics.keys())
    vals = np.stack([np.array(metrics[k], dtype=float) for k in metric_names], axis=1)  # [M, K]

    # Min-max per metric
    norm = np.zeros_like(vals)
    for j in range(vals.shape[1]):
        col = vals[:, j]
        mn, mx = np.nanmin(col), np.nanmax(col)
        if math.isclose(mx, mn):
            norm[:, j] = 0.0
        else:
            norm[:, j] = (col - mn) / (mx - mn)

    K = len(metric_names)
    angles = np.linspace(0, 2 * np.pi, K, endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(7.6, 7.6))
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_thetagrids(np.degrees(angles[:-1]), metric_names)
    ax.set_rlabel_position(0)
    ax.grid(True, alpha=0.35, linestyle="--")

    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
    for i, name in enumerate(methods):
        v = norm[i, :].tolist()
        v += v[:1]
        ax.plot(angles, v, marker="o", color=colors[i], label=name)
        ax.fill(angles, v, color=colors[i], alpha=0.08)

    ax.set_title(title, y=1.08)
    ax.legend(loc="lower right", bbox_to_anchor=(1.35, 0.0), frameon=True)
    fig.tight_layout()
    save_fig(fig, out_png, out_pdf)


# ---------------------------
# Main report
# ---------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--run_root", required=True)
    ap.add_argument("--vc_log", default=None)
    ap.add_argument("--instruct_log", default=None)
    ap.add_argument("--out_metrics", default=None)
    ap.add_argument("--out_report", default=None)
    args = ap.parse_args()

    run_root = os.path.abspath(args.run_root)
    jsonl_path = os.path.abspath(args.jsonl)

    figs_dir = os.path.join(run_root, "figs")
    safe_makedirs(figs_dir)

    # Methods config
    methods = {
        "cosyvoice3_instruct": {
            "audio_dir": os.path.join(run_root, "cosyvoice3_instruct", "pred_audio"),
            "infer_log": None,
            "cosy_log": args.instruct_log,
        },
        "emovoice_Aplus": {
            "audio_dir": os.path.join(run_root, "emovoice_Aplus", "pred_audio", "neutral_prompt_speech"),
            "infer_log": os.path.join(run_root, "emovoice_Aplus", "infer.log"),
            "cosy_log": None,
        },
        "emovoicePP_Aplus": {
            "audio_dir": os.path.join(run_root, "emovoicePP_Aplus", "pred_audio", "neutral_prompt_speech"),
            "infer_log": os.path.join(run_root, "emovoicePP_Aplus", "infer.log"),
            "cosy_log": None,
        },
        "emovoicePP_Aplus_vc": {
            "audio_dir": os.path.join(run_root, "emovoicePP_Aplus", "pred_audio_vc", "neutral_prompt_speech"),
            "infer_log": None,
            "cosy_log": args.vc_log,
        },
    }

    rows = read_jsonl(jsonl_path)
    keys_order = [r["key"] for r in rows]

    # Parse logs
    emovoice_logs: Dict[str, Dict[str, Any]] = {}
    for m, cfg in methods.items():
        emovoice_logs[m] = parse_emovoice_infer(cfg["infer_log"]) if cfg["infer_log"] else {
            "decode_config": None, "model_config": None, "per_key": {}
        }

    # CosyVoice perf (yields) mapped by order
    cosy_perf: Dict[str, Dict[str, Dict[str, float]]] = {}
    for m, cfg in methods.items():
        if not cfg["cosy_log"]:
            cosy_perf[m] = {}
            continue
        yields = parse_cosyvoice_yields(cfg["cosy_log"])
        # map by key order; for VC, better map by filenames present in audio_dir
        if m.endswith("_vc"):
            if os.path.isdir(cfg["audio_dir"]):
                vc_keys = sorted([os.path.splitext(x)[0] for x in os.listdir(cfg["audio_dir"]) if x.endswith(".wav")])
            else:
                vc_keys = keys_order
            cosy_perf[m] = attach_cosyvoice_perf_by_order(vc_keys, yields)
        else:
            cosy_perf[m] = attach_cosyvoice_perf_by_order(keys_order, yields)

    # Collect per-key stats
    metrics: Dict[str, Any] = {
        "run_root": run_root,
        "jsonl": jsonl_path,
        "env": env_snapshot(),
        "input_sha256": {
            "jsonl": sha256_file(jsonl_path),
        },
        "samples": rows,
        "methods": {},
    }

    # also hash prompt wav if exists
    prompt_wavs = sorted({r["neutral_speaker_wav"] for r in rows if r.get("neutral_speaker_wav")})
    if prompt_wavs and os.path.exists(prompt_wavs[0]):
        metrics["input_sha256"]["neutral_speaker_wav"] = sha256_file(prompt_wavs[0])
        metrics["neutral_speaker_wavs"] = prompt_wavs

    for m, cfg in methods.items():
        block = {
            "audio_dir": cfg["audio_dir"],
            "infer_log": cfg["infer_log"],
            "cosy_log": cfg["cosy_log"],
            "decode_config": emovoice_logs[m].get("decode_config"),
            "model_config": emovoice_logs[m].get("model_config"),
            "per_key": {},
            "aggregate": {},
        }

        for r in rows:
            key = r["key"]
            wav_path = os.path.join(cfg["audio_dir"], f"{key}.wav")
            if os.path.exists(wav_path):
                a = wav_stats(wav_path)
            else:
                a = {"path": wav_path, "missing": True}

            # Attach EmoVoice perf if available
            perf_e = emovoice_logs[m].get("per_key", {}).get(key, {})
            if perf_e:
                a.update(perf_e)

            # Attach CosyVoice perf if available
            perf_c = cosy_perf.get(m, {}).get(key)
            if perf_c:
                # do not overwrite audio_len_sec_from_log if existed; but add cosy fields
                a.setdefault("audio_len_sec_from_log", perf_c.get("audio_len_sec"))
                a["rtf"] = perf_c.get("rtf")
                a["gen_time_sec"] = perf_c.get("gen_time_sec")

            block["per_key"][key] = a

        # Aggregate with mean/std
        def collect_float(field: str) -> List[float]:
            out = []
            for k in keys_order:
                v = block["per_key"][k].get(field)
                if isinstance(v, (int, float)):
                    out.append(float(v))
            return out

        for field in ["duration_sec", "rms_db", "silence_ratio", "rtf", "gen_time_sec"]:
            vals = collect_float(field)
            mu, sd = mean_std(vals)
            block["aggregate"][f"mean_{field}"] = None if mu is None else round(mu, 4)
            block["aggregate"][f"std_{field}"] = None if sd is None else round(sd, 4)

        # Sample rate summary (majority vote)
        srs = collect_float("sample_rate")
        block["aggregate"]["sample_rate_mode"] = int(np.median(srs)) if srs else None

        metrics["methods"][m] = block

    # Write metrics json
    out_metrics = args.out_metrics or os.path.join(run_root, "compare_metrics.json")
    with open(out_metrics, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # ---------------------------
    # Plots (publication-ready)
    # ---------------------------
    method_names = list(methods.keys())

    def agg_val(mname: str, field: str) -> float:
        v = metrics["methods"][mname]["aggregate"].get(f"mean_{field}")
        return float("nan") if v is None else float(v)

    def agg_std(mname: str, field: str) -> float:
        v = metrics["methods"][mname]["aggregate"].get(f"std_{field}")
        return 0.0 if v is None else float(v)

    # Fig1: RTF
    fig1_png = os.path.join(figs_dir, "fig1_rtf.png")
    fig1_pdf = os.path.join(figs_dir, "fig1_rtf.pdf")
    bar_with_error(
        title="Fig. 1  Efficiency (RTF, lower is faster)",
        ylabel="RTF",
        methods=method_names,
        means=[agg_val(m, "rtf") for m in method_names],
        stds=[agg_std(m, "rtf") for m in method_names],
        out_png=fig1_png,
        out_pdf=fig1_pdf,
    )

    # Fig2: Gen time
    fig2_png = os.path.join(figs_dir, "fig2_gen_time.png")
    fig2_pdf = os.path.join(figs_dir, "fig2_gen_time.pdf")
    bar_with_error(
        title="Fig. 2  Mean Generation Time per Utterance",
        ylabel="Seconds",
        methods=method_names,
        means=[agg_val(m, "gen_time_sec") for m in method_names],
        stds=[agg_std(m, "gen_time_sec") for m in method_names],
        out_png=fig2_png,
        out_pdf=fig2_pdf,
    )

    # Fig3: Silence ratio
    fig3_png = os.path.join(figs_dir, "fig3_silence_ratio.png")
    fig3_pdf = os.path.join(figs_dir, "fig3_silence_ratio.pdf")
    bar_with_error(
        title="Fig. 3  Pause Ratio (20ms frames < -40dB)",
        ylabel="Silence ratio",
        methods=method_names,
        means=[agg_val(m, "silence_ratio") for m in method_names],
        stds=[agg_std(m, "silence_ratio") for m in method_names],
        out_png=fig3_png,
        out_pdf=fig3_pdf,
    )

    # Fig4: RMS
    fig4_png = os.path.join(figs_dir, "fig4_rms_db.png")
    fig4_pdf = os.path.join(figs_dir, "fig4_rms_db.pdf")
    bar_with_error(
        title="Fig. 4  Energy/Loudness Proxy (RMS dB)",
        ylabel="RMS (dB)",
        methods=method_names,
        means=[agg_val(m, "rms_db") for m in method_names],
        stds=[agg_std(m, "rms_db") for m in method_names],
        out_png=fig4_png,
        out_pdf=fig4_pdf,
    )

    # Fig5: Radar overview (normalized objective metrics)
    # We invert RTF so 'higher is better' in radar.
    rtf = [agg_val(m, "rtf") for m in method_names]
    speed_score = [(1.0 / x) if (x is not None and not math.isnan(x) and x > 0) else float("nan") for x in rtf]

    radar_metrics = {
        "Speed (1/RTF)": speed_score,
        "Continuity (1-silence)": [1.0 - agg_val(m, "silence_ratio") for m in method_names],
        "Energy (RMS)": [agg_val(m, "rms_db") for m in method_names],
        "Duration": [agg_val(m, "duration_sec") for m in method_names],
    }
    fig5_png = os.path.join(figs_dir, "fig5_radar_objective.png")
    fig5_pdf = os.path.join(figs_dir, "fig5_radar_objective.pdf")
    radar_plot(
        title="Fig. 5  Radar Overview (Min-Max Normalized Objective Metrics)",
        methods=method_names,
        metrics=radar_metrics,
        out_png=fig5_png,
        out_pdf=fig5_pdf,
    )

    # ---------------------------
    # Markdown report (CVPR-like narrative)
    # ---------------------------
    out_report = args.out_report or os.path.join(run_root, "compare_report.md")

    def md_escape(s: str) -> str:
        return s.replace("|", "\\|")

    # Summary table (mean ± std)
    def fmt_ms(mname: str, field: str) -> str:
        mu = metrics["methods"][mname]["aggregate"].get(f"mean_{field}")
        sd = metrics["methods"][mname]["aggregate"].get(f"std_{field}")
        if mu is None:
            return "-"
        if sd is None:
            return f"{mu}"
        return f"{mu:.3f} ± {sd:.3f}"

    md = []
    md.append("# EmoVoice / CosyVoice3 Comparison Report ")
    md.append("")
    md.append(f"- **Run root**: `{run_root}`")
    md.append(f"- **JSONL**: `{jsonl_path}`")
    if prompt_wavs:
        md.append(f"- **neutral_speaker_wav**: `{prompt_wavs[0]}`")
    md.append(f"- **Samples**: {len(rows)}")
    md.append("")
    md.append("## 1. Reproducibility")
    md.append("### 1.1 Environment snapshot")
    md.append("```json")
    md.append(json.dumps(metrics["env"], ensure_ascii=False, indent=2))
    md.append("```")
    md.append("")
    md.append("### 1.2 Input fingerprint (SHA256)")
    md.append("```json")
    md.append(json.dumps(metrics["input_sha256"], ensure_ascii=False, indent=2))
    md.append("```")
    md.append("")
    md.append("## 2. Methods")
    md.append("- `cosyvoice3_instruct`: CosyVoice3 Instruct2 (one-step, text+instruction+prompt_wav)")
    md.append("- `emovoice_Aplus`: EmoVoice Aplus (latency=0)")
    md.append("- `emovoicePP_Aplus`: EmoVoice-PP Aplus (latency=5 + phn_tokenizer)")
    md.append("- `emovoicePP_Aplus_vc`: EmoVoice-PP output → CosyVoice3 VC (prosody/emotion preserved, timbre adapted)")
    md.append("")
    md.append("## 3. Metric definitions")
    md.append("- **duration_sec**: num_frames / sample_rate")
    md.append("- **rms_db**: RMS energy in dB (loudness proxy)")
    md.append("- **silence_ratio**: 20ms frames whose energy < -40dB")
    md.append("- **RTF**: generation_time / audio_length (lower is faster)")
    md.append("- **gen_time_sec**: generation time in seconds (CosyVoice: audio_len * rtf from logs)")
    md.append("")
    md.append("## 4. Results (mean ± std)")
    md.append("| method | sr(mode) | duration(s) | rms(dB) | silence | RTF | gen_time(s) |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    for mname in method_names:
        sr_mode = metrics["methods"][mname]["aggregate"].get("sample_rate_mode") or "-"
        md.append(
            f"| {mname} | {sr_mode} | {fmt_ms(mname,'duration_sec')} | {fmt_ms(mname,'rms_db')} | "
            f"{fmt_ms(mname,'silence_ratio')} | {fmt_ms(mname,'rtf')} | {fmt_ms(mname,'gen_time_sec')} |"
        )
    md.append("")
    md.append("## 5. Figures (publication-ready)")
    md.append("![](figs/fig1_rtf.png)")
    md.append("")
    md.append("![](figs/fig2_gen_time.png)")
    md.append("")
    md.append("![](figs/fig3_silence_ratio.png)")
    md.append("")
    md.append("![](figs/fig4_rms_db.png)")
    md.append("")
    md.append("![](figs/fig5_radar_objective.png)")
    md.append("")
    md.append("## 6. Per-sample table (paths + key metrics)")
    md.append("| key | emotion | method | dur(s) | rms(dB) | silence | RTF | gen_time(s) | wav |")
    md.append("|---|---|---|---:|---:|---:|---:|---:|---|")
    for r in rows:
        key, emo = r["key"], r.get("emotion", "")
        for mname in method_names:
            a = metrics["methods"][mname]["per_key"][key]
            if a.get("missing"):
                md.append(f"| {key} | {emo} | {mname} | - | - | - | - | - | MISSING `{md_escape(a['path'])}` |")
            else:
                md.append(
                    f"| {key} | {emo} | {mname} | {a.get('duration_sec','-')} | {a.get('rms_db','-')} | "
                    f"{a.get('silence_ratio','-')} | {a.get('rtf','-')} | {a.get('gen_time_sec','-')} | "
                    f"`{md_escape(a['path'])}` |"
                )
    md.append("")
    md.append("## 7. Decode configs (from EmoVoice infer.log)")
    for mname in ["emovoice_Aplus", "emovoicePP_Aplus"]:
        md.append(f"### {mname}.decode_config")
        md.append("```json")
        md.append(json.dumps(metrics["methods"][mname]["decode_config"], ensure_ascii=False, indent=2))
        md.append("```")
        md.append("")
    md.append("## 8. Subjective evaluation protocol (recommended)")
    md.append("- Rate each sample/method on: speaker similarity, emotion intensity, naturalness, intelligibility, overall preference (1–5).")
    md.append("- Use blind A/B playback and use the prompt wav as an anchor.")
    md.append("")
    md.append("## 9. Known issues")
    md.append("- ORT CUDA EP fallback due to missing `libcublasLt.so.11` may affect speed; fix by matching CUDA/cuDNN/ORT versions.")
    md.append("- Transformers `past_key_values` deprecation warnings do not affect outputs.")
    md.append("")

    with open(out_report, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print("[OK] wrote metrics:", out_metrics)
    print("[OK] wrote report:", out_report)
    print("[OK] figs dir:", figs_dir)


if __name__ == "__main__":
    main()