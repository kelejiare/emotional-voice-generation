import json, os
import matplotlib.pyplot as plt

p = "/data/lizhen/EmoVoice/outputs/runs/ljh_en_Aplus_strong/compare_metrics.json"
with open(p, "r", encoding="utf-8") as f:
    m = json.load(f)

out_dir = os.path.join(m["run_root"], "figs")
os.makedirs(out_dir, exist_ok=True)

methods = list(m["methods"].keys())

def getv(k):
    return [m["methods"][x]["aggregate"][k] for x in methods]

# 1) duration
plt.figure()
plt.bar(methods, getv("mean_duration_sec"))
plt.xticks(rotation=25, ha="right")
plt.ylabel("mean_duration_sec")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "mean_duration.png"))
plt.close()

# 2) silence ratio
plt.figure()
plt.bar(methods, getv("mean_silence_ratio"))
plt.xticks(rotation=25, ha="right")
plt.ylabel("mean_silence_ratio")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "mean_silence_ratio.png"))
plt.close()

# 3) rms db
plt.figure()
plt.bar(methods, getv("mean_rms_db"))
plt.xticks(rotation=25, ha="right")
plt.ylabel("mean_rms_db")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "mean_rms_db.png"))
plt.close()

print("[OK] figs saved to:", out_dir)