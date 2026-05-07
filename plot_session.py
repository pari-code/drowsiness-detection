import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
from pathlib import Path

# Use latest session if no path provided
if len(sys.argv) > 1:
    csv_path = sys.argv[1]
else:
    sessions = sorted(Path("outputs/sessions").glob("*.csv"))
    if not sessions:
        print("No session files found in outputs/sessions/")
        sys.exit(1)
    csv_path = sessions[-1]

df = pd.read_csv(csv_path)
print(f"Loaded: {csv_path}  ({len(df):,} frames)")

fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
fig.suptitle(f"Session analysis — {Path(csv_path).stem}", fontsize=12)

frames = df["frame"]

# Plot 1 — EAR
axes[0].plot(frames, df["ear"], color="#2196F3", linewidth=0.8, label="EAR")
axes[0].axhline(0.25, color="red", linestyle="--", linewidth=0.8, label="threshold")
axes[0].set_ylabel("EAR"); axes[0].legend(fontsize=8); axes[0].set_ylim(0, 0.5)

# Plot 2 — Drowsy probability
axes[1].fill_between(frames, df["drowsy_prob"],
                      alpha=0.5, color="#E91E63")
axes[1].axhline(0.70, color="red", linestyle="--", linewidth=0.8)
axes[1].set_ylabel("Drowsy prob"); axes[1].set_ylim(0, 1)

# Plot 3 — Head pitch
axes[2].plot(frames, df["pitch"], color="#FF9800", linewidth=0.8)
axes[2].axhline(20, color="red", linestyle="--", linewidth=0.8)
axes[2].set_ylabel("Head pitch (°)")

# Plot 4 — Alarm events
axes[3].fill_between(frames, df["alarm"],
                      alpha=0.8, color="#F44336", label="Alarm active")
axes[3].set_ylabel("Alarm")
axes[3].set_xlabel("Frame")
axes[3].set_ylim(-0.1, 1.5)

plt.tight_layout()
out = f"outputs/session_timeline.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved → {out}")