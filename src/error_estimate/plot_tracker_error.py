import matplotlib.pyplot as plt
import pandas as pd

# データ定義（ログの出力順）
data = [
    {"precision": 0.6269, "recall": 0.4144, "f1": 0.4990},
    {"precision": 0.6357, "recall": 0.4128, "f1": 0.5005},
    {"precision": 0.7101, "recall": 0.4027, "f1": 0.5139},
    {"precision": 0.7468, "recall": 0.3859, "f1": 0.5088},
    {"precision": 0.7569, "recall": 0.3658, "f1": 0.4932},
    {"precision": 0.7673, "recall": 0.3154, "f1": 0.4471},
    {"precision": 0.8034, "recall": 0.2399, "f1": 0.3695},
    {"precision": 0.7742, "recall": 0.0805, "f1": 0.1459},
]

df = pd.DataFrame(data[::-1])

# 閾値の設定（実験の実際の値に合わせて調整してください）
df["conf_threshold"] = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# グラフ描画
plt.figure(figsize=(8, 5))

plt.plot(
    df["conf_threshold"],
    df["precision"],
    marker="o",
    linewidth=2,
    color="#1f77b4",
    label="Precision",
)
plt.plot(
    df["conf_threshold"],
    df["recall"],
    marker="s",
    linewidth=2,
    color="#ff7f0e",
    label="Recall",
)
plt.plot(
    df["conf_threshold"],
    df["f1"],
    marker="^",
    linewidth=2,
    color="#2ca02c",
    label="F1-score",
)

plt.title("Metrics vs. Conf Threshold", fontsize=12, fontweight="bold")
plt.xlabel("Confidence Threshold", fontsize=10)
plt.ylabel("Score", fontsize=10)
plt.ylim(0, 1.0)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(loc="best")

plt.tight_layout()
plt.tight_layout()
plt.savefig("/mnt/d/result/tracker_error_prediction/tracker_error_analysis.png", dpi=300)