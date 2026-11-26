"""
Trade-off between execution time and detection performance
(fixed multi-version configurations vs. proposed method).

Outputs (per map and aggregated over all maps as "all")
  {MAP}_fig_tradeoff.pdf     F1 score and mAP
  {MAP}_fig_tradeoff_pr.pdf  Precision and Recall
"""
import matplotlib.pyplot as plt
import pandas as pd
from src.config import RESULT_DIR, DATASET_DIR
from src.error_estimate_exp.config import CONFIG_ROOT

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "pdf.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#8a8a85",
    "axes.labelcolor": "#2b2b29",
    "xtick.color": "#5c5c58",
    "ytick.color": "#5c5c58",
})

# Integration methods: color
METHODS = {
    "affirmative": ("#2878B5", "Affirmative"),
    "consensus": ("#E07A2D", "Consensus"),
    "conf_base": ("#26966F", "Conf. base"),
}

# Number of versions: marker shape
MARKERS = {2: "s", 3: "^"}

# (column in fixed-config CSVs, column in proposed result.csv, axis label)
F1_MAP = (("F1_Score", "F1", "F1 Score"), ("mAP", "mAP", "mAP"))
PREC_REC = (("Precision", "prec", "Precision"), ("Recall", "rec", "Recall"))

EXCLUDE = r"rtdetr(?!_torch)|fcos"

MAPS = [f"{seq:04d}" for seq in range(11, 21) if seq != 17]


def n_versions(models: pd.Series) -> pd.Series:
    """'_' 区切りのモデル数。rtdetr_torch 内の '_' は区切りとして数えない。"""
    return models.str.replace("rtdetr_torch", "rtdetrtorch").str.count("_") + 1


def load_results(MAP):
    prop = pd.read_csv(CONFIG_ROOT / f"conf_base_{MAP}/result.csv").iloc[0]

    multi = pd.read_csv(
        DATASET_DIR / "n_version_detection/KITTI/eval_result/results.csv",
        dtype={"map": str},
    )
    multi = multi[
        multi.integrate_way.isin(METHODS)
        & ~multi.models.str.contains(EXCLUDE, case=False, na=False)
        & (multi["map"] == MAP)
    ].copy()
    multi["n"] = n_versions(multi.models)

    single = pd.read_csv(
        DATASET_DIR / f"single_model_detection/KITTI/{MAP}/eval_result/results.csv"
    )
    single = single[
        ~single.model.str.contains(EXCLUDE, case=False, na=False)
    ].copy()
    return single, multi, prop


def aggregate(results):
    """各マップの結果を構成ごとに平均する。"""
    singles, multis, props = zip(*results)
    single = (
        pd.concat(singles)
        .groupby("model", as_index=False)
        .mean(numeric_only=True)
    )
    multi = (
        pd.concat(multis)
        .groupby(["models", "integrate_way", "n"], as_index=False)
        .mean(numeric_only=True)
    )
    prop = pd.DataFrame(props).mean(numeric_only=True)
    return single, multi, prop


def plot_tradeoff(single, multi, prop, panels, out):
    fig, axes = plt.subplots(1, len(panels), figsize=(7.0, 3.0), constrained_layout=True)

    for ax, (col, pcol, label) in zip(axes, panels):
        ax.grid(True, color="#e6e5e0", linewidth=0.6)
        ax.set_axisbelow(True)

        # Single-model baseline
        ax.scatter(
            single.Execution_Time_s, single[col],
            s=30, c="#888888", marker="o",
            edgecolors="#2b2b29", linewidths=0.4,
            label="Single model", zorder=2,
        )

        # Multi-version configurations
        for method, (color, method_name) in METHODS.items():
            for n, marker in MARKERS.items():
                g = multi[(multi.integrate_way == method) & (multi.n == n)]
                if g.empty:
                    continue
                ax.scatter(
                    g.Execution_Time_s, g[col],
                    s=35, c=color, marker=marker,
                    edgecolors="#2b2b29", linewidths=0.4,
                    label=f"{method_name} ({n}V)", zorder=2,
                )

        # Proposed method
        ax.scatter(
            prop.exe_time, prop[pcol],
            s=140, c="#2b2b29", marker="*",
            edgecolors="white", linewidths=0.8,
            label="Proposed method", zorder=3,
        )
        ax.annotate(
            "Proposed method", (prop.exe_time, prop[pcol]),
            xytext=(0, 9), textcoords="offset points",
            ha="center", fontsize=8, color="#2b2b29",
        )

        ax.set_xlabel("Execution Time [s]")
        ax.set_ylabel(label)

    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside upper center", ncol=4, fontsize=7, frameon=False)

    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    plt.close(fig)
    print(out)


def plot_both(single, multi, prop, out_dir, tag):
    plot_tradeoff(single, multi, prop, F1_MAP, out_dir / f"{tag}_fig_tradeoff.pdf")
    plot_tradeoff(single, multi, prop, PREC_REC, out_dir / f"{tag}_fig_tradeoff_pr.pdf")


def main():
    out_dir = RESULT_DIR / "propesed_method"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = [load_results(MAP) for MAP in MAPS]
    for MAP, res in zip(MAPS, results):
        plot_both(*res, out_dir, MAP)

    plot_both(*aggregate(results), out_dir, "all")


if __name__ == "__main__":
    main()