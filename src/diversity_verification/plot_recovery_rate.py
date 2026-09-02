"""
recovery_rate_relation.pyが出力するsummary.csv
    timestamp, pair, third, dataset, map,
    total_unresolved, total_resolved, overall_recovery_rate,
    total_unresolved_fn, total_resolved_fn, overall_recovery_rate_fn,
    total_unresolved_fp, total_resolved_fp, overall_recovery_rate_fp,
    jaccard_recovery_correlation
を読み込み、「どのpair(A,B)にどのthirdモデルを追加するとFN/FPの解消率が高いか」を分析する。

やること:
    1. (pair, third, dataset, map)の重複(再実行分)を除き、最新のtimestampを残す
    2. マップ横断で(pair, third)ごとに件数(total_unresolved, total_resolved)を合算し、
       集計版のrecovery rateを求める(1行1行のoverall_recovery_rateを単純平均しない)
       FN由来/FP由来についても同様に合算する
    3. pairごとに、候補thirdモデルをrecovery rateの高い順にランキングする
       -> 「このpairにはこのモデルを追加するのが最も効果的」という結論を出す
    4. thirdモデルごとに、どれだけのpairで高いrecovery rateを出せているかを集計する
       -> 汎用的に効くモデルと、特定pairにだけ効くモデルを区別する
    5. pair x thirdのrecovery rateをヒートマップで可視化する
       全体(recoveryRate)に加え、FN由来(recoveryRateFn)・FP由来(recoveryRateFp)でも
       個別にヒートマップを出す
       -> 「このpairはFNを解消するのが得意なモデルを足すべきか、FPを解消するのが得意な
          モデルを足すべきか」を見分けられる
"""

import argparse
from pathlib import Path

import pandas as pd


# ------------------------------------------------------------
# 読み込み / 集計
# ------------------------------------------------------------

def loadSummary(csvPath: Path) -> pd.DataFrame:
    """summary.csvを読み込み、(pair, third, dataset, map)の重複を除く(最新timestampを残す)"""
    df = pd.read_csv(csvPath)
    df.columns = [c.strip().strip('"') for c in df.columns]
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.strip().str.strip('"')

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = (
        df.sort_values("timestamp")
        .drop_duplicates(subset=["pair", "third", "dataset", "map"], keep="last")
    )
    return df


def aggregatePairThird(df: pd.DataFrame) -> pd.DataFrame:
    """
    マップ横断で(pair, third)ごとにtotal_unresolved/total_resolvedを合算し、
    集計版recovery rateを求める。FN由来/FP由来についても同様に合算する。
    jaccard_recovery_correlationは平均値も併記する
    """
    print(df)
    agg = df.groupby(["pair", "third"]).agg(
        nMaps=("map", "nunique"),
        totalUnresolved=("total_unresolved", "sum"),
        totalResolved=("total_resolved", "sum"),
        totalUnresolvedFn=("total_unresolved_fn", "sum"),
        totalResolvedFn=("total_resolved_fn", "sum"),
        totalUnresolvedFp=("total_unresolved_fp", "sum"),
        totalResolvedFp=("total_resolved_fp", "sum"),
        meanJaccardRecoveryCorrelation=("jaccard_recovery_correlation", "mean"),
        totalBothFn=("total_both_model_fn", "sum"),
        totalResolvedBothFn=("total_resolved_both_model_fn", "sum"),
        totalBothFp=("total_both_model_fp", "sum"),
        totalResolvedBothFp=("total_resolved_both_model_fp", "sum"),
        total_unresolved_both_model=("total_unresolved_both_model", sum),
        total_resolved_both_model=("total_resolved_both_model", sum)
    ).reset_index()

    agg["recoveryRate"] = agg["totalResolved"] / agg["totalUnresolved"].where(agg["totalUnresolved"] > 0)
    agg["recoveryRateFn"] = agg["totalResolvedFn"] / agg["totalUnresolvedFn"].where(agg["totalUnresolvedFn"] > 0)
    agg["recoveryRateFp"] = agg["totalResolvedFp"] / agg["totalUnresolvedFp"].where(agg["totalUnresolvedFp"] > 0)
    agg["recoveryRateBothFn"] = agg["totalResolvedBothFn"] / agg["totalBothFn"].where(agg["totalBothFn"] > 0)
    agg["recoveryRateBothFp"] = agg["totalResolvedBothFp"] / agg["totalBothFp"].where(agg["totalBothFp"] > 0)
    agg["recoveryRateBothModel"] = agg["total_resolved_both_model"] / agg["total_unresolved_both_model"].where(agg["total_unresolved_both_model"] > 0)
    return agg


def rankThirdPerPair(agg: pd.DataFrame) -> pd.DataFrame:
    """pairごとにthirdモデルをrecovery rateの高い順に並べ、順位列を付与する"""
    ranked = agg.sort_values(["pair", "recoveryRate"], ascending=[True, False]).copy()
    ranked["rankWithinPair"] = ranked.groupby("pair")["recoveryRate"].rank(
        method="first", ascending=False
    ).astype(int)
    return ranked


def bestThirdPerPair(ranked: pd.DataFrame) -> pd.DataFrame:
    """pairごとにrecovery rateが最も高いthirdモデルだけを抜き出す"""
    best = ranked[ranked["rankWithinPair"] == 1].copy()
    return best.sort_values("recoveryRate", ascending=False)[
        ["pair", "third", "recoveryRate", "totalUnresolved", "totalResolved", "nMaps"]
    ]


def summarizeThirdModel(agg: pd.DataFrame, ranked: pd.DataFrame) -> pd.DataFrame:
    """
    thirdモデルごとに、
      ・関わったpair数
      ・pair内で1位(最善)になった回数
      ・recovery rateの平均/最小/最大
    を集計する。「汎用的に効くモデル」か「特定pairにしか効かないモデル」かを見分ける
    """
    bestCount = (
        ranked[ranked["rankWithinPair"] == 1]
        .groupby("third").size().rename("timesBest")
    )

    rows = []
    for third, g in agg.groupby("third"):
        rows.append({
            "third": third,
            "nPairsTested": g["pair"].nunique(),
            "timesBest": int(bestCount.get(third, 0)),
            "meanRecoveryRate": g["recoveryRate"].mean(),
            "minRecoveryRate": g["recoveryRate"].min(),
            "maxRecoveryRate": g["recoveryRate"].max(),
        })

    return pd.DataFrame(rows).sort_values("meanRecoveryRate", ascending=False)


# ------------------------------------------------------------
# 可視化
# ------------------------------------------------------------

def plotRecoveryHeatmap(
    agg: pd.DataFrame,
    outputDir: Path,
    valueCol: str = "recoveryRate",
    title: str = "recovery rate: pair x third model",
    filename: str = "recovery_rate_heatmap.png",
) -> None:
    """pair x thirdのrecovery rate(または指定した列)をヒートマップにする(未計測の組み合わせは白抜き)"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    outputDir.mkdir(parents=True, exist_ok=True)

    pivot = agg.pivot(index="pair", columns="third", values=valueCol)

    fig, ax = plt.subplots(figsize=(0.9 * len(pivot.columns) + 3, 0.5 * len(pivot.index) + 2))
    masked = np.ma.masked_invalid(pivot.values)
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad(color="lightgray")

    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("third model")
    ax.set_ylabel("pair (A,B)")
    ax.set_title(title)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if v == v:  # not NaN
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7)

    fig.colorbar(im, ax=ax, label=valueCol)
    fig.tight_layout()

    savePath = outputDir / filename
    fig.savefig(savePath, dpi=150)
    plt.close(fig)
    print(f"保存しました: {savePath}")


def plotBestThirdPerPair(best: pd.DataFrame, outputDir: Path) -> None:
    """pairごとの最善thirdモデルとそのrecovery rateを棒グラフにする"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outputDir.mkdir(parents=True, exist_ok=True)

    s = best.sort_values("recoveryRate")
    labels = [f"{row.pair}  (+{row.third})" for row in s.itertuples()]

    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(s))))
    ax.barh(labels, s["recoveryRate"], color="tab:green")
    ax.set_xlabel("recovery rate (best third model)")
    ax.set_title("best third model to add, per pair")
    ax.set_xlim(0, 1)
    fig.tight_layout()

    savePath = outputDir / "best_third_per_pair.png"
    fig.savefig(savePath, dpi=150)
    plt.close(fig)
    print(f"保存しました: {savePath}")


def plotThirdModelRanking(thirdSummary: pd.DataFrame, outputDir: Path) -> None:
    """thirdモデルごとの平均recovery rateと、1位を取った回数を並べて可視化する"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outputDir.mkdir(parents=True, exist_ok=True)

    s = thirdSummary.sort_values("meanRecoveryRate")

    fig, (axRate, axBest) = plt.subplots(1, 2, figsize=(11, max(4, 0.35 * len(s))))

    axRate.barh(s["third"], s["meanRecoveryRate"], color="tab:blue")
    axRate.set_xlabel("mean recovery rate (across tested pairs)")
    axRate.set_title("third model: mean recovery rate")
    axRate.set_xlim(0, 1)

    axBest.barh(s["third"], s["timesBest"], color="tab:purple")
    axBest.set_xlabel("times ranked #1 for a pair")
    axBest.set_title("third model: times best-for-pair")

    fig.tight_layout()
    savePath = outputDir / "third_model_ranking.png"
    fig.savefig(savePath, dpi=150)
    plt.close(fig)
    print(f"保存しました: {savePath}")


# ------------------------------------------------------------
# メイン
# ------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="recovery_rate_relationのsummary.csvから、pairごとの最適な3モデル目を分析する"
    )

    args = parser.parse_args()

    from src.config import RESULT_DIR
    inputCsv = RESULT_DIR / "recovery_rate_relation/summary.csv"
    outputDir = RESULT_DIR / "recovery_rate_relation/analyze"
    outputDir.mkdir(parents=True, exist_ok=True)

    df = loadSummary(inputCsv)
    print(f"読み込んだ行数: {len(df)}  pair数: {df['pair'].nunique()}  third候補数: {df['third'].nunique()}")

    agg = aggregatePairThird(df)
    agg.to_csv(outputDir / "pair_third_aggregated.csv", index=False)

    ranked = rankThirdPerPair(agg)
    ranked.to_csv(outputDir / "pair_third_ranked.csv", index=False)

    best = bestThirdPerPair(ranked)
    best.to_csv(outputDir / "best_third_per_pair.csv", index=False)

    thirdSummary = summarizeThirdModel(agg, ranked)
    thirdSummary.to_csv(outputDir / "third_model_summary.csv", index=False)

    print("\n=== pairごとの最善third(recovery rateが最も高いモデル) ===")
    print(best.to_string(index=False))

    print("\n=== thirdモデルごとの汎用性(全pair横断) ===")
    print(thirdSummary.to_string(index=False))

    plotRecoveryHeatmap(
        agg, outputDir,
        valueCol="recoveryRate",
        title="recovery rate",
        filename="recovery_rate_heatmap.png",
    )
    plotRecoveryHeatmap(
        agg, outputDir,
        valueCol="recoveryRateFn",
        title="recovery rate (FN only)",
        filename="recovery_rate_heatmap_fn.png",
    )
    plotRecoveryHeatmap(
        agg, outputDir,
        valueCol="recoveryRateFp",
        title="recovery rate (FP only)",
        filename="recovery_rate_heatmap_fp.png",
    )
    plotRecoveryHeatmap(
    agg,
    outputDir,
    valueCol="recoveryRateBothFn",
    title="recovery rate (common FN)",
    filename="recovery_rate_heatmap_common_fn.png",
    )
    plotRecoveryHeatmap(
        agg,
        outputDir,
        valueCol="recoveryRateBothFp",
        title="recovery rate (common FP)",
        filename="recovery_rate_heatmap_common_fp.png",
    )
    plotRecoveryHeatmap(
        agg,
        outputDir,
        valueCol="recoveryRateBothModel",
        title="recovery rate (common Error)",
        filename="recovery_rate_heatmap_common.png",
    )
    plotBestThirdPerPair(best, outputDir)
    plotThirdModelRanking(thirdSummary, outputDir)

    print("\n=== pairごとの全thirdモデルのランキング(上位2件) ===")
    for pair, g in ranked.groupby("pair"):
        top = g.sort_values("rankWithinPair").head(2)
        entries = ", ".join(f"{row.third}({row.recoveryRate:.3f})" for row in top.itertuples())
        print(f"{pair}: {entries}")

    print(f"\n結果一式を保存しました: {outputDir}")