"""
jaccard_error_relation.pyが出力する1フレーム1行のCSV
    imageId, jaccard, totalGtCount, commonFpCount, commonFnCount,
    oneModelFnCount, oneModelFpCount, maxErrorCount, minErrorCount,
    maxErrorRate, minErrorRate
が {modelA}_{modelB}_{dataset}_{map}.csv という命名で1つのディレクトリに
大量に置かれている状況を想定し、まとめて読み込んで分析する。

やること:
    1. ディレクトリ内の全CSVを読み込み、ファイル名からmodelペア/dataset/mapを復元して結合する
    2. モデルペアごとの集計(件数を合算したerror rate、jaccardとerror rateの相関係数)を出す
       -> "jaccardが低いほどerrorが高い"という関係がどのペアでも成立しているかを確認できる
    3. 全データをプールしてjaccardをbin化し、error rateの分布を可視化する
       -> 1ペアだけよりずっと多いサンプル数で全体傾向を確認できる
    4. ペアごとの相関係数をランキングしたグラフを作る
       -> 関係が一部のペアだけの偶然ではないかを確認できる
    5. commonFp/commonFn/oneModelFn/oneModelFpをそれぞれ個別に同様の分析にかける
       -> maxErrorRate/minErrorRateという合成指標の中でどの成分がjaccardと強く関係しているか、
          またペアによってFN寄り/FP寄り、共通エラー寄り/片方のみのエラー寄りの性質が違うかを見る

各成分のrate化(1フレームごと):
    commonFpRate    = commonFpCount / totalUnit   (totalUnit = totalGtCount + commonFpCount + oneModelFpCount)
    oneModelFpRate  = oneModelFpCount / totalUnit
    commonFnRate    = commonFnCount / totalGtCount
    oneModelFnRate  = oneModelFnCount / totalGtCount
    (FP系は誤検出も含めた"判定対象の総数"を分母に、FN系はGT数を分母にする)

前提:
    ファイル名はモデル名にアンダースコアを含まない
    "{modelA}_{modelB}_{dataset}_{map}.csv" (ちょうど4トークン)を想定している
"""

import argparse
import math
import re
from pathlib import Path

import pandas as pd


FILENAME_PATTERN = re.compile(
    r"^(?P<modelA>[^_]+)_(?P<modelB>[^_]+)_(?P<dataset>[^_]+)_(?P<map>[^_]+)$"
)


# ------------------------------------------------------------
# 読み込み
# ------------------------------------------------------------

def parseFileName(stem: str) -> dict | None:
    """
    "{modelA}_{modelB}_{dataset}_{map}" 形式のファイル名(拡張子抜き)をパースする
    一致しない場合はNoneを返す
    """
    match = FILENAME_PATTERN.match(stem)
    if match is None:
        return None
    return match.groupdict()


# commonFp/oneModelFpの分母(誤検出も含めた判定対象の総数)
# commonFn/oneModelFnの分母(GTオブジェクト数)
FP_RATE_COLUMNS = {"commonFpCount": ("commonTpCount","commonFpRate"), "oneModelFpCount": ("oneModelFnCount","oneModelFpRate")}
FN_RATE_COLUMNS = {"commonFnCount": "commonFnRate", "oneModelFnCount": "oneModelFnRate"}


def _addComponentRateColumns(df: pd.DataFrame) -> pd.DataFrame:
    """
    commonFp/commonFn/oneModelFn/oneModelFpのカウントを、フレームごとにrate化した列を追加する
    """
    totalGt = df["totalGtCount"]

    for countCol, (remain, rateCol) in FP_RATE_COLUMNS.items():
        denominator = df[countCol] + df[remain]
        df[rateCol] = (df[countCol] / denominator).where(denominator > 0)

    for countCol, rateCol in FN_RATE_COLUMNS.items():
        df[rateCol] = (df[countCol] / totalGt).where(totalGt > 0)

    return df


def loadAllFrameCsvs(inputDir: Path) -> pd.DataFrame:
    """
    inputDir配下の全CSVを読み込み、ファイル名から復元したmodelペア/dataset/mapの列を
    付与して1つのDataFrameに結合する
    """
    frames = []
    skipped = []

    for csvPath in sorted(Path(inputDir).glob("*.csv")):
        parsed = parseFileName(csvPath.stem)
        if parsed is None:
            skipped.append(csvPath.name)
            continue

        df = pd.read_csv(csvPath)
        df["modelA"] = parsed["modelA"]
        df["modelB"] = parsed["modelB"]
        df["pair"] = f"{parsed['modelA']}_{parsed['modelB']}"
        df["dataset"] = parsed["dataset"]
        df["map"] = parsed["map"]
        df["sourceFile"] = csvPath.name
        frames.append(df)

    if skipped:
        print(f"命名規則(modelA_modelB_dataset_map.csv)に一致せずスキップしたファイル: {skipped}")

    if not frames:
        raise ValueError(f"{inputDir} に読み込めるCSVが見つかりませんでした")

    combined = pd.concat(frames, ignore_index=True)
    return _addComponentRateColumns(combined)


# ------------------------------------------------------------
# 集計
# ------------------------------------------------------------

def pearsonCorrelation(xs: list[float], ys: list[float]) -> float:
    """外部ライブラリ無しでPearsonの相関係数を計算する"""
    n = len(xs)
    if n < 2:
        return float("nan")

    meanX = sum(xs) / n
    meanY = sum(ys) / n
    covXY = sum((x - meanX) * (y - meanY) for x, y in zip(xs, ys))
    varX = sum((x - meanX) ** 2 for x in xs)
    varY = sum((y - meanY) ** 2 for y in ys)

    denom = math.sqrt(varX * varY)
    return covXY / denom if denom > 0 else float("nan")


def _correlationOnValidRows(df: pd.DataFrame, yCol: str) -> float:
    valid = df.dropna(subset=["jaccard", yCol])
    return pearsonCorrelation(valid["jaccard"].tolist(), valid[yCol].tolist())


def summarizePerPair(df: pd.DataFrame) -> pd.DataFrame:
    """
    モデルペアごとに、件数を合算したerror rateとjaccard-error rateの相関係数を求める
    (フレームごとのrateを単純平均するのではなく、件数を合算してから比率を取る)
    commonFp/commonFn/oneModelFn/oneModelFpについても同様に集計し、
    さらに「エラー全体のうちその成分が占める割合(composition share)」も求める
    """
    rows = []
    for pair, g in df.groupby("pair"):
        totalGt = g["totalGtCount"].sum()
        commonFp = g["commonFpCount"].sum()
        commonFn = g["commonFnCount"].sum()
        oneFn = g["oneModelFnCount"].sum()
        oneFp = g["oneModelFpCount"].sum()

        maxErr = commonFp + commonFn + oneFn + oneFp
        minErr = commonFp + commonFn
        totalUnit = totalGt + commonFp + oneFp
        total_detection = totalUnit - commonFn

        rows.append({
            "pair": pair,
            "nFrames": len(g),
            "nMaps": g["map"].nunique(),
            "meanJaccard": g["jaccard"].mean(),
            "aggregatedMaxErrorRate": maxErr / totalUnit if totalUnit > 0 else float("nan"),
            "aggregatedMinErrorRate": minErr / totalUnit if totalUnit > 0 else float("nan"),
            "corr_jaccard_maxErrorRate": _correlationOnValidRows(g, "maxErrorRate"),
            "corr_jaccard_minErrorRate": _correlationOnValidRows(g, "minErrorRate"),
            # 成分ごとの集計rate(件数合算ベース)
            "aggregatedCommonFpRate": commonFp / total_detection if total_detection > 0 else float("nan"),
            "aggregatedOneModelFpRate": oneFp / total_detection if total_detection > 0 else float("nan"),
            "aggregatedCommonFnRate": commonFn / totalGt if totalGt > 0 else float("nan"),
            "aggregatedOneModelFnRate": oneFn / totalGt if totalGt > 0 else float("nan"),
            # 成分ごとのjaccardとの相関係数(フレーム単位のrate列を使用)
            "corr_jaccard_commonFpRate": _correlationOnValidRows(g, "commonFpRate"),
            "corr_jaccard_oneModelFpRate": _correlationOnValidRows(g, "oneModelFpRate"),
            "corr_jaccard_commonFnRate": _correlationOnValidRows(g, "commonFnRate"),
            "corr_jaccard_oneModelFnRate": _correlationOnValidRows(g, "oneModelFnRate"),
            # エラー全体(maxErr)のうち各成分が占める割合(内訳の性質を見る)
            "share_commonFp": commonFp / maxErr if maxErr > 0 else float("nan"),
            "share_commonFn": commonFn / maxErr if maxErr > 0 else float("nan"),
            "share_oneModelFn": oneFn / maxErr if maxErr > 0 else float("nan"),
            "share_oneModelFp": oneFp / maxErr if maxErr > 0 else float("nan"),
        })

    return pd.DataFrame(rows).sort_values("aggregatedMaxErrorRate")


def binJaccard(jaccard: float, binWidth: float) -> float:
    numBins = round(1.0 / binWidth)
    binIndex = int(jaccard / binWidth)
    binIndex = min(max(binIndex, 0), numBins - 1)
    return round(binIndex * binWidth, 10)


def summarizeByBinPooled(df: pd.DataFrame, binWidth: float) -> pd.DataFrame:
    """
    全ペア・全マップをプールしてjaccardをbin化し、各binのerror rate分布の要約を出す
    """
    d = df.copy()
    d["jaccardBinLower"] = d["jaccard"].apply(lambda j: binJaccard(j, binWidth))

    grouped = d.groupby("jaccardBinLower").agg(
        nFrames=("jaccard", "size"),
        meanMaxErrorRate=("maxErrorRate", "mean"),
        medianMaxErrorRate=("maxErrorRate", "median"),
        meanMinErrorRate=("minErrorRate", "mean"),
        medianMinErrorRate=("minErrorRate", "median"),
        meanCommonFpRate=("commonFpRate", "mean"),
        medianCommonFpRate=("commonFpRate", "median"),
        meanOneModelFpRate=("oneModelFpRate", "mean"),
        medianOneModelFpRate=("oneModelFpRate", "median"),
        meanCommonFnRate=("commonFnRate", "mean"),
        medianCommonFnRate=("commonFnRate", "median"),
        meanOneModelFnRate=("oneModelFnRate", "mean"),
        medianOneModelFnRate=("oneModelFnRate", "median"),
    ).reset_index().sort_values("jaccardBinLower")

    grouped["jaccardBin"] = grouped["jaccardBinLower"].apply(
        lambda b: f"{b:.1f}-{b + binWidth:.1f}"
    )
    return grouped


# ------------------------------------------------------------
# 可視化
# ------------------------------------------------------------

def plotPooledJaccardVsError(
    df: pd.DataFrame,
    outputDir: Path,
    binWidth: float,
    metric: str,
) -> None:
    """全ペア・全マップをプールしたjaccard-bin別の箱ひげ図"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outputDir.mkdir(parents=True, exist_ok=True)

    d = df.dropna(subset=[metric, "jaccard"])
    binned = d["jaccard"].apply(lambda j: binJaccard(j, binWidth))
    uniqueBins = sorted(binned.unique())
    grouped = [d[metric][binned == b].tolist() for b in uniqueBins]
    labels = [f"{b:.1f}-{b + binWidth:.1f}" for b in uniqueBins]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot(grouped, positions=range(len(uniqueBins)), widths=0.6, showfliers=False)
    ax.set_xticks(range(len(uniqueBins)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel(f"jaccard (bin width={binWidth})")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric}")
    fig.tight_layout()

    savePath = outputDir / f"pooled_{metric}_vs_jaccard.png"
    fig.savefig(savePath, dpi=150)
    plt.close(fig)
    print(f"保存しました: {savePath}")


def plotPairCorrelationRanking(
    pairSummary: pd.DataFrame,
    outputDir: Path,
    metric: str,
) -> None:
    """
    ペアごとのjaccard-error rate相関係数をランキング表示する
    (負の相関が多ければ"jaccardが低いほどerrorが高い"がペア横断で成立していると言える)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outputDir.mkdir(parents=True, exist_ok=True)

    s = pairSummary.dropna(subset=[metric]).sort_values(metric)
    if s.empty:
        print(f"{metric}が全てNaNのためランキングプロットをスキップします")
        return

    fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(s))))
    colors = ["tab:red" if v > 0 else "tab:blue" for v in s[metric]]
    ax.barh(s["pair"], s[metric], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(metric)
    ax.set_title("correlation(jaccard, error rate) per model pair\n(blue = negative, expected direction)")
    fig.tight_layout()

    savePath = outputDir / f"pair_ranking_{metric}.png"
    fig.savefig(savePath, dpi=150)
    plt.close(fig)
    print(f"保存しました: {savePath}")


def plotErrorComposition(pairSummary: pd.DataFrame, outputDir: Path) -> None:
    """
    ペアごとに、エラー全体(maxErrorCount)に占めるcommonFp/commonFn/oneModelFn/oneModelFpの
    割合を積み上げ横棒グラフにする
    -> FN寄りかFP寄りか、共通エラー寄りか片方のみのエラー寄りか、ペアごとの性質の違いを見る
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outputDir.mkdir(parents=True, exist_ok=True)

    shareCols = ["share_commonFp", "share_commonFn", "share_oneModelFn", "share_oneModelFp"]
    s = pairSummary.dropna(subset=shareCols).sort_values("aggregatedMaxErrorRate")
    if s.empty:
        print("shareの列が全てNaNのため構成比プロットをスキップします")
        return

    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(s))))
    left = pd.Series(0.0, index=s.index)
    labels = {
        "share_commonFp": "common FP",
        "share_commonFn": "common FN",
        "share_oneModelFn": "one-model FN",
        "share_oneModelFp": "one-model FP",
    }
    colors = {
        "share_commonFp": "tab:red",
        "share_commonFn": "tab:orange",
        "share_oneModelFn": "tab:blue",
        "share_oneModelFp": "tab:cyan",
    }

    for col in shareCols:
        ax.barh(s["pair"], s[col], left=left, label=labels[col], color=colors[col])
        left = left + s[col]

    ax.set_xlabel("share of total error (maxErrorCount)")
    ax.set_title("error composition per model pair")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=4, fontsize=8)
    fig.tight_layout()

    savePath = outputDir / "pair_error_composition.png"
    fig.savefig(savePath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"保存しました: {savePath}")


# ------------------------------------------------------------
# メイン
# ------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="複数のjaccard_error_relation出力CSVをまとめて分析する"
    )
    parser.add_argument("--bin-width", type=float, default=0.1, dest="binWidth")

    args = parser.parse_args()

    from src.config import RESULT_DIR
    inputDir = RESULT_DIR / "jaccard_error_relation"
    outputDir = RESULT_DIR / "jaccard_error_relation/analyze"
    outputDir.mkdir(parents=True, exist_ok=True)

    df = loadAllFrameCsvs(inputDir)
    print(f"読み込んだ行数: {len(df)}  ファイル数: {df['sourceFile'].nunique()}  ペア数: {df['pair'].nunique()}")

    pairSummary = summarizePerPair(df)
    pairSummary.to_csv(outputDir / "pair_summary.csv", index=False)
    print("\n=== ペアごとの集計 (aggregatedMaxErrorRateの昇順) ===")
    print(pairSummary.to_string(index=False))

    binSummary = summarizeByBinPooled(df, args.binWidth)
    binSummary.to_csv(outputDir / "pooled_bin_summary.csv", index=False)
    print("\n=== プールしたjaccard bin別の集計 ===")
    print(binSummary.to_string(index=False))

    COMPONENT_METRICS = [
        "maxErrorRate", "minErrorRate",
        "commonFpRate", "oneModelFpRate", "commonFnRate", "oneModelFnRate",
    ]

    for metric in COMPONENT_METRICS:
        plotPooledJaccardVsError(df, outputDir, args.binWidth, metric=metric)

    CORR_COLUMNS = [
        "corr_jaccard_maxErrorRate", "corr_jaccard_minErrorRate",
        "corr_jaccard_commonFpRate", "corr_jaccard_oneModelFpRate",
        "corr_jaccard_commonFnRate", "corr_jaccard_oneModelFnRate",
    ]
    for corrCol in CORR_COLUMNS:
        plotPairCorrelationRanking(pairSummary, outputDir, metric=corrCol)

    plotErrorComposition(pairSummary, outputDir)

    overall = df.dropna(subset=["jaccard", "maxErrorRate"])
    overallCorrMax = pearsonCorrelation(overall["jaccard"].tolist(), overall["maxErrorRate"].tolist())
    overall2 = df.dropna(subset=["jaccard", "minErrorRate"])
    overallCorrMin = pearsonCorrelation(overall2["jaccard"].tolist(), overall2["minErrorRate"].tolist())

    print(f"\n全データプールでのjaccard-maxErrorRate相関係数: {overallCorrMax:.4f}")
    print(f"全データプールでのjaccard-minErrorRate相関係数: {overallCorrMin:.4f}")

    for corrCol in CORR_COLUMNS:
        nNegative = (pairSummary[corrCol] < 0).sum()
        nTotal = pairSummary[corrCol].notna().sum()
        print(f"{corrCol}が負であるペア: {nNegative}/{nTotal}")

    print(f"\n結果一式を保存しました: {outputDir}")