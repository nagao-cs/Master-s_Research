"""
summarize.py
比較手法の結果をマップごと・手法ごとの表にまとめる。

    python -m src.comparison_exp.summarize
    python -m src.comparison_exp.summarize --maps 0008 0020
    python -m src.comparison_exp.summarize --cfg_name fixed_0020 random_0020

読む結果:
    比較手法  comparison_exp/config/<cfg_name>/result.csv（random は seed_<n>/ の平均と標準偏差）
    提案手法  各 config の base_config（error_estimate_exp/config/<base_config>/result.csv）
              ※ comparison_exp に proposed の config があればそちらを使う

出力:
    summary.csv            1 行 = 1 マップ × 1 手法
        config / map / method / n_runs / <result.csv の各列>（n_runs > 1 なら <列>_std も）
        models_used  selected_models.csv から、各モデルを実行したフレーム数（シード平均）
    summary_by_method.csv  1 行 = 1 手法。summary.csv の各列のマップ平均（n_maps = マップ数）
"""
from __future__ import annotations

import csv
import math
from argparse import ArgumentParser
from collections import Counter
from pathlib import Path

from src.error_estimate_exp.config import config_path as ee_config_path

from .config import COMPARISON_EXP_ROOT, all_config_names, base_config_name, config_path, load_config

METHOD_ORDER = ["proposed", "fixed", "random", "oracle", "recovery_block", "confidence_exit"]


def _runs(cfg_dir: Path) -> list[Path]:
    """結果のあるディレクトリ（cfg_dir 自体か seed_<n>/）。"""
    seeds = sorted(p.parent for p in cfg_dir.glob("seed_*/result.csv"))
    if seeds:
        return seeds
    return [cfg_dir] if (cfg_dir / "result.csv").exists() else []


def _read_result(run_dir: Path) -> dict[str, float]:
    with (run_dir / "result.csv").open(newline="") as f:
        row = list(csv.DictReader(f))[-1]  # 追記されている場合は最新の行
    out = {}
    for k, v in row.items():
        try:
            out[k] = float(v)
        except (TypeError, ValueError):
            out[k] = float("nan")
    return out


def _models_used(run_dir: Path) -> Counter:
    path = run_dir / "selected_models.csv"
    counts: Counter = Counter()
    if path.exists():
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                counts.update(m for m in row["models"].split("+") if m)
    return counts


def _mean_std(values: list[float]) -> tuple[float, float]:
    vals = [v for v in values if not math.isnan(v)]
    if not vals:
        return float("nan"), float("nan")
    mean = sum(vals) / len(vals)
    std = math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) if len(vals) > 1 else 0.0
    return mean, std


def summarize_dir(cfg_dir: Path, name: str, map_name: str, method: str) -> dict | None:
    runs = _runs(cfg_dir)
    if not runs:
        print(f"skip {name}: result.csv がありません")
        return None
    results = [_read_result(r) for r in runs]

    row: dict = {"config": name, "map": map_name, "method": method, "n_runs": len(runs)}
    for col in results[0]:
        mean, std = _mean_std([r.get(col, float("nan")) for r in results])
        row[col] = mean
        if len(runs) > 1:
            row[f"{col}_std"] = std

    total: Counter = Counter()
    for r in runs:
        total.update(_models_used(r))
    row["models_used"] = " ".join(f"{m}:{c / len(runs):.0f}" for m, c in total.most_common())
    return row


def summarize(cfg_names: list[str], maps: list[str] | None = None) -> list[dict]:
    rows = []
    proposed_bases: dict[str, str] = {}  # map -> 提案手法の config 名（base_config）
    for name in cfg_names:
        cfg, method = load_config(config_path(name))
        if maps and cfg.test_map not in maps:
            continue
        if method.name != "proposed":
            proposed_bases.setdefault(cfg.test_map, base_config_name(config_path(name)))
        row = summarize_dir(config_path(name).parent, name, cfg.test_map, method.name)
        if row:
            rows.append(row)

    have_proposed = {r["map"] for r in rows if r["method"] == "proposed"}
    for map_name, base in sorted(proposed_bases.items()):
        if map_name not in have_proposed:
            row = summarize_dir(ee_config_path(base).parent, base, map_name, "proposed")
            if row:
                rows.append(row)

    def key(r):
        m = r["method"]
        return r["map"], METHOD_ORDER.index(m) if m in METHOD_ORDER else len(METHOD_ORDER)

    return sorted(rows, key=key)


def by_method(rows: list[dict]) -> list[dict]:
    """手法ごとのマップ平均（全マップにそろっていない手法は n_maps / maps で分かる）。"""
    out = []
    methods = sorted({r["method"] for r in rows}, key=lambda m: METHOD_ORDER.index(m) if m in METHOD_ORDER else 99)
    for method in methods:
        group = [r for r in rows if r["method"] == method]
        row: dict = {"method": method, "n_maps": len(group), "maps": " ".join(r["map"] for r in group)}
        cols = dict.fromkeys(c for r in group for c, v in r.items()
                             if isinstance(v, float) and not c.endswith("_std"))
        for col in cols:
            row[col] = _mean_std([r.get(col, float("nan")) for r in group])[0]
        out.append(row)
    return out


def _write(path: Path, rows: list[dict]) -> None:
    header: list[str] = []
    for r in rows:
        header += [k for k in r if k not in header]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def _print(rows: list[dict], shown: list[str]) -> None:
    print(" ".join(f"{c:>15}" for c in shown))
    for r in rows:
        cells = []
        for c in shown:
            v = r.get(c, "")
            cells.append(f"{v:>15.4f}" if isinstance(v, float) else f"{v!s:>15}")
        print(" ".join(cells))


def main():
    parser = ArgumentParser(description="Summarize comparison results")
    parser.add_argument("--cfg_name", nargs="+", default=None, help="省略時は config/ 以下すべて")
    parser.add_argument("--maps", nargs="+", default=None)
    parser.add_argument("--out_dir", type=Path, default=COMPARISON_EXP_ROOT)
    args = parser.parse_args()

    rows = summarize(args.cfg_name or all_config_names(), args.maps)
    if not rows:
        print("集計できる結果がありません")
        return
    methods = by_method(rows)
    _write(args.out_dir / "summary.csv", rows)
    _write(args.out_dir / "summary_by_method.csv", methods)

    metrics = ["mAP", "Cov_OD", "Cer_OD", "F1", "cost", "state_1", "state_2", "state_3"]
    _print(rows, ["map", "method", "n_runs", *metrics])
    print("\n--- マップ平均 ---")
    _print(methods, ["method", "n_maps", *metrics])
    print(f"\nSaved to {args.out_dir / 'summary.csv'} and {args.out_dir / 'summary_by_method.csv'}")


if __name__ == "__main__":
    main()
