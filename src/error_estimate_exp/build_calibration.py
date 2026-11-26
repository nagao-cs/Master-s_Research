"""calibration JSON（例: KITTI.json）を再生成する。

- resolution : summary.csv から (model1, model2) ごとのFP/FN解消率を作り直す
- two_model  : ペアのプロファイルJSON（profilesに2モデル）を追加・上書きする
- one_model など上記以外 : ベースの calibration から引き継ぐ

使い方:
    python build_calibration.py --base KITTI.json --summary summary.csv \
        --pairs yolov8n_rtdetr_torch.json --out KITTI_new.json
"""
import argparse
import copy
import csv
import json
import math
import sys
from pathlib import Path


def load_resolution(csv_path, dataset):
    """summary.csv から解消率を読む。同じペアが複数行あれば後の行を採用し、nanの行は捨てる。"""
    res = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["dataset"].lower() != dataset.lower():
                continue
            fp = float(row["recovery_rate_fp"])
            fn = float(row["recovery_rate_fn"])
            if math.isnan(fp) or math.isnan(fn):
                continue
            res.setdefault(row["model1"], {})[row["model2"]] = {"fp": fp, "fn": fn}
    return res


def load_pair_profiles(paths, dataset):
    """ペアのプロファイルJSONを two_model 形式 {model: {partner: profile}} にまとめる。"""
    two_model, fit_maps = {}, {}
    for path in paths:
        data = json.loads(Path(path).read_text())
        if data["dataset"] != dataset:
            raise ValueError(f"{path}: dataset {data['dataset']} != {dataset}")
        names = list(data["profiles"])
        if len(names) != 2:
            raise ValueError(f"{path}: profiles must contain exactly 2 models, got {names}")
        a, b = names
        two_model.setdefault(a, {})[b] = data["profiles"][a]
        two_model.setdefault(b, {})[a] = data["profiles"][b]
        fit_maps[str(path)] = data.get("fit_maps")
    return two_model, fit_maps


def build(base, resolution, two_model):
    calib = copy.deepcopy(base)
    calib.setdefault("two_model", {})
    for model, partners in two_model.items():
        calib["two_model"].setdefault(model, {}).update(partners)
    calib["resolution"] = resolution
    return calib


def find_warnings(calib, pair_fit_maps):
    warnings = []
    for path, maps in pair_fit_maps.items():
        if maps != calib.get("fit_maps"):
            warnings.append(f"fit_maps mismatch: {path} {maps} vs base {calib.get('fit_maps')}")

    models = set(calib["one_model"]) | set(calib["two_model"]) | set(calib["resolution"])
    for m in sorted(models - set(calib["one_model"])):
        warnings.append(f"one_model missing: {m}")
    for m1, partners in calib["two_model"].items():
        for m2 in partners:
            if m2 not in calib["resolution"].get(m1, {}):
                warnings.append(f"resolution missing: {m1} -> {m2}")
    return warnings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="既存のcalibration JSON")
    ap.add_argument("--summary", required=True, help="解消率のsummary.csv")
    ap.add_argument("--pairs", nargs="*", default=[], help="ペアのプロファイルJSON")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base = json.loads(Path(args.base).read_text())
    dataset = base["dataset"]
    resolution = load_resolution(args.summary, dataset)
    two_model, pair_fit_maps = load_pair_profiles(args.pairs, dataset)
    calib = build(base, resolution, two_model)

    for w in find_warnings(calib, pair_fit_maps):
        print(f"[warn] {w}", file=sys.stderr)

    Path(args.out).write_text(json.dumps(calib, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()