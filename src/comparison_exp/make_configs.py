"""
make_configs.py
比較手法の config をマップごとに作る（config/<method>_<map>/config.yaml）。
各 config は同じマップの提案手法の config（error_estimate_exp/config/<prefix><map>）を base_config にするので、
test_map・閾値・統合方法・候補は提案手法とそろう。

    python -m src.comparison_exp.make_configs                     # conf_base_* のある全マップ
    python -m src.comparison_exp.make_configs --maps 0008 0020
    python -m src.comparison_exp.make_configs --methods fixed random --overwrite

既にある config は --overwrite を付けない限り上書きしない。
提案手法の結果は error_estimate_exp/config/<base_config>/ にあり、summarize がそこから読むので作らない
（同じ Runner でもう一度動かしたいときは --methods proposed）。
"""
from __future__ import annotations

from argparse import ArgumentParser

import yaml

from src.error_estimate_exp.config import CONFIG_ROOT as EE_CONFIG_ROOT, config_path as ee_config_path

from .config import CONFIG_ROOT, METHODS, _parse_method, read_raw

# 手法ごとの config 本文（base_config 以外）
TEMPLATES: dict[str, str] = {
    "proposed": """\
# 提案手法（エラー推定で追加モデルを選ぶ）
method:
  name: proposed
""",
    "fixed": """\
# モデル選択を固定。状態遷移は提案手法のまま。
# 示せること: 選択することの効果
method:
  name: fixed
  fixed_models: [rtdetr_torch, yolov11n]   # 第 2 モデル, 第 3 モデル
""",
    "random": """\
# モデル選択をランダム（候補から一様）。状態遷移は提案手法のまま。
# 示せること: エラー推定に意味があるか（シードごとに seed_<n>/ に保存し、summarize で平均）
method:
  name: random
  seeds: [0, 1, 2, 3, 4]
""",
    "oracle": """\
# オラクル。正解データを使い、そのフレームで実際に最もエラー（FP + FN）を減らすモデルを選ぶ。
# 状態遷移は提案手法のまま。示せること: 選択で到達できる上限
# 候補の検出は選択のために全部実行するが、cost・state には選んだモデルだけを数える。
# exe_time は候補の分も含むので比較に使わないこと。
method:
  name: oracle
""",
    "recovery_block": """\
# Afrin & Machida のリカバリブロック [2]
# 主ブロックの 2 モデルを常に動かし、一致度が threshold 以下なら主ブロックの出力は使わず
# 予備ブロック単独で検出した出力にする。
method:
  name: recovery_block
  primary: [yolov8n, yolov11n]
  backup: rtdetr_torch
  threshold: 0.5          # theta_low。省略時は base_config の thresholds.theta_low
""",
    "confidence_exit": """\
# Inoue の確信度による Early Exit [13]
# order の先頭モデルから始め、直前に追加したモデルの平均確信度が confidence_threshold 以下なら
# 次のモデルを追加する（最大 3 モデル）。検出が 1 つもないモデルは確信度 0 とみなす。
# 出力は実行した全モデルのアンサンブル（base_config の integrate_way = conf_base）。
method:
  name: confidence_exit
  order: [yolov8n, yolov11n, rtdetr_torch]
  confidence_threshold: 0.5   # theta_p。省略時は base_config の thresholds.tau_p
""",
}
DEFAULT_METHODS = ("fixed", "random", "oracle", "recovery_block", "confidence_exit")


def main():
    parser = ArgumentParser(description="Generate per-map comparison configs")
    parser.add_argument("--maps", nargs="+", default=None, help="省略時は error_estimate_exp に <prefix><map> がある全マップ")
    parser.add_argument("--base_prefix", default="conf_base_", help="base_config 名 = <prefix><map>")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(DEFAULT_METHODS))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    prefix = args.base_prefix
    maps = args.maps or sorted(
        p.parent.name[len(prefix):] for p in EE_CONFIG_ROOT.glob(f"{prefix}*/config.yaml")
    )
    for map_name in maps:
        base = f"{prefix}{map_name}"
        if not (EE_CONFIG_ROOT / base / "config.yaml").exists():
            print(f"skip {map_name}: error_estimate_exp/config/{base}/config.yaml がありません")
            continue
        for method in args.methods:
            path = CONFIG_ROOT / f"{method}_{map_name}" / "config.yaml"
            if path.exists() and not args.overwrite:
                print(f"exists {path.parent.name}")
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"base_config: {base}\n\n{TEMPLATES[method]}{_selection_override(base, method)}",
                            encoding="utf-8")
            print(f"wrote  {path.parent.name}")


def _selection_override(base: str, method: str) -> str:
    """手法が使うモデルが base_config の候補にないときだけ、candidates に足す上書きを書く。"""
    sel = read_raw(ee_config_path(base))["selection"]
    have = {sel["base_model"], *sel["candidates"]}
    needed = _parse_method(yaml.safe_load(TEMPLATES[method])["method"]).models()
    missing = [m for m in needed if m not in have]
    if not missing:
        return ""
    print(f"       {base} の candidates に {missing} がないので追加します")
    cands = ", ".join([*sel["candidates"], *missing])
    return (f"\n# {base} の candidates に {missing} がないため追加（この手法は固定のモデルしか使わない）\n"
            f"selection:\n  candidates: [{cands}]\n")


if __name__ == "__main__":
    main()
