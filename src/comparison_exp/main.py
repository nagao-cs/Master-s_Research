"""
main.py
比較手法を実行する。複数の config を続けて実行すると検出器の読み込みは 1 回で済む。

    python -m src.comparison_exp.main --cfg_name fixed_0020
    python -m src.comparison_exp.main --cfg_name fixed_0020 random_0020
    python -m src.comparison_exp.main --all                                  # config/ 以下すべて
    python -m src.comparison_exp.main --maps 0008 0020                       # そのマップの全手法
    python -m src.comparison_exp.main --methods recovery_block confidence_exit  # その手法の全マップ
    python -m src.comparison_exp.main --maps 0020 --methods fixed random     # 組み合わせ

random は method.seeds のシードごとに config/<cfg_name>/seed_<n>/ に結果を保存する。
config の作成は python -m src.comparison_exp.make_configs、集計は python -m src.comparison_exp.summarize
"""
from argparse import ArgumentParser

from src.config import PROJECT_ROOT

from .config import METHODS, all_config_names, config_path, load_config
from .runner import ComparisonRunner


def select_configs(cfg_names, run_all, maps, methods) -> list[str]:
    if cfg_names and not (run_all or maps or methods):
        return list(cfg_names)
    names = list(cfg_names or []) if not run_all else []
    if run_all or maps or methods:
        for name in all_config_names():
            cfg, method = load_config(config_path(name))
            if maps and cfg.test_map not in maps:
                continue
            if methods and method.name not in methods:
                continue
            names.append(name)
    return list(dict.fromkeys(names))


def run_config(cfg_name: str, shared_models: dict) -> None:
    cfg_path = config_path(cfg_name)
    print(f"\n===== {cfg_name} ({cfg_path}) =====")
    cfg, method = load_config(cfg_path)

    seeds = method.seeds if method.name == "random" else (None,)
    for seed in seeds:
        # シードごとに結果を分ける（BaseRunner は cfg_path の親ディレクトリに保存する）
        out_path = cfg_path if seed is None else cfg_path.parent / f"seed_{seed}" / "config.yaml"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if seed is not None:
            print(f"--- seed {seed} ---")
        ComparisonRunner(cfg, method, PROJECT_ROOT, out_path, seed=seed, shared_models=shared_models).run()


def main():
    parser = ArgumentParser(description="Run comparison methods for AdROD with error-estimate-based selection")
    parser.add_argument("--cfg_name", nargs="+", default=None)
    parser.add_argument("--all", action="store_true", help="config/ 以下のすべての config を実行する")
    parser.add_argument("--maps", nargs="+", default=None, help="test_map で絞り込む")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=None, help="method.name で絞り込む")
    args = parser.parse_args()

    names = select_configs(args.cfg_name, args.all, args.maps, args.methods)
    if not names:
        parser.error("実行する config がありません（--cfg_name / --all / --maps / --methods を指定）")
    print("configs:", " ".join(names))
    shared_models: dict = {}
    for name in names:
        run_config(name, shared_models)


if __name__ == "__main__":
    main()
