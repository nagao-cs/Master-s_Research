from argparse import ArgumentParser
from pathlib import Path

from .config.config import load_config
from .runner.cache_runner import CacheRunner
from .runner.online_runner import OnlineRunner


def build_runner(mode: str, cfg, base_dir, cfg_path):
    if mode == "cache":
        return CacheRunner(cfg, base_dir, cfg_path)
    elif mode == "online":
        return OnlineRunner(cfg, base_dir, cfg_path)

    raise ValueError(f"Unknown mode: {mode}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Time_aware_Adrod")
    parser.add_argument("--cfg_name", type=str, default="default")
    parser.add_argument("--mode", type=str, default="online")
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent.parent # windowsnoeditor

    cfg_path = (
        base_dir
        / "src"
        / "time_aware_exp"
        / "config"
        / f"{args.cfg_name}"
        / "default.yaml"
    )
    cfg = load_config(cfg_path)

    runner = build_runner(args.mode, cfg, base_dir, cfg_path)
    runner.run()


if __name__ == "__main__":
    main()