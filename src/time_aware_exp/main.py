from argparse import ArgumentParser
from pathlib import Path

from .config.config import load_config
from .runner.cache_runner import CacheRunner
from .runner.online_runner import OnlineRunner


def parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["cache", "online"],
        default="cache"
    )

    return parser.parse_args()


def build_runner(args, cfg, base_dir, cfg_path):

    if args.mode == "cache":
        return CacheRunner(cfg, base_dir, cfg_path)

    elif args.mode == "online":
        return OnlineRunner(cfg, base_dir, cfg_path)

    raise ValueError(f"Unknown mode: {args.mode}")


def main():
    base_dir = Path(__file__).parent.parent.parent # windowsnoeditor

    cfg_path = (
        base_dir
        / "src"
        / "time_aware_exp"
        / "config"
        / "default"
        / "default.yaml"
    )

    cfg = load_config(cfg_path)

    args = parse_args()

    runner = build_runner(args, cfg, base_dir, cfg_path)

    runner.run()


if __name__ == "__main__":
    main()