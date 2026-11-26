"""
main.py
    python -m src.error_estimate_exp.main --cfg_name default --mode online
"""
from argparse import ArgumentParser

from src.config import PROJECT_ROOT

from .config import config_path, load_config
from .runner import OnlineSelectionRunner

RUNNERS = {"online": OnlineSelectionRunner}


def main():
    parser = ArgumentParser(description="Evaluate AdROD with error-estimate-based model selection")
    parser.add_argument("--cfg_name", default="default")
    parser.add_argument("--mode", choices=list(RUNNERS), default="online")
    args = parser.parse_args()

    cfg_path = config_path(args.cfg_name)
    print(cfg_path)
    runner = RUNNERS[args.mode](load_config(cfg_path), PROJECT_ROOT, cfg_path)
    runner.run()


if __name__ == "__main__":
    main()
