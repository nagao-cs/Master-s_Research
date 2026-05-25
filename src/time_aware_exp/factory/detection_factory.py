import os
from pathlib import Path

from ..strategy.detectionStrategy import (
    CacheDetectionStrategy,
    ModelDetectionStrategy
)

from src.Evaluation.dataset import fileReader



def build_model_detection(cfg):

    infer_fns = {
        cfg.model_1: build_infer_fn(cfg.model_1),
        cfg.model_2: build_infer_fn(cfg.model_2),
        cfg.model_3: build_infer_fn(cfg.model_3),
    }

    return ModelDetectionStrategy(infer_fns)