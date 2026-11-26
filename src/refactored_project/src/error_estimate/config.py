from dataclasses import dataclass


@dataclass(frozen=True)
class ShapeMatcherConfig:
    max_distance: float = 2.5
    max_center_distance: float = 2.0
    max_size_ratio: float = 2.5
    max_aspect_ratio: float = 2.0
    center_weight: float = 1.0
    size_weight: float = 0.75
    ratio_weight: float = 0.75


@dataclass(frozen=True)
class ErrorPredictionConfig:
    low_confidence_threshold: float = 0.5
    dominant_margin: float = 0.0
