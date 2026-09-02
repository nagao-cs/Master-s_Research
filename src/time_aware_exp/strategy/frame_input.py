from dataclasses import dataclass
import numpy as np


@dataclass
class FrameInput:
    frame_idx: int
    image: np.ndarray | None = None