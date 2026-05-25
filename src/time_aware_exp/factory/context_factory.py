from ..state.stateContext import StateContext
from ..state.AdrodState import SingleState

from ..tracker.tracker import SortTracker

from ..config.config import AdrodConfig

from src.boundingBox.integrator.affirmativeIntegrator import (
    ConfidenceBaseIntegrator
)

from src.boundingBox.integrator.majorityIntegrator import (
    MajorityIntegrator
)


def build_integrator(cfg: AdrodConfig):
    if cfg.integrate_way == "affirmative":
        return ConfidenceBaseIntegrator(
            iouThreshold=0.5,
            confidenceThreshold=0.0
        )
    elif cfg.integrate_way == "conf_base":

        return ConfidenceBaseIntegrator(
            iouThreshold=0.5,
            confidenceThreshold=cfg.thresholds.tau_p
        )
    elif cfg.integrate_way == "consensus":
        return MajorityIntegrator(
            iouThreshold=0.5,
            maxVersion=3
        )
    raise ValueError(
        f"Unknown integrate_way: {cfg.integrate_way}"
    )


def build_context(cfg: AdrodConfig):
    tracker = SortTracker(cfg.sort)
    integrator = build_integrator(cfg)

    return StateContext(
        thresholds=cfg.thresholds,
        integrator=integrator,
        tracker=tracker,
        m1=cfg.model_1,
        m2=cfg.model_2,
        m3=cfg.model_3,
        initial_state=SingleState(),
    )