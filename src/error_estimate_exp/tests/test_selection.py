import pytest

from src.error_estimate_exp.frame import Frame
from src.error_estimate_exp.selection import (
    ErrorEstimate,
    Resolution,
    expected_reduction,
    select_model,
)


def test_expected_reduction_sums_over_running_models():
    est = {"a": ErrorEstimate(fp=2, fn=1), "b": ErrorEstimate(fp=0, fn=3)}
    res = {("a", "m"): Resolution(fp=0.5, fn=0.2), ("b", "m"): Resolution(fp=0.9, fn=0.1)}
    assert expected_reduction(est, res, "m") == pytest.approx(2 * 0.5 + 1 * 0.2 + 3 * 0.1)


def test_select_model_divides_by_cost():
    est = {"a": ErrorEstimate(fp=1, fn=1)}
    res = {("a", "big"): Resolution(1.0, 1.0), ("a", "small"): Resolution(0.6, 0.6)}
    cost = {"big": 4.0, "small": 1.0}
    assert select_model(est, ["big", "small"], res, cost.__getitem__) == "small"


def test_select_model_follows_error_type():
    res = {("a", "fp_fixer"): Resolution(0.9, 0.0), ("a", "fn_fixer"): Resolution(0.0, 0.9)}
    cost = {"fp_fixer": 1.0, "fn_fixer": 1.0}.__getitem__
    assert select_model({"a": ErrorEstimate(3, 0)}, ["fp_fixer", "fn_fixer"], res, cost) == "fp_fixer"
    assert select_model({"a": ErrorEstimate(0, 3)}, ["fp_fixer", "fn_fixer"], res, cost) == "fn_fixer"


def test_select_model_prefers_cheaper_on_tie():
    res = {("a", "x"): Resolution(0.5, 0.5), ("a", "y"): Resolution(0.5, 0.5)}
    cost = {"x": 2.0, "y": 1.0}
    assert select_model({"a": ErrorEstimate(0, 0)}, ["x", "y"], res, cost.__getitem__) == "y"


def test_select_model_rejects_empty_pool():
    with pytest.raises(ValueError):
        select_model({}, [], {}, lambda m: 1.0)


def test_frame_runs_each_model_once():
    calls = []
    frame = Frame(lambda m: calls.append(m) or [m], lambda m, d: list(d))
    frame.det("x")
    frame.det("x")
    frame.track("y")
    frame.track("y")
    assert calls == ["x", "y"]
    assert frame.executed == ("x", "y")
