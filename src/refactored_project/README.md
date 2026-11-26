# Refactored error-estimation modules

`src/error_estimate` is organized as a small reusable library. The experiment runner is kept separate from matching and FP/FN estimation so that alternative methods can be attached without rewriting the experiment loop.

## Structure

```text
src/error_estimate/
├── config.py
├── types.py
├── matching/
│   ├── base.py
│   ├── _hungarian.py
│   ├── iou.py
│   └── shape.py
├── estimation/
│   ├── base.py
│   └── rule_based.py
├── tracking/
│   ├── base.py
│   └── sort.py
├── evaluation/
│   ├── error_evaluator.py
│   └── ground_truth.py
├── io/
│   └── records.py
├── visualization/
│   └── debug.py
└── experiments/
    └── two_model_two_tracker.py
```

## Replacement points

The experiment injects three matchers:

- `model_matcher`: current M1/M2 matching. Default: `IoUOneToOneMatcher`.
- `tracker_matcher`: detection/tracker matching. Default: `ShapeOneToOneMatcher`.
- `evidence_matcher`: other-model/tracker evidence merging. Default: same shape matcher.

Therefore an experiment can replace only one part, e.g.:

```python
experiment = TwoModelTwoTrackerExperiment(
    model_matcher=IoUOneToOneMatcher(0.5),
    tracker_matcher=ShapeOneToOneMatcher(shape_config),
    evidence_matcher=ShapeOneToOneMatcher(shape_config),
)
```

All matchers use Hungarian assignment and therefore preserve strict one-to-one matching.

## Important separation

GT is never used by the estimator. `GroundTruthEvaluator` and `match_prediction_to_gt` are evaluation-only. This keeps the prediction method independent from the evaluation implementation.
