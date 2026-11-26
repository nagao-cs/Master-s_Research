# Frame-level FP/FN prediction

`frame_type_error_prediction.py` implements the frame-level rule-based error prediction discussed for AdROD.

## Evidence classes

### One model + one SORT tracker

- class 1: detector-only, low confidence
- class 2: detector-only, high confidence
- class 3: tracker-only
- class 4: detector + tracker

The estimator uses empirical class-specific error probabilities and a frame-level FN leak term.

### Two models + two SORT trackers

For each target model, classes 1--4 are defined from its detector/tracker pair. The other model and other tracker are represented as a support count:

- `s0`: no support from the other side
- `s1`: support from one source
- `s2`: support from both sources

Class 5 represents evidence that exists only on the other side while the target detector and target tracker both have no evidence.

The resulting parameter keys are, for example, `c1_s0`, `c1_s1`, `c3_s2`, and `c5_s2`.

## Calibration and evaluation

The probabilities are not hard-coded A/B/C assumptions. They are estimated from GT on a calibration split. This is intentional: A/B/C should represent empirical model types (high precision / balanced / high recall), rather than arbitrary manually selected probabilities.

Example:

```bash
python -m src.error_estimate.experiments.frame_type_error_prediction \
  --mode one \
  --model yolov8n \
  --model1-type B \
  --dataset KITTI \
  --map 0005 \
  --fit-profile \
  --calibration-ratio 0.3
```

For two models:

```bash
python -m src.error_estimate.experiments.frame_type_error_prediction \
  --mode two \
  --model1 yolov8n --model1-type B \
  --model2 rtdetr --model2-type C \
  --dataset KITTI \
  --map 0005 \
  --fit-profile \
  --calibration-ratio 0.3
```

For a rigorous experiment, fit the profile on a separate sequence and use it for the evaluation sequence. The generated profile JSON can be reused with `--profile`.

## Evaluation output

Each run writes frame-level records containing:

- predicted FP probability
- predicted FN probability
- expected FP/FN counts
- actual FP/FN counts
- dominant error type

It also reports Pearson correlation between predicted expected error counts/probabilities and the actual FP/FN counts. This directly evaluates whether the frame-level estimate tracks the amount of error, not only whether an error exists.

Summary files are separated by model count and shared across runs:

- `RESULT_DIR/frame_type_error_prediction/one_model/summary.csv`
- `RESULT_DIR/frame_type_error_prediction/two_model/summary.csv`

Each subsequent model/run is appended as a new row.
