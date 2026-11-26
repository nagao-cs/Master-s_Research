"""
two_model_error_estimator.py

grouping.py が算出した GroupingResult（agreed / det_only / trk_only）を使って、
2モデル体制でのFP/FN疑いの残数を推定する。

責務分担:
- grouping.py: モデルの検出とそのトラッカー予測をグルーピングし、
  agreed / det_only / trk_only に分ける（分類のみ）
- Integrator（例: TwoModelTrackIntegrator）: グルーピング結果を使って
  最終的な統合検出結果を構築する（voting・出力の責務）
- このモジュール: グルーピング結果（det_only / trk_only）を使って、
  解消されずに残ったFP/FN疑いの件数を推定する（エラー推定の責務）

前提:
1. 2モデルが一致する検出はたいがいTP
2. 片方のみの検出はTP/FPどちらもありうる
3. 2モデルの一致度と、両モデルの見逃し（common FN）はあまり関係がない

手順:
1. FP疑い（det_only）を2段階で解消する
   a. もう片方のモデルの検出と一致すれば採用（2モデル共通として解消。premise 1）
   b. aで解消されなかったものは、もう片方のモデルのトラッキングと一致すれば採用
2. FN疑い（trk_only）は、もう片方のモデルの検出と一致すれば採用（回収）
3. 1・2で解消されなかった残数をFP/FN疑いの件数として返す
   （フレームがエラーありかどうかの閾値判定は呼び出し側に委ねる）

NOTE: FP疑いを2段階にしている理由は、det_onlyがそのモデル自身の検出であり、
もう片方のモデルの検出とすでに一致していれば premise 1 だけで説明がつくため。
1で解消できなかったものだけが、もう片方のモデルのトラッキングという
別の独立した信号でさらに確認する価値がある。trk_only（FN疑い）はこの時点で
もう片方の検出とまだ一度も比較されていないため、検出との比較のみで十分。
"""
from src.boundingBox.boundingBox import DetectionBoundingBox
from src.BBox_Integrator.grouping import grouping_detections
from src.BBox_Integrator.grouping import GroupingResult


def _has_match(
    box: DetectionBoundingBox,
    candidates: list[DetectionBoundingBox],
    iou_threshold: float,
) -> bool:
    """box が candidates の中に同一クラス・IoU>=閾値の一致相手を持つか判定する"""
    return any(
        c.classId == box.classId and box.computeIoU(c) >= iou_threshold
        for c in candidates
    )


def _resolve_suspects(
    suspects: list[DetectionBoundingBox],
    reference: list[DetectionBoundingBox],
    iou_threshold: float,
) -> list[DetectionBoundingBox]:
    """reference と一致しなかった（解消されなかった）ものだけを返す"""
    return [s for s in suspects if not _has_match(s, reference, iou_threshold)]


def _resolve_fp_suspects(
    grouping_a: GroupingResult,
    grouping_b: GroupingResult,
    model_a_detections: list[DetectionBoundingBox],
    model_b_detections: list[DetectionBoundingBox],
    iou_threshold: float,
) -> list[DetectionBoundingBox]:
    """
    過去の自分の検出またはもう片方のモデルとも一致していなければFP疑いとして出力
    """
    remaining = list()
    detection_groups = grouping_detections({"model_a": model_a_detections, "model_b": model_b_detections}, iou_threshold=iou_threshold)
    for group in detection_groups:
        if len(group) > 1:
            continue

        target = group[0]
        if (target in grouping_a.det_only) or (target in grouping_b.det_only):
            remaining.append(target)
    return remaining


def _resolve_fn_suspects(
    grouping_a: GroupingResult,
    grouping_b: GroupingResult,
    model_a_detections: list[DetectionBoundingBox],
    model_b_detections: list[DetectionBoundingBox],
    iou_threshold: float,
) -> list[DetectionBoundingBox]:
    """"""
    remaining = list()
    for target in grouping_a.trk_only:
        if not _has_match(target, model_b_detections, iou_threshold):
            remaining.append(target)
    for target in grouping_b.trk_only:
        if not _has_match(target, model_a_detections, iou_threshold):
            remaining.append(target)
    return remaining


def estimate_two_model_errors(
    grouping_a: GroupingResult,
    grouping_b: GroupingResult,
    model_a_detections,
    model_a_tracking,
    model_b_detections,
    model_b_tracking,
    iou_threshold: float,
) -> dict[str, int]:
    """
    grouping.py の group_detection_and_tracking() で得た grouping_a / grouping_b
    （それぞれモデルと自身のトラッカーのグルーピング結果）を使い、モデル間の
    突き合わせで解消されずに残ったFP/FN疑いの件数を算出する。

    :return: {"residual_fp_count": int, "residual_fn_count": int}
    """
    residual_fp = _resolve_fp_suspects(
        grouping_a, grouping_b, model_a_detections, model_b_detections, iou_threshold
    )
    residual_fn = _resolve_fn_suspects(grouping_a, grouping_b, model_a_detections, model_b_detections, iou_threshold)

    return {
        "residual_fp_count": len(residual_fp),
        "residual_fn_count": len(residual_fn),
    }