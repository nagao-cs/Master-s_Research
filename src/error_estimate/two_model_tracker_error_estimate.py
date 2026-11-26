"""
two_model_one_tracker_error_prediction_experiment.py

ベースモデル(M1)のFP/FNを、M1自身のトラッカー(T1)との比較と、
追加モデル(M2)の検出との突き合わせによって推定できるかを検証する。

1モデル+1トラッカーだけの推定（tracker_error_prediction_experiment.py）は
FP推定はある程度機能したが、FN推定の精度が低かった。これは、トラッカー予測
だけを根拠にするとノイズ（位置ずれ等）に引きずられやすいためと考えられる。
本スクリプトでは、追加モデルM2による独立した裏付けを要求することで、
特にFN推定の精度改善を狙う。

手順:
1. 毎フレーム、M1の検出とT1（M1の検出で毎フレーム補正される通常のトラッカー）の
   予測(Dest, 補正前)をグルーピングし、
   det_only（M1のみ検出 = FP疑い）・trk_only（T1のみ予測 = FN疑い）を得る
2. FP疑い（det_only）はM2の検出と一致しなければFP候補として残す
   （T1・M2どちらからも裏付けが得られない検出は、M1固有のFPである可能性が高い）
3. FN疑いは、Tracker-only（trk_only）とM2-onlyをすべてFN候補にする
   （Recallを優先し、どちらか一方だけに現れた検出も候補から除外しない。
   Tracker-onlyとM2-onlyが同一物体を指す場合は1つにまとめる）
4. estimated_fp/estimated_fnそれぞれについて、M1単体のFP/FN（GTとの比較）を
   どれだけ言い当てられているかをオブジェクト単位のprecision/recall/f1で評価する

NOTE: 検出とトラッカー予測・M2検出との対応付けはすべてグローバル1対1割当
（_match_one_to_one）で行う。1本のtrackが複数detectionにマッチする、
または1つのGT側エラーが複数の推定候補に重複カウントされる問題を避けるため。
"""
import argparse
import csv
from pathlib import Path

import cv2

from tqdm import tqdm

from src.boundingBox.boundingBox import ClassifiedBoundingBox, ClassifyCategory, DetectionBoundingBox, BoundingBox
from src.BBox_Integrator.grouping import group_detection_and_tracking

from src.Evaluation.classifier.detectionClassifier import DetectionClassifier
from src.Evaluation.dataset import fileReader

from src.ObjectDetection.models.factory import build_model

from src.util.dataset import build_dataset
from src.config import IOU_THRESHOLD, RESULT_DIR
from src.tracker.tracker import SortTracker


def count_errors(classified_boxes: list[ClassifiedBoundingBox]) -> tuple[int, int]:
    """DetectionClassifier.classify() が返す分類済みboxリストから FP数・FN数 を数える。"""
    fp = sum(1 for b in classified_boxes if b.classifyCategory == ClassifyCategory.FP)
    fn = sum(1 for b in classified_boxes if b.classifyCategory == ClassifyCategory.FN)
    return fp, fn


def _match_one_to_one(
    tracks: list[DetectionBoundingBox],
    detections: list[BoundingBox],
    iou_threshold: float,
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    """
    2つのbox集合をグローバルに1対1対応付けする（同一クラス・IoU>=閾値の候補で
    IoU合計最大の割当をHungarian法で求める）。
    「どれか1つでも一致すればmatch」という方式だと、1本が複数と同時にmatch
    扱いになる問題があるため、ここでは常にこの1対1割当を使う。
    """
    if not tracks or not detections:
        return [], list(range(len(tracks))), list(range(len(detections)))

    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as e:
        raise ImportError("scipy が必要です。`pip install scipy` を実行してください。") from e

    import numpy as np

    n_tracks = len(tracks)
    n_detections = len(detections)
    INVALID_COST = 1e6

    cost = np.full((n_tracks, n_detections), INVALID_COST, dtype=float)
    ious = np.zeros((n_tracks, n_detections), dtype=float)

    for i, track in enumerate(tracks):
        for j, detection in enumerate(detections):
            if track.classId != detection.classId:
                continue
            iou = track.computeIoU(detection)
            ious[i, j] = iou
            if iou >= iou_threshold:
                cost[i, j] = 1.0 - iou

    row_ind, col_ind = linear_sum_assignment(cost)

    matches = []
    matched_tracks = set()
    matched_detections = set()
    for i, j in zip(row_ind, col_ind):
        if cost[i, j] >= INVALID_COST:
            continue
        matches.append((int(i), int(j), float(ious[i, j])))
        matched_tracks.add(int(i))
        matched_detections.add(int(j))

    unmatched_tracks = [i for i in range(n_tracks) if i not in matched_tracks]
    unmatched_detections = [j for j in range(n_detections) if j not in matched_detections]

    return matches, unmatched_tracks, unmatched_detections


class TwoModelOneTrackerErrorPredictionExperiment:
    """ベースモデル(M1)のFP/FNを、M1のトラッカーと追加モデルM2の検出から推定できるかを検証する"""

    def __init__(
        self,
        base_model_name: str = "yolov8n",
        aux_model_name: str = "ssd",
        dataset_name: str = "KITTI",
        map_name: str = "0001",
        iou_threshold: float = IOU_THRESHOLD,
        cross_check_iou: float = 0.3,
        prediction_frames: int = 1,
    ):
        self.base_model_name = base_model_name
        self.aux_model_name = aux_model_name
        self.dataset_name = dataset_name
        self.map_name = map_name
        self.iou_threshold = iou_threshold
        # M2による裏付けチェック用のIoU閾値。位置ずれを考慮しGT評価用より緩められる
        self.cross_check_iou = cross_check_iou
        self.prediction_frames = prediction_frames
        if self.prediction_frames != 1:
            raise ValueError("現在は1フレーム予測のみ対応しています。--prediction-frames 1 を指定してください。")

        self.dataset = build_dataset(dataset_name=dataset_name, map_name=map_name)
        self.base_model = build_model(model_name=base_model_name, dataset=dataset_name, device="cuda")
        self.aux_model = build_model(model_name=aux_model_name, dataset=dataset_name, device="cuda")
        # M1専用の通常運用トラッカー（毎フレーム検出で補正する）
        self.tracker = SortTracker()
        self.classifier = DetectionClassifier(iouThreshold=iou_threshold)

        self.output_dir = (
            RESULT_DIR
            / f"two_model_one_tracker_error_prediction/{dataset_name}/{map_name}/{base_model_name}_{aux_model_name}"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.debug_dir = self.output_dir / "debug_images"
        self.debug_dir.mkdir(parents=True, exist_ok=True)

        self.frame_records: list[dict] = []

    def run(self):
        print(
            f"Base model (M1): {self.base_model_name}, "
            f"Aux model (M2): {self.aux_model_name}, "
            f"Dataset: {self.dataset_name}, map: {self.map_name}, "
            f"prediction_frames: {self.prediction_frames}, "
            f"cross_check_iou: {self.cross_check_iou}"
        )
        print("Both M1 and M2 are executed on every frame.")
        print(f"Total frames: {len(self.dataset)}")

        for frame_idx, (image_path, _label_path) in enumerate(tqdm(self.dataset)):
            self._process_frame(frame_idx, image_path)

        self._save_frame_records()
        self._evaluate()

    def _process_frame(self, frame_idx: int, image_path: Path):
        # M1 と M2 の両方を毎フレーム実行
        base_detections = self.base_model.predict(image_path)
        aux_detections = self.aux_model.predict(image_path)

        # T1 は M1 専用。SortTracker はフレームをまたいで状態を保持する。
        # update() 内では、現在の M1 検出で補正する「前」の予測状態を
        # predicted_boxes として返すため、これを 1 フレーム先の予測として使う。
        tracking_result = self.tracker.update(base_detections)
        dest = tracking_result.predicted_boxes

        # Step 1: M1とT1をグルーピングし、FP疑い(det_only)・FN疑い(trk_only)を得る
        grouping = group_detection_and_tracking(
            model_detections=base_detections, tracking_detections=dest, iou_threshold=self.iou_threshold
        )

        # Step 2: FP疑いはM2の検出と一致しなければFP候補として残す
        _, unmatched_det_only, _ = _match_one_to_one(
            tracks=grouping.det_only, detections=aux_detections, iou_threshold=self.cross_check_iou
        )
        estimated_fp = [grouping.det_only[i] for i in unmatched_det_only]

        # Step 3: FN候補は「Trackerのみ」と「M2のみ」の両方をすべて含める
        #
        #  - Tracker-only: M1にはなく、Trackerが予測している物体
        #  - M2-only: M1にはなく、M2が検出している物体
        #
        # Recallを優先するため、Tracker/M2のどちらか一方だけに現れた候補を
        # すべてFN候補として扱う。両者が同じ物体を指す場合は1つにまとめる。
        _, _, unmatched_m2_only = _match_one_to_one(
            tracks=base_detections,
            detections=aux_detections,
            iou_threshold=self.cross_check_iou,
        )
        m2_only = [aux_detections[i] for i in unmatched_m2_only]

        estimated_fn = list(grouping.trk_only)
        _, _, unmatched_m2_candidates = _match_one_to_one(
            tracks=estimated_fn,
            detections=m2_only,
            iou_threshold=self.cross_check_iou,
        )
        estimated_fn.extend(m2_only[i] for i in unmatched_m2_candidates)

        # GT と比較して、実際のM1のFP/FNとどれだけ一致するかを確認する
        gt_filename = image_path.stem + ".txt"
        gt_path = self.dataset.label_dir / gt_filename
        if gt_path.exists():
            gt_boxes = fileReader.convertGroundTruthFileToBoundingBoxList(str(gt_path))
        else:
            gt_boxes = []

        classified = self.classifier.classify(gt_boxes, base_detections)

        fps = [b for b in classified if b.classifyCategory == ClassifyCategory.FP]
        true_estimate_fp = sum(1 for b in estimated_fp if b in fps)

        fns = [b for b in classified if b.classifyCategory == ClassifyCategory.FN]
        _, unmatched_estimated_fn, _ = _match_one_to_one(
            tracks=estimated_fn, detections=fns, iou_threshold=self.iou_threshold
        )
        true_estimate_fn = len(estimated_fn) - len(unmatched_estimated_fn)

        fp_count, fn_count = count_errors(classified)

        self.frame_records.append({
            "frame_idx": frame_idx,
            "fp_estimate": len(estimated_fp),
            "fn_estimate": len(estimated_fn),
            "true_estimate_fp": true_estimate_fp,
            "true_estimate_fn": true_estimate_fn,
            "fp_count": fp_count,
            "fn_count": fn_count,
        })

        self._save_debug_image(
            image_path=image_path,
            base_detections=base_detections,
            aux_detections=aux_detections,
            tracker_predictions=dest,
            estimated_fp=estimated_fp,
            estimated_fn=estimated_fn,
            frame_idx=frame_idx,
        )

    @staticmethod
    def _box_to_xyxy(box, width, height):
        x1 = int((box.xCenter - box.width / 2) * width)
        y1 = int((box.yCenter - box.height / 2) * height)
        x2 = int((box.xCenter + box.width / 2) * width)
        y2 = int((box.yCenter + box.height / 2) * height)
        return x1, y1, x2, y2

    def _draw_boxes(self, image, boxes, color, label):
        h, w = image.shape[:2]
        for box in boxes:
            x1, y1, x2, y2 = self._box_to_xyxy(box, w, h)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                image, label, (x1, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
            )

    def _save_debug_image(
        self, image_path, base_detections, aux_detections,
        tracker_predictions, estimated_fp, estimated_fn, frame_idx
    ):
        """M1/M2/Tracker/FP/FN候補を1枚に重ねて保存する。"""
        image = cv2.imread(str(image_path))
        if image is None:
            return

        # BGR: M1=blue, M2=green, tracker=red, FP=yellow, FN=magenta
        self._draw_boxes(image, base_detections, (255, 0, 0), "M1")
        self._draw_boxes(image, aux_detections, (0, 255, 0), "M2")
        self._draw_boxes(image, tracker_predictions, (0, 0, 255), "TRK")
        self._draw_boxes(image, estimated_fp, (0, 255, 255), "EST-FP")
        self._draw_boxes(image, estimated_fn, (255, 0, 255), "EST-FN")

        cv2.putText(
            image,
            f"frame={frame_idx}  M1={len(base_detections)} M2={len(aux_detections)} "
            f"TRK={len(tracker_predictions)} EST-FP={len(estimated_fp)} EST-FN={len(estimated_fn)}",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA
        )
        cv2.putText(
            image,
            f"frame={frame_idx}  M1={len(base_detections)} M2={len(aux_detections)} "
            f"TRK={len(tracker_predictions)} EST-FP={len(estimated_fp)} EST-FN={len(estimated_fn)}",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA
        )

        cv2.imwrite(str(self.debug_dir / f"{image_path.stem}.jpg"), image)

    def _save_frame_records(self):
        if not self.frame_records:
            return
        path = self.output_dir / "frame_records.csv"
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.frame_records[0].keys()))
            writer.writeheader()
            writer.writerows(self.frame_records)
        print(f"Frame-level records saved to {path}")

    def _precision_recall_f1(self, estimated_key: str, true_key: str, actual_key: str) -> dict:
        """オブジェクト単位での推定精度を全フレーム合算で算出する。"""
        total_estimated = sum(r[estimated_key] for r in self.frame_records)
        total_true = sum(r[true_key] for r in self.frame_records)
        total_actual = sum(r[actual_key] for r in self.frame_records)

        precision = total_true / total_estimated if total_estimated else 0.0
        recall = total_true / total_actual if total_actual else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        return {
            "estimated": total_estimated,
            "true_positive": total_true,
            "actual": total_actual,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    def _evaluate(self):
        """FP推定・FN推定それぞれについて、オブジェクト単位の精度を独立に評価する。"""
        fp_stats = self._precision_recall_f1("fp_estimate", "true_estimate_fp", "fp_count")
        fn_stats = self._precision_recall_f1("fn_estimate", "true_estimate_fn", "fn_count")

        summary = {
            "base_model": self.base_model_name,
            "aux_model": self.aux_model_name,
            "dataset": self.dataset_name,
            "cross_check_iou": self.cross_check_iou,
            "prediction_frames": self.prediction_frames,
            "n_frames": len(self.frame_records),
        }
        for prefix, stats in (("fp", fp_stats), ("fn", fn_stats)):
            for key, value in stats.items():
                summary[f"{prefix}_{key}"] = value

        summary_path = RESULT_DIR / "two_model_one_tracker_error_prediction" / "summary.csv"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = summary_path.exists()
        with open(summary_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(summary)

        print(f"\nSummary saved to {summary_path}")
        for k, v in summary.items():
            print(f"{k}: {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="2モデル+1トラッカー（ベースモデル+追加モデル+ベースのトラッカー）によるFP/FN予測の妥当性検証"
    )
    parser.add_argument("--base-model", type=str, default="yolov8n")
    parser.add_argument("--aux-model", type=str, default="ssd")
    parser.add_argument("--dataset", type=str, default="KITTI", choices=["CARLA", "KITTI"])
    parser.add_argument("--map", type=str, default="0001")
    parser.add_argument(
        "--prediction-frames",
        type=int,
        default=1,
        help="トラッカー予測の先読みフレーム数。現在は1のみ対応。",
    )
    parser.add_argument(
        "--cross-check-iou",
        type=float,
        default=0.3,
        help="M2による裏付けチェックのIoU閾値（デフォルト0.3）",
    )
    args = parser.parse_args()

    experiment = TwoModelOneTrackerErrorPredictionExperiment(
        base_model_name=args.base_model,
        aux_model_name=args.aux_model,
        dataset_name=args.dataset,
        map_name=args.map,
        cross_check_iou=args.cross_check_iou,
        prediction_frames=args.prediction_frames,
    )
    experiment.run()