"""
tracker_error_prediction_experiment.py

「検出で補正せずhistory_windowフレーム分だけ予測を進めたトラッカーの位置」と
「実際の検出」を毎フレーム比較し、FP候補・FN候補を抽出する。

方式（history_window=2の場合）:
  実検出で毎フレーム補正し続けるメイントラッカーを1つだけ保持する。
  各フレームで補正後のメイントラッカーをdeepcopyしてシャドートラッカーを作り、
  シャドー側だけは以降の検出を与えずhistory_windowフレーム先まで予測する。
  これによりシャドーは、作成時点まで蓄積した速度・Kalman stateを引き継ぐ。

  フレーム0: mainをD0で補正 -> S0=deepcopy(main)。評価なし。
  フレーム1: S0を1ステップ予測。mainをD1で補正 -> S1=deepcopy(main)。評価なし。
  フレーム2: S0を2ステップ目まで予測して評価。mainをD2で補正 -> S2=deepcopy(main)。
  フレーム3: S1が完成、評価する。S2は1ステップ予測。S3を生成。
  ...以降、毎フレーム1つずつ「2フレーム前に生まれ、補正なしで2ステップ進んだ
  シャドートラッカー」が完成し、そのフレームの実検出と比較される。

  つまりスキップされるのは最初のhistory_window（デフォルト2）フレームのみで、
  それ以降は毎フレーム評価する。

FP候補: 評価フレームの実検出のうち、完成したシャドートラッカーの予測と
       1対1対応しないもの（confidence < 0.5）
FN候補: 完成したシャドートラッカーの予測のうち、評価フレームの実検出と
       1対1対応しないもの

対応付け:
  同一クラスの候補だけを対象とし、IoU最大となるグローバル1対1割当を行う。
  そのため、1本のtrackerが近接する複数detectionと同時にmatchすることはない。
  tracker prediction用IoU thresholdは detector の評価閾値と分離する。

NOTE: SORTの実装では、検出とマッチしなかった状態が1フレームでも続くと
hit_streakが0にリセットされ、min_hits>=1の確認ゲートで出力から消えてしまう。
そのためシャドートラッカーは SortTracker(max_age=..., min_hits=0) として
生成し、確認ゲートを迂回している（tracker.pyのSortTrackerコンストラクタに
max_age/min_hits/iou_thresholdを渡せるよう修正済み。デフォルト値は
これまでのモジュール定数のままなので、他の呼び出し元の挙動は変わらない）。
max_ageがhistory_window未満だと、補正なしの間にトラック自体が削除され
history_windowフレーム先まで予測が残らないため、max_ageはhistory_window
以上（デフォルト2）にしている。
"""
import argparse
import copy
import csv
from collections import deque
from pathlib import Path
import cv2

from tqdm import tqdm

from src.boundingBox.boundingBox import ClassifiedBoundingBox, ClassifyCategory, DetectionBoundingBox

from src.Evaluation.classifier.detectionClassifier import DetectionClassifier
from src.Evaluation.dataset import fileReader

from src.ObjectDetection.models.factory import build_model

from src.util.dataset import build_dataset
from src.config import IOU_THRESHOLD, RESULT_DIR
from src.tracker.tracker import SortTracker
from src.tracker.bbox_visualization import draw_detection_boxes, draw_tracker_boxes


def count_errors(classified_boxes: list[ClassifiedBoundingBox]) -> tuple[int, int]:
    """
    DetectionClassifier.classify() が返す分類済みboxリストから FP数・FN数 を数える。
    """
    fp = sum(1 for b in classified_boxes if b.classifyCategory == ClassifyCategory.FP)
    fn = sum(1 for b in classified_boxes if b.classifyCategory == ClassifyCategory.FN)
    return fp, fn


def _match_one_to_one(
    tracks: list[DetectionBoundingBox],
    detections: list[DetectionBoundingBox],
    iou_threshold: float,
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    """
    Tracker prediction と current detection をグローバルに1対1対応付けする。

    旧実装の _has_match() は「どれか1つでも一致すればmatch」だったため、
    1本のtrackが複数detectionと同時にmatchした扱いになる問題があった。
    ここでは同一クラスかつIoU>=thresholdの候補について、
    線形割当でIoU合計が最大になる1対1対応を求める。
    """
    if not tracks or not detections:
        return [], list(range(len(tracks))), list(range(len(detections)))

    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as e:
        raise ImportError(
            "scipy が必要です。`pip install scipy` を実行してください。"
        ) from e

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

    unmatched_tracks = [
        i for i in range(n_tracks) if i not in matched_tracks
    ]
    unmatched_detections = [
        j for j in range(n_detections) if j not in matched_detections
    ]

    return matches, unmatched_tracks, unmatched_detections


def _match_prediction_to_error(
    prediction: DetectionBoundingBox,
    error_boxes: list[ClassifiedBoundingBox],
    iou_threshold: float,
) -> bool:
    """tracker prediction が実際の同一クラスのエラーboxと対応するか判定する。"""
    return any(
        prediction.classId == error_box.classId
        and prediction.computeIoU(error_box) >= iou_threshold
        for error_box in error_boxes
    )


class TrackerErrorPredictionExperiment:
    """
    複数の過去フレームから独立に得た tracker prediction と current detection を
    比較し、FP/FN候補を時間方向に統合する実験。

    例: history_windows=(1, 2)
      - t-1 の検出で補正した tracker を t まで1ステップ予測
      - t-2 の検出で補正した tracker を t まで2ステップ予測
      - それぞれについて FP/FN prediction を作成
      - 複数の履歴から支持された候補を最終的な error prediction とする

    FP:
      current detection をアンカーとして各historyから「対応trackerなし」を判定。
      confidence < 0.5 の検出だけをFP候補とする（従来条件を維持）。
      同じcurrent detectionを何historyがFPと判定したかを vote として統合する。

    FN:
      各historyの unmatched tracker prediction をFN候補とする。
      異なるhistoryから近い位置に出たFN候補をIoUでクラスタリングし、
      同じ物体を複数historyが支持した場合に vote を与える。
    """

    def __init__(
        self,
        model_name: str = "yolov8n",
        dataset_name: str = "KITTI",
        map_name: str = "0001",
        history_windows: tuple[int, ...] = (1, 2),
        max_age: int | None = None,
        iou_threshold: float = IOU_THRESHOLD,
        prediction_match_iou: float = 0.1,
        gt_match_iou: float = IOU_THRESHOLD,
        min_votes: int = 1,
        show_debug: bool = True,
        conf_threhold: float = 0.5
    ):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.map_name = map_name

        self.history_windows = tuple(sorted(set(history_windows)))
        if not self.history_windows or any(w < 1 for w in self.history_windows):
            raise ValueError("history_windows は1以上の整数を1つ以上指定してください。")

        self.max_age = max_age if max_age is not None else max(self.history_windows)
        self.iou_threshold = iou_threshold
        self.prediction_match_iou = prediction_match_iou
        self.gt_match_iou = gt_match_iou
        self.min_votes = min_votes
        self.show_debug = show_debug
        self.conf_threshold = conf_threhold

        if self.max_age < max(self.history_windows):
            raise ValueError("max_age は最大history_window以上である必要があります。")
        if not 1 <= min_votes <= len(self.history_windows):
            raise ValueError("min_votes は1以上、history_windows数以下にしてください。")

        assert 0.0 < prediction_match_iou <= 1.0
        assert 0.0 < gt_match_iou <= 1.0

        self.dataset = build_dataset(dataset_name=dataset_name, map_name=map_name)
        self.model = build_model(model_name=model_name, dataset=dataset_name, device="cuda")
        self.classifier = DetectionClassifier(iouThreshold=iou_threshold)

        # 現在の実検出で毎フレーム補正し続けるメイントラッカー。
        self.main_tracker = SortTracker(max_age=self.max_age, min_hits=0)

        # history_windowごとに独立したshadowを保持する。
        # pending_shadows[w] = [(shadow_tracker, remaining_steps), ...]
        self.pending_shadows: dict[int, deque] = {
            w: deque() for w in self.history_windows
        }

        self.output_dir = RESULT_DIR / (
            f"tracker_error_prediction/{dataset_name}/{map_name}/{model_name}"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frame_records: list[dict] = []

    def run(self):
        print(
            f"Model: {self.model_name}, Dataset: {self.dataset_name}, "
            f"history_windows: {self.history_windows}, min_votes: {self.min_votes}"
        )
        print(f"Total frames: {len(self.dataset)}")

        for frame_idx, (image_path, _label_path) in enumerate(tqdm(self.dataset)):
            self._process_frame(frame_idx, image_path)

        self._save_frame_records()
        self._evaluate()

    def _advance_shadows(self) -> dict[int, list[DetectionBoundingBox]]:
        """
        各historyについてshadowを1ステップ進め、
        今フレームで完成したpredictionを返す。
        """
        matured_by_history: dict[int, list[DetectionBoundingBox]] = {}

        for history_window in self.history_windows:
            matured_prediction = None
            still_pending = deque()

            while self.pending_shadows[history_window]:
                shadow, remaining = self.pending_shadows[history_window].popleft()
                prediction = shadow.update([]).predicted_boxes
                remaining -= 1

                if remaining == 0:
                    # 同一historyでは原理上1つだけ完成する。
                    matured_prediction = prediction
                else:
                    still_pending.append((shadow, remaining))

            self.pending_shadows[history_window] = still_pending

            if matured_prediction is not None:
                matured_by_history[history_window] = matured_prediction

        return matured_by_history

    def _create_shadows(self):
        """現在のmain_trackerを各history_window用にコピーする。"""
        for history_window in self.history_windows:
            shadow = copy.deepcopy(self.main_tracker)
            self.pending_shadows[history_window].append(
                (shadow, history_window)
            )

    def _estimate_from_history(
        self,
        history: list[DetectionBoundingBox],
        model_detections: list[DetectionBoundingBox],
    ) -> tuple[list[DetectionBoundingBox], list[DetectionBoundingBox], list[tuple[int, int, float]]]:
        """
        1つのhistoryについてFP/FN候補を作る。
        """
        matches, unmatched_track_indices, unmatched_detection_indices = (
            _match_one_to_one(
                history,
                model_detections,
                self.prediction_match_iou,
            )
        )

        estimated_fp = [
            model_detections[i]
            for i in unmatched_detection_indices
            if model_detections[i].confidenceScore < self.conf_threshold
        ]

        estimated_fn = [
            history[i]
            for i in unmatched_track_indices
        ]

        return estimated_fp, estimated_fn, matches

    @staticmethod
    def _iou(box1: DetectionBoundingBox, box2: DetectionBoundingBox) -> float:
        if box1.classId != box2.classId:
            return 0.0
        return box1.computeIoU(box2)

    def _fuse_fp_predictions(
        self,
        model_detections: list[DetectionBoundingBox],
        fp_by_history: dict[int, list[DetectionBoundingBox]],
    ) -> tuple[list[DetectionBoundingBox], list[int]]:
        """
        FPはcurrent detectionを共通アンカーにする。
        各historyでFPと判定されたかをvoteとして数え、
        min_votes以上の検出を最終FP predictionとする。

        history間でtracker位置が多少ずれていても、同一detectionに対する
        判定を直接統合できるのがFP側の利点。
        """
        votes = [0] * len(model_detections)

        for estimated_fp in fp_by_history.values():
            for detection in estimated_fp:
                # estimated_fpはmodel_detectionsそのものを参照している。
                for idx, current in enumerate(model_detections):
                    if detection is current:
                        votes[idx] += 1
                        break

        fused_fp = [
            model_detections[i]
            for i, vote in enumerate(votes)
            if vote >= self.min_votes
        ]
        return fused_fp, votes

    def _fuse_fn_predictions(
        self,
        fn_by_history: dict[int, list[DetectionBoundingBox]],
    ) -> tuple[list[DetectionBoundingBox], list[int]]:
        """
        FNはhistoryごとにtracker predictionの座標が異なるため、
        history間で同一クラスかつIoU>=prediction_match_iouの候補を統合する。

        各クラスタの代表boxには「何historyが支持したか」をvoteとして持たせる。
        代表boxは、候補群の平均座標を作るのではなく、最初に得られたtracker
        predictionを使用する（既存DetectionBoundingBoxを壊さないため）。
        """
        candidates = []
        for history_window, boxes in fn_by_history.items():
            for box in boxes:
                candidates.append((history_window, box))

        clusters: list[dict] = []

        for history_window, box in candidates:
            best_cluster = None
            best_iou = 0.0

            for cluster in clusters:
                # 既に同じhistoryから入った候補は、同一voteとして二重加算しない。
                if history_window in cluster["histories"]:
                    continue

                iou = self._iou(box, cluster["representative"])
                if iou >= self.prediction_match_iou and iou > best_iou:
                    best_iou = iou
                    best_cluster = cluster

            if best_cluster is None:
                clusters.append({
                    "representative": box,
                    "histories": {history_window},
                })
            else:
                best_cluster["histories"].add(history_window)

        fused_fn = []
        votes = []

        for cluster in clusters:
            vote = len(cluster["histories"])
            if vote >= self.min_votes:
                fused_fn.append(cluster["representative"])
                votes.append(vote)

        return fused_fn, votes

    def _process_frame(self, frame_idx: int, image_path: Path):
        model_detections = self.model.predict(image_path)

        # 1) 過去フレーム由来のshadowを、それぞれのhistory分だけ進める。
        matured_by_history = self._advance_shadows()

        # 2) 現在の検出でmain_trackerを補正。
        self.main_tracker.update(model_detections)

        # 3) 現在の状態から、各history_window用shadowを作る。
        self._create_shadows()

        # 初期history分は評価対象外。
        if len(matured_by_history) < len(self.history_windows):
            return

        # 4) 各historyから独立にFP/FN predictionを作る。
        fp_by_history: dict[int, list[DetectionBoundingBox]] = {}
        fn_by_history: dict[int, list[DetectionBoundingBox]] = {}
        matching_stats: dict[int, tuple[int, int, int]] = {}

        for history_window in self.history_windows:
            history = matured_by_history[history_window]
            estimated_fp, estimated_fn, matches = self._estimate_from_history(
                history,
                model_detections,
            )
            fp_by_history[history_window] = estimated_fp
            fn_by_history[history_window] = estimated_fn

            unmatched_tracks = len(history) - len(matches)
            unmatched_detections = len(model_detections) - len(matches)
            matching_stats[history_window] = (
                len(history),
                len(matches),
                unmatched_tracks + unmatched_detections,
            )

        # 5) 複数historyのpredictionを統合。
        estimated_fp, fp_votes = self._fuse_fp_predictions(
            model_detections,
            fp_by_history,
        )
        estimated_fn, fn_votes = self._fuse_fn_predictions(fn_by_history)

        # 6) GTと比較して、統合predictionが実際のerrorを当てたか確認。
        gt_filename = image_path.stem + ".txt"
        gt_path = self.dataset.label_dir / gt_filename
        if gt_path.exists():
            gt_boxes = fileReader.convertGroundTruthFileToBoundingBoxList(str(gt_path))
        else:
            gt_boxes = []

        classified = self.classifier.classify(gt_boxes, model_detections)

        fps = [
            b for b in classified
            if b.classifyCategory == ClassifyCategory.FP
        ]
        fns = [
            b for b in classified
            if b.classifyCategory == ClassifyCategory.FN
        ]

        true_estimate_fp = sum(1 for b in estimated_fp if b in fps)
        true_estimate_fn = sum(
            1
            for b in estimated_fn
            if _match_prediction_to_error(b, fns, self.gt_match_iou)
        )

        # 7) debug image
        if self.show_debug:
            base_image = cv2.imread(str(image_path))
            if base_image is None:
                raise RuntimeError(f"画像を読み込めません: {image_path}")

            estimate_fn_image = draw_tracker_boxes(
                image=base_image.copy(),
                boxes=estimated_fn,
                color=(0, 0, 255),
                label_prefix="estimate_fn",
            )
            true_fn_image = draw_tracker_boxes(
                image=estimate_fn_image,
                boxes=fns,
                color=(0, 255, 0),
                label_prefix="true_fn",
            )
            detection_add_image = draw_detection_boxes(
                image=true_fn_image,
                boxes=model_detections,
                color=(255, 0, 0),
            )

            det_add_image_dir = self.output_dir / "debug_images"
            det_add_image_dir.mkdir(exist_ok=True)
            if not cv2.imwrite(
                str(det_add_image_dir / f"{frame_idx:06d}.png"),
                detection_add_image,
            ):
                raise RuntimeError("debug imageを保存できません")

            cv2.imshow(winname="fn", mat=detection_add_image)
            cv2.waitKey(1)

        fp_count, fn_count = count_errors(classified)

        record = {
            "frame_idx": frame_idx,
            "fp_estimate": len(estimated_fp),
            "fn_estimate": len(estimated_fn),
            "true_estimate_fp": true_estimate_fp,
            "true_estimate_fn": true_estimate_fn,
            "fp_count": fp_count,
            "fn_count": fn_count,
        }

        # history別のprediction数と統合前後の情報をCSVに残す。
        for history_window in self.history_windows:
            history = matured_by_history[history_window]
            record[f"h{history_window}_num_predictions"] = len(history)
            record[f"h{history_window}_fp_estimate"] = len(
                fp_by_history[history_window]
            )
            record[f"h{history_window}_fn_estimate"] = len(
                fn_by_history[history_window]
            )
            record[f"h{history_window}_num_matches"] = matching_stats[
                history_window
            ][1]

        record["fused_fp_max_vote"] = max(fp_votes, default=0)
        record["fused_fn_max_vote"] = max(fn_votes, default=0)
        record["fused_fp_supported"] = sum(
            vote >= self.min_votes for vote in fp_votes
        )
        record["fused_fn_supported"] = sum(
            vote >= self.min_votes for vote in fn_votes
        )

        self.frame_records.append(record)

    def _save_frame_records(self):
        if not self.frame_records:
            return

        path = self.output_dir / "frame_records.csv"
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(self.frame_records[0].keys()),
            )
            writer.writeheader()
            writer.writerows(self.frame_records)

        print(f"Frame-level records saved to {path}")

    def _precision_recall_f1(
        self,
        estimated_key: str,
        true_key: str,
        actual_key: str,
    ) -> dict:
        """
        オブジェクト単位での推定精度を全フレーム合算で算出する。
        """
        total_estimated = sum(r[estimated_key] for r in self.frame_records)
        total_true = sum(r[true_key] for r in self.frame_records)
        total_actual = sum(r[actual_key] for r in self.frame_records)

        precision = total_true / total_estimated if total_estimated else 0.0
        recall = total_true / total_actual if total_actual else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )

        return {
            "estimated": total_estimated,
            "true_positive": total_true,
            "actual": total_actual,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    def _evaluate(self):
        """
        統合後のFP/FN predictionを評価する。
        """
        fp_stats = self._precision_recall_f1(
            "fp_estimate",
            "true_estimate_fp",
            "fp_count",
        )
        fn_stats = self._precision_recall_f1(
            "fn_estimate",
            "true_estimate_fn",
            "fn_count",
        )

        summary = {
            "model": self.model_name,
            "dataset": self.dataset_name,
            "history_windows": ",".join(map(str, self.history_windows)),
            "max_age": self.max_age,
            "prediction_match_iou": self.prediction_match_iou,
            "gt_match_iou": self.gt_match_iou,
            "min_votes": self.min_votes,
            "n_frames": len(self.frame_records),
        }

        for prefix, stats in (("fp", fp_stats), ("fn", fn_stats)):
            for key, value in stats.items():
                summary[f"{prefix}_{key}"] = value

        summary_path = RESULT_DIR / "tracker_error_prediction" / "summary.csv"
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
        description=(
            "複数の過去フレームから独立に予測したtracker位置をcurrent "
            "detectionと比較し、FP/FN predictionを統合する"
        )
    )
    parser.add_argument("--model", type=str, default="yolov8n")
    parser.add_argument(
        "--dataset",
        type=str,
        default="KITTI",
        choices=["CARLA", "KITTI"],
    )
    parser.add_argument("--map", type=str, default="0001")

    parser.add_argument(
        "--windows",
        type=int,
        nargs="+",
        default=[1, 2],
        help="比較に使う過去フレーム数。デフォルト: 1 2",
    )
    parser.add_argument(
        "--max-age",
        type=int,
        default=None,
        help="SortTrackerのmax_age。未指定なら最大history_window",
    )
    parser.add_argument(
        "--prediction-iou",
        type=float,
        default=0.1,
        help="tracker prediction と current detection の1対1 matching IoU threshold",
    )
    parser.add_argument(
        "--gt-iou",
        type=float,
        default=IOU_THRESHOLD,
        help="FN候補とactual FNを対応付けるIoU threshold",
    )
    parser.add_argument(
        "--min-votes",
        type=int,
        default=1,
        help=(
            "最終predictionとして採用する最低history支持数。"
            "1ならunion、2なら両history一致"
        ),
    )
    parser.add_argument(
        "--no-debug",
        action="store_true",
        help="debug画像表示・保存を無効化",
    )
    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=0.5
    )

    args = parser.parse_args()

    experiment = TrackerErrorPredictionExperiment(
        model_name=args.model,
        dataset_name=args.dataset,
        map_name=args.map,
        history_windows=tuple(args.windows),
        max_age=args.max_age,
        prediction_match_iou=args.prediction_iou,
        gt_match_iou=args.gt_iou,
        min_votes=args.min_votes,
        show_debug=not args.no_debug,
        conf_threhold=args.conf_threshold
    )
    experiment.run()
