from src.boundingBox.integrator.integrator import Integrator
from src.boundingBox.boundingBox import DetectionBoundingBox
from .grouping import group_detection_and_tracking, GroupingResult


class TwoModelTrackIntegrator(Integrator):
    """
    モデルA・Bの検出結果とそれぞれのトラッカー予測から、統合された最終検出結果を
    構築する Integrator。

    手順:
    ① モデルA・Bそれぞれについて、検出とそのトラッカー予測をグルーピングする
       （grouping.pyに委譲）
    ② グルーピング結果から、各モデルのFP疑い(det_only)・FN疑い(trk_only)が得られる
    ③ FP疑いはもう片方のモデルの検出と比較し、一致すれば採用する。
       FN疑いも同様に、もう片方のモデルの検出と比較し、一致すれば採用する
       （自己一貫した検出=agreedはそのまま採用する）
    """

    def __init__(self, iouThreshold: float):
        self.iouThreshold: float = iouThreshold

    def _hasMatch(
        self,
        box: DetectionBoundingBox,
        candidates: list[DetectionBoundingBox],
    ) -> bool:
        """box が candidates の中に同一クラス・IoU>=閾値の一致相手を持つか判定する"""
        return any(
            c.classId == box.classId and box.computeIoU(c) >= self.iouThreshold
            for c in candidates
        )

    def _confirmed(
        self,
        suspects: list[DetectionBoundingBox],
        otherModelDetections: list[DetectionBoundingBox],
    ) -> list[DetectionBoundingBox]:
        """suspects のうち、もう片方のモデルの検出と一致したものだけを返す（採用分）"""
        return [s for s in suspects if self._hasMatch(s, otherModelDetections)]

    def integrate(
        self,
        groupingA: GroupingResult,
        groupingB: GroupingResult,
        modelADetections: list[DetectionBoundingBox],
        modelBDetections: list[DetectionBoundingBox],
    ) -> list[DetectionBoundingBox]:
        """
        ②③: グルーピング結果（FP疑い・FN疑い）をもう片方のモデルの検出と
        突き合わせ、一致したものだけを採用して最終出力を組み立てる。
        自己一貫した検出（agreed）はそのまま採用する。
        """
        confirmedFpA = self._confirmed(groupingA.det_only, modelBDetections)
        confirmedFpB = self._confirmed(groupingB.det_only, modelADetections)
        confirmedFnA = self._confirmed(groupingA.trk_only, modelBDetections)
        confirmedFnB = self._confirmed(groupingB.trk_only, modelADetections)

        return (
            groupingA.agreed
            + groupingB.agreed
            + confirmedFpA
            + confirmedFpB
            + confirmedFnA
            + confirmedFnB
        )

    def execute(
        self,
        modelADetections: list[DetectionBoundingBox],
        modelATracking: list[DetectionBoundingBox],
        modelBDetections: list[DetectionBoundingBox],
        modelBTracking: list[DetectionBoundingBox],
    ) -> list[DetectionBoundingBox]:
        """
        ①: モデルA・Bそれぞれと、それぞれのトラッカーをグルーピングしてから、
        integrate() で最終出力を構築する。
        """
        groupingA = group_detection_and_tracking(
            model_detections=modelADetections, tracking_detections=modelATracking, iou_threshold=self.iouThreshold
        )
        groupingB = group_detection_and_tracking(
            model_detections=modelBDetections, tracking_detections=modelBTracking, iou_threshold=self.iouThreshold
        )

        return self.integrate(groupingA, groupingB, modelADetections, modelBDetections)

    def __call__(
        self,
        modelADetections: list[DetectionBoundingBox],
        modelATracking: list[DetectionBoundingBox],
        modelBDetections: list[DetectionBoundingBox],
        modelBTracking: list[DetectionBoundingBox],
    ) -> list[DetectionBoundingBox]:
        return self.execute(modelADetections, modelATracking, modelBDetections, modelBTracking)