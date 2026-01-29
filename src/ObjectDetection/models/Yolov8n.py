from ultralytics import YOLO
from .AbstractObjectDetector import AbstractObjectDetector
from ..utils import utils
from src.boundingBox.boundingBox import DetectionBoundingBox
import torch


class Yolov8nDetector(AbstractObjectDetector):
    def __init__(self):
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"device is {self.device}")
        # self.device = "cpu"
        self.load_model()

    def load_model(self):
        try:
            self.model = YOLO("yolov8n.pt")
            # self.model = YOLO(
            # "ObjectDetection/trainingModel/trainingyolov8n/weights/best.pt")
            print(f"YOLOv8n model loaded")

        except Exception as e:
            raise RuntimeError(f"Error loading yolov8n model: {e}")

    def predict(self, imagePath):
        detectionResult = self.model.predict(
            source=imagePath,
            device=self.device,
            conf=utils.CONF_THRESHOLD,
            classes=[0, 2, 9, 11],
            verbose=False
        )[0]  # 1枚の画像を処理しても、バッチやストリームで処理してもリストで結果が返ってくるらしい。→ 最初の1つを見るためのインデックスで[0]

        rawBoundingBoxList = detectionResult.boxes
        imageHeight, imageWidth = detectionResult.orig_shape

        if len(rawBoundingBoxList) == 0:
            return []

        # 全ボックスを取得
        xywhTensor = rawBoundingBoxList.xywhn  # shape: [N, 4]
        confidenceScoreTensor = rawBoundingBoxList.conf    # shape: [N,]
        classIdTensor = rawBoundingBoxList.cls      # shape: [N,]

        # サイズ計算
        widthTensor = xywhTensor[:, 2]      # shape: [N]
        heightTensor = xywhTensor[:, 3]     # shape: [N]
        sizeTensor = widthTensor * imageWidth * \
            heightTensor * imageHeight  # shape: [N]

        # サイズについてのマスク作成
        sizeMask = sizeTensor >= utils.SIZE_THRESHOLD

        # マスク適用
        validIndices = torch.where(sizeMask)[0]

        if len(validIndices) == 0:
            return []

        # フィルタ済みデータを取得
        filteredXYWH = xywhTensor[validIndices]  # shape: [M, 4]
        # shape: [M]
        filteredConfidenceScore = confidenceScoreTensor[validIndices]
        filteredClassId = classIdTensor[validIndices]       # shape: [M]

        # CPU に転送
        xywh = filteredXYWH.cpu().numpy()
        confidencescoreList = filteredConfidenceScore.cpu().numpy()
        classIdList = filteredClassId.cpu().numpy().astype(int)

        # 各バウンディングボックスを処理
        outputBoundingBoxList: list[DetectionBoundingBox] = []
        for i in range(len(validIndices)):
            xCenter = xywh[i, 0]
            yCenter = xywh[i, 1]
            width = xywh[i, 2]
            height = xywh[i, 3]
            classId = classIdList[i]
            confidencescore = confidencescoreList[i]

            boundingBoxInstance = DetectionBoundingBox(
                float(xCenter), float(yCenter), float(width),
                float(height), int(classId), float(confidencescore)
            )
            outputBoundingBoxList.append(boundingBoxInstance)

        return outputBoundingBoxList
