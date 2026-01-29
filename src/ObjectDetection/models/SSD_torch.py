import cv2

import torch
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
from torchvision.transforms import functional as F

from .AbstractObjectDetector import AbstractObjectDetector
from ..utils import utils
from src.boundingBox.boundingBox import DetectionBoundingBox


class SSDDetector(AbstractObjectDetector):
    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.load_model()

    def load_model(self):
        try:
            weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
            self.model = ssdlite320_mobilenet_v3_large(weights=weights)
            self.model.to(self.device)
            self.model.eval()
            print(f"PyTorch SSD model loaded on {self.device}")
        except Exception as e:
            raise RuntimeError(f"Error loading SSD model: {e}")

    def predict(self, imagePath):
        image = cv2.imread(imagePath)
        if image is None:
            return []

        imageHeight, imageWidth = image.shape[:2]

        # RGBへ変換し、Tensor化
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        imageTensor = F.to_tensor(imageRGB).to(self.device)

        with torch.no_grad():
            detections = self.model([imageTensor])[0]

        # 結果の取り出し
        classIdList = detections["labels"]
        # [xmin, ymin, xmax, ymax] (ピクセル単位)
        xyxyList = detections["boxes"]
        confidenceScoreList = detections["scores"]

        # 信頼度のマスク
        confidenceScoreMask = confidenceScoreList >= utils.CONF_THRESHOLD

        # クラスIDのマスク
        targetClasseId = torch.tensor([1, 3, 10, 12], device=self.device)
        classIdMask = torch.isin(classIdList, targetClasseId)

        # サイズのマスク
        widths = xyxyList[:, 2] - xyxyList[:, 0]
        heights = xyxyList[:, 3] - xyxyList[:, 1]
        pixelSizes = widths * heights
        sizeMask = pixelSizes >= utils.SIZE_THRESHOLD

        # 全ての条件を統合
        combinedMask = confidenceScoreMask & classIdMask & sizeMask
        validIndices = torch.where(combinedMask)[0]

        if len(validIndices) == 0:
            return []

        # フィルタリングされたデータのみを取得・一括変換
        filteredBoundingBoxList = xyxyList[validIndices]
        fileterdConfidenceScoreList = confidenceScoreList[validIndices]
        fileterdClassIdList = classIdList[validIndices]

        # 正規化座標への変換
        xMin, yMin, xMax, yMax = \
            filteredBoundingBoxList[:, 0], \
            filteredBoundingBoxList[:, 1], \
            filteredBoundingBoxList[:, 2], \
            filteredBoundingBoxList[:, 3],

        normarizedWidths = (xMax - xMin) / imageWidth
        normarizedHeights = (yMax - yMin) / imageHeight
        normarizedXCenters = ((xMin + xMax) / 2) / imageWidth
        normarizedYCenters = ((yMin + yMax) / 2) / imageHeight

        # CPUへ一括転送
        XCenters = normarizedXCenters.cpu().numpy()
        YCenters = normarizedYCenters.cpu().numpy()
        Widths = normarizedWidths.cpu().numpy()
        Heights = normarizedHeights.cpu().numpy()
        confidenceScoreList = fileterdConfidenceScoreList.cpu().numpy()
        classIdList = fileterdClassIdList.cpu().numpy().astype(int) - 1
        outputBoundingBoxList = []
        for i in range(len(confidenceScoreList)):
            xCenter = XCenters[i]
            yCenter = YCenters[i]
            width = Widths[i]
            height = Heights[i]
            confidenceScore = confidenceScoreList[i]
            classId = classIdList[i]

            boundingBoxInstance = DetectionBoundingBox(
                xCenter, yCenter, width, height, classId, confidenceScore)
            outputBoundingBoxList.append(boundingBoxInstance)

        return outputBoundingBoxList
