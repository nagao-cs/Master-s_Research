from abc import ABC, abstractmethod
from pathlib import Path
import cv2
from boundingbox.boundingBox import BoundingBox


class DetectionResult:
    def __init__(self):
        pass


class AbstractObjectDetector:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.load_model()

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def predict(self, image):
        pass

    def _drawBoundingBox(self, image, boundingBoxList: list[BoundingBox]):
        imageWidth = image.shape[1]
        imageHeight = image.shape[0]
        for boundingBox in boundingBoxList:
            absoluteXCenter = boundingBox.xCenter * imageWidth
            absoluteYCenter = boundingBox.yCenter * imageHeight
            absoluteWidth = boundingBox.width * imageWidth
            absoluteHeight = boundingBox.height * imageHeight

            absoluteXMin = int(absoluteXCenter - absoluteWidth / 2)
            absoluteXMax = int(absoluteXCenter + absoluteWidth / 2)
            absoluteYMin = int(absoluteYCenter - absoluteHeight / 2)
            absoluteYMax = int(absoluteYCenter + absoluteHeight / 2)

            label = boundingBox.classId
            confidenceScore = boundingBox.confidenceScore

            cv2.rectangle(image, (absoluteXMin, absoluteYMin),
                          (absoluteXMax, absoluteYMax), (0, 255, 0), 2)
            text = f"{label} {confidenceScore:.2f}"
            cv2.putText(image, text, (absoluteXMin, absoluteYMin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return image

    def save_result(self, imagePath: str, boundingBoxList: list[BoundingBox], mapName: str, camera, index, modelName: str):
        import os
        import cv2
        """検出結果を保存（共通処理）"""
        image = cv2.imread(imagePath)

        # 出力ディレクトリの設定
        cwd = Path(__file__).parent
        outputImageDir = cwd.parent / "detectionResult" / \
            f"{mapName}" / "images" / f"{modelName}" / f"{camera}"
        outputLabelDir = cwd.parent / "detectionResult" / \
            f"{mapName}" / "labels" / f"{modelName}" / f"{camera}"

        os.makedirs(outputImageDir, exist_ok=True)
        os.makedirs(outputLabelDir, exist_ok=True)

        # 画像の保存
        boundingBoxImage = self._drawBoundingBox(image, boundingBoxList)
        outputImagePath = os.path.join(outputImageDir, f"{index}.png")
        cv2.imwrite(outputImagePath, boundingBoxImage)

        # ラベルの保存
        outputLabelPath = os.path.join(outputLabelDir, f"{index}.txt")
        with open(outputLabelPath, 'w') as f:
            for boundingBox in boundingBoxList:
                xCenter = boundingBox.xCenter
                yCenter = boundingBox.yCenter
                width = boundingBox.width
                height = boundingBox.height
                confidenceScore = boundingBox.confidenceScore
                label = boundingBox.label
                classId = boundingBox.classId
                f.write(
                    f"{classId} {xCenter:.6f} {yCenter:.6f} {width:.6f} {height:.6f} {confidenceScore:.6f}\n")
