from abc import ABC, abstractmethod
from pathlib import Path
import cv2
from boundingBox.boundingBox import DetectionBoundingBox


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

    def _drawBoundingBox(self, image, boundingBoxList: list[DetectionBoundingBox]):
        imageWidth = image.shape[1]
        imageHeight = image.shape[0]
        for boundingBox in boundingBoxList:
            absoluteXCenter: float = boundingBox.xCenter * imageWidth
            absoluteYCenter: float = boundingBox.yCenter * imageHeight
            absoluteWidth: float = boundingBox.width * imageWidth
            absoluteHeight: float = boundingBox.height * imageHeight

            absoluteXMin: int = int(absoluteXCenter - absoluteWidth / 2)
            absoluteXMax: int = int(absoluteXCenter + absoluteWidth / 2)
            absoluteYMin: int = int(absoluteYCenter - absoluteHeight / 2)
            absoluteYMax: int = int(absoluteYCenter + absoluteHeight / 2)

            label: str = boundingBox.label
            confidenceScore: float = boundingBox.confidenceScore

            cv2.rectangle(image, (absoluteXMin, absoluteYMin),
                          (absoluteXMax, absoluteYMax), (0, 255, 0), 2)
            text = f"{label} {confidenceScore:.2f}"
            cv2.putText(image, text, (absoluteXMin, absoluteYMin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return image

    def save_result(self, imagePath: str, boundingBoxList: list[DetectionBoundingBox], mapName: str, camera, index, modelName: str):
        import os
        import cv2
        """検出結果を保存（共通処理）"""
        image = cv2.imread(imagePath)

        # 出力ディレクトリの設定
        cwd: Path = Path(__file__).parent
        outputImageDir: Path = cwd.parent / "detectionResult" / \
            f"{mapName}" / "images" / f"{modelName}" / f"{camera}"
        outputLabelDir: Path = cwd.parent / "detectionResult" / \
            f"{mapName}" / "labels" / f"{modelName}" / f"{camera}"

        os.makedirs(outputImageDir, exist_ok=True)
        os.makedirs(outputLabelDir, exist_ok=True)

        # 画像の保存
        boundingBoxImage = self._drawBoundingBox(image, boundingBoxList)
        outputImagePath: Path = outputImageDir / f"{index:06}.png"
        cv2.imwrite(outputImagePath, boundingBoxImage)

        # ラベルの保存
        outputLabelPath: Path = outputLabelDir / f"{index:06}.txt"
        with open(outputLabelPath, 'w') as f:
            for boundingBox in boundingBoxList:
                xCenter: float = boundingBox.xCenter
                yCenter: float = boundingBox.yCenter
                width: float = boundingBox.width
                height: float = boundingBox.height
                confidenceScore: float = boundingBox.confidenceScore
                label: str = boundingBox.label
                classId: int = boundingBox.classId
                f.write(
                    f"{classId} {xCenter:.6f} {yCenter:.6f} {width:.6f} {height:.6f} {confidenceScore:.6f}\n")
