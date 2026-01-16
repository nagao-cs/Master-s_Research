from logging import getLogger
from tqdm import tqdm
import time
from pathlib import Path
import os
import argparse
import cv2

from boundingBox.boundingBox import DetectionBoundingBox

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="one version Object Detection"
    )
    argparser.add_argument(
        "--model",
        type=str,
        required=True,
    )
    argparser.add_argument(
        "--map",
        type=str,
        choices=["Town01", "Town02", "Town03", "Town04", "Town05", "Town10HD"],
        help="Map name: Town01, Town02, etc.",
        required=True
    )

    args = argparser.parse_args()
    print(args)

    modelName: str = args.model
    mapName: str = args.map

    if modelName == "yolov8n":
        from ObjectDetection.models.Yolov8n import Yolov8nDetector
        model = Yolov8nDetector()
    elif modelName == "yolov11n":
        from ObjectDetection.models.Yolov11n import Yolov11nDetector
        model = Yolov11nDetector()
    elif modelName == "yolov5n":
        from ObjectDetection.models.Yolov5n import Yolov5nDetector
        model = Yolov5nDetector()
    elif modelName == "rtdetr":
        from ObjectDetection.models.rtDETR import RTDETRDetector
        model = RTDETRDetector()
    elif modelName == 'yolov8l':
        from ObjectDetection.models.yolov8l import Yolov8lDetector
        model = Yolov8lDetector()
    elif modelName == "ssd":
        from ObjectDetection.models.SSD_torch import SSDDetector
        model = SSDDetector()
    elif modelName == "fastrcnn":
        from ObjectDetection.models.FastRCNN import FasterRCNNDetector
        model = FasterRCNNDetector()
    elif modelName == "fcos":
        from ObjectDetection.models.FCOS import FcosDetector
        model = FcosDetector()
    else:
        raise ValueError(
            f"モデル '{modelName}' はサポートされていません。\n"
        )

    cwd: Path = Path(__file__).parent

    inputImageDir: Path = cwd / "output" / "image" / \
        f"{mapName}" / "original" / "front"

    outputDetectionList: list[list[DetectionBoundingBox]] = list()

    print("map: ", mapName)
    print("model: ", model)

    logger = getLogger('ultralytics')
    logger.disabled = True

    if not os.path.exists(inputImageDir):
        raise FileNotFoundError(
            f"Input directory does not exist: {inputImageDir},\n execution file is {Path(__file__)}")

    inputImagePathList: list[Path] = [
        inputImageDir / inputImageFile for inputImageFile in os.listdir(inputImageDir)]

    CONF_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.5

    # 計測開始
    start: float = time.time()
    for inputImagePath in tqdm(inputImagePathList):
        if not os.path.exists(inputImagePath):
            raise FileNotFoundError(f"{inputImagePath} does not exist")

        finalDetections = model.predict(imagePath=inputImagePath)
        outputDetectionList.append(finalDetections)

    # 計測終了
    end: float = time.time()
    executionTime: float = end - start
    print(f"total object detection time: {end - start:.2f} seconds")

    outputLabelDir: Path = cwd / "oneVersionDetectionResult" / "labels" / \
        f"{mapName}" / f"{modelName}"
    outputImageDir: Path = cwd / "oneVersionDetectionResult" / "debugImages" / \
        f"{mapName}" / f"{modelName}"
    os.makedirs(outputLabelDir, exist_ok=True)
    os.makedirs(outputImageDir, exist_ok=True)

    index: int = 0
    for inputImagePath, outputLabelList in tqdm(zip(inputImagePathList, outputDetectionList)):
        outputImagePath: Path = outputImageDir / f"{index:06}.png"
        outputLabelPath: Path = outputLabelDir / f"{index:06}.txt"

        outputImage = cv2.imread(inputImagePath)

        with open(outputLabelPath, 'w') as outputFile:
            for boundingBox in outputLabelList:
                outputImage = boundingBox.drawBoundingBoxOnImage(outputImage)

                xCenter = boundingBox.xCenter
                yCenter = boundingBox.yCenter
                width = boundingBox.width
                height = boundingBox.height
                classId = boundingBox.classId
                confidenceScore = boundingBox.confidenceScore
                outputFile.write(
                    f"{classId} {xCenter} {yCenter} {width} {height} {confidenceScore}\n")
        cv2.imwrite(outputImagePath, outputImage)
        index += 1
