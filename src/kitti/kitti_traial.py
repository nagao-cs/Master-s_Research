from logging import getLogger
from tqdm import tqdm
import time
from pathlib import Path
import os
import argparse
import cv2

from src.boundingBox.boundingBox import DetectionBoundingBox

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="one version Object Detection"
    )
    argparser.add_argument(
        "--model",
        type=str,
        required=True,
    )

    args = argparser.parse_args()
    print(args)

    modelName: str = args.model

    if modelName == "yolov8n":
        from src.ObjectDetection.models.Yolov8n import Yolov8nDetector
        model = Yolov8nDetector()
    elif modelName == "yolov11n":
        from src.ObjectDetection.models.Yolov11n import Yolov11nDetector
        model = Yolov11nDetector()
    elif modelName == "yolov5n":
        from src.ObjectDetection.models.Yolov5n import Yolov5nDetector
        model = Yolov5nDetector()
    elif modelName == "yolov26x":
        from src.ObjectDetection.models.Yolo26x import Yolov26xDetector
        model = Yolov26xDetector()
    elif modelName == "rtdetr":
        from src.ObjectDetection.models.rtDETR import RTDETRDetector
        model = RTDETRDetector()
    elif modelName == 'yolov8l':
        from src.ObjectDetection.models.yolov8l import Yolov8lDetector
        model = Yolov8lDetector()
    elif modelName == "ssd":
        from src.ObjectDetection.models.SSD_torch import SSDDetector
        model = SSDDetector()
    elif modelName == "fastrcnn":
        from src.ObjectDetection.models.FastRCNN import FasterRCNNDetector
        model = FasterRCNNDetector()
    elif modelName == "fcos":
        from src.ObjectDetection.models.FCOS import FcosDetector
        model = FcosDetector()
    elif modelName == "retinanet":
        from src.ObjectDetection.models.retinanet import RetinanetDetector
        model = RetinanetDetector()
    elif modelName == "trainYolo":
        from src.ObjectDetection.models.trainedYolov8n import Yolov8nTrainedDetector
        model = Yolov8nTrainedDetector()
    else:
        raise ValueError(
            f"モデル '{modelName}' はサポートされていません。\n"
        )

    cwd: Path = Path(__file__).parent
    base_dir: Path = cwd.parent.parent  # WindowsNoEditor

    input_image_dir: Path = base_dir.parent.parent.parent / "d" / "data_tracking_image_2" / "training" / "image_02" / "0020"

    outputDetectionList: list[list[DetectionBoundingBox]] = list()

    print("model: ", model)

    logger = getLogger('ultralytics')
    logger.disabled = True

    if not os.path.exists(input_image_dir):
        raise FileNotFoundError(
            f"Input directory does not exist: {input_image_dir},\n execution file is {Path(__file__)}")

    inputImagePathList: list[Path] = [
        input_image_dir / inputImageFile for inputImageFile in os.listdir(input_image_dir)]

    # 計測開始
    start: float = time.time()
    for inputImagePath in tqdm(inputImagePathList):
        if not os.path.exists(inputImagePath):
            raise FileNotFoundError(f"{inputImagePath} does not exist")

        finalDetections = model.predict(image_path=inputImagePath)
        outputDetectionList.append(finalDetections)

    # 計測終了
    end: float = time.time()
    executionTime: float = end - start
    print(f"total object detection time: {end - start:.2f} seconds")

    outputLabelDir: Path = cwd / "result" / modelName /"label"
    outputImageDir: Path = cwd / "result" / modelName / "image"
    os.makedirs(outputLabelDir, exist_ok=True)
    os.makedirs(outputImageDir, exist_ok=True)

    index: int = 0
    for inputImagePath, outputLabelList in tqdm(zip(inputImagePathList, outputDetectionList), desc="[result save]", total=len(outputDetectionList)):
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
