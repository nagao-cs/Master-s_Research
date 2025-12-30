import sys
from logging import getLogger
from tqdm import tqdm
import time
from pathlib import Path
import os
from typing import List

from adaptiveNversion.NversionExecutor import NversionExecutor
from adaptiveNversion.versionController.VersionController import VersionController, VersionState
from adaptiveNversion.integrator import MajorityIntegrator

from ObjectDetection.boundingbox.boundingBox import BoundingBox


if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser(
        description="Adaptive Object Detection"
    )
    argparser.add_argument(
        "--models",
        type=str,
        required=True,
        choices=["yolov8n", "yolov5n", "yolov11n",
                 "rtdetr", "ssd", "yolov8l", "fastrcnn"],
        nargs='+',
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

    modelNames = args.models
    numModel = len(modelNames)
    mapName = args.map

    modelList = list()

    for modelName in modelNames:
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
            from ObjectDetection.models.SSD import SSDDetector
            model = SSDDetector()
        elif modelName == "fastrcnn":
            from ObjectDetection.models.FastRCNN import FastRCNNDetector
            model = FastRCNNDetector()
        else:
            raise ValueError(
                f"モデル '{modelName}' はサポートされていません。\n"
            )
        modelList.append(model)

    cwd = Path(__file__).parent

    inputImageDir = cwd / "output" / "image" / \
        f"{mapName}" / "original" / "front"

    numInference = 0
    outputLabelList: list[list[BoundingBox]] = list()

    print("map: ", mapName)
    print("models: ", modelList)

    logger = getLogger('ultralytics')
    logger.disabled = True

    if not os.path.exists(inputImageDir):
        raise FileNotFoundError(
            f"Input directory does not exist: {inputImageDir},\n execution file is {Path(__file__)}")

    inputFileList = os.listdir(inputImageDir)

    CONF_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.5
    AGREEMENT_THRESHOLD = 0.8

    numVersion = len(modelList)

    detectionIntegrator = MajorityIntegrator(
        iouThreshold=IOU_THRESHOLD, maxVersion=numVersion)
    detectionExecutor = NversionExecutor(modelList, detectionIntegrator)
    versionController = VersionController(
        confidenceScoreThreshold=CONF_THRESHOLD, agreementScoreThreshold=AGREEMENT_THRESHOLD, maxVersion=numVersion)

    # 計測開始
    start = time.time()
    for inputFile in tqdm(inputFileList):
        inputImagePath = os.path.join(inputImageDir, inputFile)
        if not os.path.exists(inputImagePath):
            raise FileNotFoundError(f"{inputImagePath} does not exist")

        versionState = versionController.state
        if versionState == VersionState.ONE:
            baseDetection = detectionExecutor.executeOneVersionDetection(
                inputImagePath)
            versionController.updateState(detections=baseDetection)

            if versionState == VersionState.N:
                finalDetections = detectionExecutor.executeNVersionDetection(
                    inputImagePath, baseDetection)
                numInference += len(modelList)
            else:
                finalDetections = baseDetection
                numInference += 1
        else:
            integratedBoundingBoxList, groupedBoundingBoxList = detectionExecutor.executeNVersionDetection(
                inputImagePath)
            versionController.updateState(
                groupedBoundingBoxList=groupedBoundingBoxList)
            finalDetections = integratedBoundingBoxList
            numInference += len(modelList)

        outputLabelList.append(finalDetections)

    # 計測終了
    end = time.time()
    print(f"total object detection time: {end - start:.2f} seconds")

    outputLabelDir = cwd / "detectionResult" / "labels" / \
        f"{mapName}" / f"{'_'.join(modelNames)}"
    os.makedirs(outputLabelDir, exist_ok=True)

    index = 0
    for outputLabel in outputLabelList:
        outputLabelPath = os.path.join(outputLabelDir, f"{index:6f}.txt")
        with open(outputLabelPath, 'w') as f:
            for bbox in outputLabel:
                x_center = bbox['x_center']
                y_center = bbox['y_center']
                width = bbox['width']
                height = bbox['height']
                class_id = bbox['class_id']
                conf = bbox['confidence']
                f.write(
                    f"{class_id} {x_center} {y_center} {width} {height} {conf}\n")
        index += 1
    print(f"Total inferences made: {numInference}")
    elapsed = end - start
    print("total time:", elapsed)
