import sys
from logging import getLogger
from tqdm import tqdm
import time
from pathlib import Path
import os
import argparse

from src.boundingBox.integrator.majorityIntegrator import MajorityIntegrator
from src.boundingBox.integrator.affirmativeIntegrator import ConfidenceBaseIntegrator

from src.boundingBox.boundingBox import DetectionBoundingBox
from src.Evaluation.dataset import fileReader

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="N-version Object Detection (Cache-based)"
    )
    argparser.add_argument(
        "--models",
        type=str,
        required=True,
        nargs='+',
    )
    argparser.add_argument(
        "--map",
        type=str,
        choices=["Town01", "Town02", "Town03", "Town04", "Town05", "Town10HD"],
        help="Map name: Town01, Town02, etc.",
        required=True
    )
    argparser.add_argument(
        "--integrate",
        type=str,
        default="affirmative"
    )
    
    args = argparser.parse_args()
    print(args)

    modelNameList: list[str] = args.models
    mapName: str = args.map
    integrate_way = args.integrate

    cwd: Path = Path(__file__).parent
    baseDir: Path = cwd.parent.parent

    # -----------
    # キャッシュから検出結果を読み込み
    # -----------
    detectionBaseDir: Path = baseDir / "oneVersionDetectionResult" / "labels" / mapName

    print("Loading detection results from cached files...")
    detection_cache: dict[str, list[list[DetectionBoundingBox]]] = {}

    for modelName in modelNameList:
        modelDetectionDir = detectionBaseDir / modelName
        if not os.path.exists(modelDetectionDir):
            raise FileNotFoundError(
                f"Detection directory does not exist: {modelDetectionDir}\n"
                f"Available models: {os.listdir(detectionBaseDir)}"
            )

        detectionFiles = sorted([
            f for f in os.listdir(modelDetectionDir)
            if f.endswith('.txt')
        ])

        modelDetections: list[list[DetectionBoundingBox]] = []
        for detectionFile in detectionFiles:
            detectionFilePath = modelDetectionDir / detectionFile
            detectionBBoxList = fileReader.convertDetectionFileToBoundingBoxList(
                str(detectionFilePath)
            )
            modelDetections.append(detectionBBoxList)

        detection_cache[modelName] = modelDetections
        print(f"  Loaded {len(modelDetections)} frames for {modelName}")

    numFrames = len(detection_cache[modelNameList[0]])
    print(f"Total frames: {numFrames}\n")

    outputDetectionList: list[list[DetectionBoundingBox]] = list()

    print("map: ", mapName)
    print("models: ", modelNameList)

    logger = getLogger('ultralytics')
    logger.disabled = True

    CONF_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.5

    if integrate_way == "affirmative":
        integrator = ConfidenceBaseIntegrator(
            iouThreshold=IOU_THRESHOLD,
            confidenceThreshold=0.0
        )
    elif integrate_way == "conf_base":
        integrator = ConfidenceBaseIntegrator(
            iouThreshold=IOU_THRESHOLD,
            confidenceThreshold=CONF_THRESHOLD
        )
    elif integrate_way == "consensus":
        integrator = MajorityIntegrator(
            iouThreshold=IOU_THRESHOLD,
            maxVersion=len(modelNameList)
        )

    # 計測開始
    start: float = time.time()
    for frameIdx in tqdm(range(numFrames)):
        modelDict = {
            modelName: detection_cache[modelName][frameIdx]
            for modelName in modelNameList
        }
        integratedBoundingBoxList, _ = integrator(modelDict)
        outputDetectionList.append(integratedBoundingBoxList)

    # 計測終了
    end: float = time.time()
    executionTime: float = end - start
    print(f"total processing time: {end - start:.2f} seconds")

    outputLabelDir: Path = baseDir / "NversionDetectionResult" / \
        "labels" / f"{mapName}" / f"{'_'.join(modelNameList)}" / f"{integrate_way}"
    os.makedirs(outputLabelDir, exist_ok=True)

    index: int = 0
    for outputLabelList in outputDetectionList:
        outputLabelPath = outputLabelDir / f"{index:06}.txt"
        with open(outputLabelPath, 'w') as outputFile:
            for boundingBox in outputLabelList:
                xCenter = boundingBox.xCenter
                yCenter = boundingBox.yCenter
                width = boundingBox.width
                height = boundingBox.height
                classId = boundingBox.classId
                confidenceScore = boundingBox.confidenceScore
                outputFile.write(
                    f"{classId} {xCenter} {yCenter} {width} {height} {confidenceScore}\n")
        index += 1