import argparse
import os
from collections import Counter
from logging import getLogger
from pathlib import Path
from tqdm import tqdm
import time
import yaml
from pydantic import BaseModel

from .stateController import AdrodStateController, AdrodState,ThresholdConfig
from ..boundingBox.integrator.affirmativeIntegrator import ConfidenceBaseIntegrator
from ..boundingBox.integrator.majorityIntegrator import MajorityIntegrator
from src.boundingBox.boundingBox import DetectionBoundingBox
from .detectionExecuter import FrameProcessor
from src.Evaluation.dataset import fileReader
from src.ObjectDetection.models.FLOPsDict import FLOPs_Dict

class AdrodConfig(BaseModel):
    map: str
    integrate_way: str
    thresholds: ThresholdConfig
    iou_threshold: float
    model_1: str
    model_2: str
    model_3: str
    max_version: int
    
def _create_integrator(config: AdrodConfig):
    """統合方法に応じて integrator を生成"""
    
    if config.integrate_way == "affirmative":
        return ConfidenceBaseIntegrator(
            iouThreshold=config.iou_threshold,
            confidenceThreshold=0.0
        )
    elif config.integrate_way == "conf_base":
        return ConfidenceBaseIntegrator(
            iouThreshold=config.iou_threshold,
            confidenceThreshold=config.thresholds.tau_p
        )
    elif config.integrate_way == "consensus":
        return MajorityIntegrator(
            iouThreshold=config.iou_threshold,
            maxVersion=config.max_version)
    else:
        raise ValueError(f"Unknown integration: {config.integrate_way}")

if __name__ == "__main__":
    cwd: Path = Path(__file__).parent
    base_dir: Path = cwd.parent.parent # windowsnoeditor
    
    yaml_path : Path = base_dir / "src" / "adrod" / "config" / "default.yaml" # yamlのファイルパス
    with open(yaml_path, "r") as yaml_file:
        yaml_data: dict = yaml.safe_load(yaml_file)
        
        config: AdrodConfig = AdrodConfig(**yaml_data)
    print(config)
    model_name_list: list[str] = [config.model_1, config.model_2, config.model_3]
    
    # -----------
    # キャッシュから検出結果を読み込み
    # -----------
    detectionBaseDir: Path = base_dir / "oneVersionDetectionResult" / "labels" / config.map

    print("Loading detection results from cached files...")
    detection_cache: dict[str, list[list[DetectionBoundingBox]]] = {}

    for modelName in model_name_list:
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

    numFrames = len(detection_cache[config.model_1])
    print(f"Total frames: {numFrames}\n")

    # -----------
    # 出力ディレクトリ
    # -----------
    outputLabelDir: Path = base_dir / "adaptiveDetectionResult" / "escalation" / "labels" / \
        f"{config.map}" / f"{'_'.join(model_name_list)}" / f"{config.integrate_way}"
    os.makedirs(outputLabelDir, exist_ok=True)

    # ----------
    # 初期化
    # ----------
    logger = getLogger('ultralytics')
    logger.disabled = True

    state_controller = AdrodStateController(config.thresholds)
    integrator = _create_integrator(config)

    processor = FrameProcessor(
        state_controller,
        integrator,
        detection_cache,
        config.model_1,
        config.model_2,
        config.model_3
    )
    total_FLOPs: float = 0.0

    outputDetectionList: list[list[DetectionBoundingBox]] = []
    exe_state_record: list[AdrodState] = []

    # ----------
    # 計測開始
    # ----------
    print("Processing with escalation strategy...")
    start: float = time.time()

    for frame_idx in tqdm(range(numFrames)):
        frame_result = processor.process(frame_idx)
        outputDetectionList.append(frame_result.detections)
        total_FLOPs += frame_result.flops
        exe_state: AdrodState = frame_result.state
        
        # ENSEMBLE状態から自動的にPAIRに遷移
        if state_controller.state == AdrodState.ENSEMBLE:
            state_controller.state = AdrodState.PAIR

        exe_state_record.append(exe_state)

    # 計測終了
    end: float = time.time()
    executionTime: float = end - start
    print(f"\nTotal processing time: {executionTime:.2f} seconds")

    # -----------
    # 結果を保存
    # -----------
    for idx, detections in enumerate(outputDetectionList):
        output_path = outputLabelDir / f"{idx:06d}.txt"
        with open(output_path, 'w') as f:
            for bbox in detections:
                f.write(
                    f"{bbox.classId} {bbox.xCenter} {bbox.yCenter} "
                    f"{bbox.width} {bbox.height} {bbox.confidenceScore}\n"
                )

    # -----------
    # 統計情報の出力
    # -----------
    state_counter = Counter(exe_state_record)
    print(f"\nLevel distribution: {dict(state_counter)}")
    print(f"Results saved to: {outputLabelDir}")
    print(f"総計算量: {total_FLOPs} GFLPs, {total_FLOPs / 17400} cost")
