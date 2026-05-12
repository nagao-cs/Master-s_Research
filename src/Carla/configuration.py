import os
import carla
from dataclasses import dataclass, field
from typing import List, Dict
from pathlib import Path


@dataclass
class SimulationConfig:
    # サーバー設定
    host: str = 'localhost'
    port: int = 2000
    timeout: float = 10.0

    # シミュレーション設定
    map_name: str = "Town01"
    time_duration: float = 100.0
    fixed_delta_seconds: float = 0.1
    synchronous_mode: bool = True

    # NPC設定
    num_vehicles: int = 50
    num_walkers: int = 50
    car_ratio: float = 0.6

    # カメラ・画像設定
    im_width: int = 800
    im_height: int = 600
    fov: int = 60
    valid_distance: float = 200.0
    size_threshold: float = 100.0
    num_camera: int = 5

    # 出力先設定
    outputBaseDir: Path = Path(
        __file__).parent.parent.parent / "GroundTruthDataset"

    # クラスマッピング (変更頻度が低いため定数として定義しても良いが、設定に含めると柔軟)
    class_mapping: Dict[int, int] = field(default_factory=lambda: {
        carla.CityObjectLabel.TrafficLight: 9,
        carla.CityObjectLabel.TrafficSigns: 11,
        carla.CityObjectLabel.Vehicles: 2,
        carla.CityObjectLabel.Pedestrians: 0,
        carla.CityObjectLabel.Buildings: -1,
        carla.CityObjectLabel.Fences: -1,
        carla.CityObjectLabel.Poles: -1,
        carla.CityObjectLabel.Walls: -1,
        carla.CityObjectLabel.Terrain: -1,
        carla.CityObjectLabel.Vegetation: -1,
    })

    @property
    def outputImageDir(self) -> Path:
        return self.outputBaseDir / "images"

    @property
    def outputLabelDir(self) -> Path:
        return self.outputBaseDir / "labels"

    @property
    def outputBBoxImageDir(self) -> Path:
        return self.outputBaseDir / "BBoxDebug"

    def create_directories(self):
        """保存先ディレクトリを作成"""
        os.makedirs(self.outputBaseDir, exist_ok=True)
        os.makedirs(self.outputImageDir, exist_ok=True)
        os.makedirs(self.outputLabelDir, exist_ok=True)
        print(f"Output directories created at: {self.outputBaseDir}")
