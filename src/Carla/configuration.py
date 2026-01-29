import os
import carla
from dataclasses import dataclass, field
from typing import List, Dict


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
    car_ratio: float = 0.1

    # カメラ・画像設定
    im_width: int = 600
    im_height: int = 600
    fov: int = 60
    valid_distance: float = 100.0
    size_threshold: float = 200.0
    num_camera: int = 1

    # 出力先設定
    base_output_dir: str = "C:\\CARLA_Latest\\WindowsNoEditor\\ReliabilityOfNversionObjectDetection\\dataset"
    base_output_dir: str = "C:\\CARLA_Latest\\WindowsNoEditor\\groundTruthDataset"

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
    def output_img_dir(self):
        return os.path.join(self.base_output_dir, "images")

    @property
    def output_label_dir(self):
        return os.path.join(self.base_output_dir, "labels")

    def create_directories(self):
        """保存先ディレクトリを作成"""
        os.makedirs(self.output_img_dir, exist_ok=True)
        os.makedirs(self.output_label_dir, exist_ok=True)
        print(f"Output directories created at: {self.base_output_dir}")
