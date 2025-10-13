import carla
CARLA_HOST = 'localhost'
CARLA_PORT = 2000
TIMEOUT = 10

MAP = "Town02"
TIME_DURATION = 100  # シミュレーション時間（秒）
VALID_DISTANCE = 100  # バウンディングボックス検出の最大距離
SIZE_THRESHOLD = 200  # バウンディングボックスの最小サイズ(わかりやすく大きくしてる)
FIXED_DELTA_SECONDS = 0.05  # シミュレーションのステップ時間

CAR_RATIO = 0.1  # NPC車両のスポーン割合
NUM_WALKERS = 50  # NPC歩行者の数

IM_WIDTH = 800
IM_HEIGHT = 600
FOV = 60
XMIN = 1
XMAX = 2
YMIN = 3
YMAX = 4
DIST = 5
NUM_CAMERA = 1  # カメラの数

OUTPUT_IMG_DIR = "C:\\CARLA_Latest\\WindowsNoEditor\\output\\image"
OUTPUT_LABEL_DIR = "C:\\CARLA_Latest\\WindowsNoEditor\\output\\label"

# YOLOv5 COCOデータセット準拠のクラスIDを想定
CLASS_MAPPING = {
    carla.CityObjectLabel.TrafficLight: 9,   # COCO: traffic light
    # COCO: stop sign (traffic signをstop signとして扱う)
    carla.CityObjectLabel.TrafficSigns: 11,
    carla.CityObjectLabel.Vehicles: 2,       # COCO: car
    carla.CityObjectLabel.Pedestrians: 0,    # COCO: person
    carla.CityObjectLabel.Buildings: -1,
    carla.CityObjectLabel.Fences: -1,
    carla.CityObjectLabel.Poles: -1,
    carla.CityObjectLabel.Walls: -1,
    carla.CityObjectLabel.Terrain: -1,
    carla.CityObjectLabel.Vegetation: -1,

}
