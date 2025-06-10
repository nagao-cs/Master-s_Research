import carla
CARLA_HOST = 'localhost'
CARLA_PORT = 2000
TIMEOUT = 10

MAP = "Town01_Opt"
TIME_DURATION = 1000 # シミュレーション時間（秒）
VALID_DISTANCE = 50 # バウンディングボックス検出の最大距離
FIXED_DELTA_SECONDS = 0.1 # シミュレーションのステップ時間

CAR_RATIO = 0.5 # NPC車両のスポーン割合
NUM_WALKERS = 50 # NPC歩行者の数

IM_WIDTH = 800
IM_HEIGHT = 600
FOV = 60

OUTPUT_IMG_DIR = "C:\\CARLA_Latest\\WindowsNoEditor\\output\\image"
OUTPUT_LABEL_DIR = "C:\\CARLA_Latest\\WindowsNoEditor\\output\\label"

# YOLOv5 COCOデータセット準拠のクラスIDを想定
CLASS_MAPPING = {
    carla.CityObjectLabel.TrafficLight: 9,   # COCO: traffic light
    carla.CityObjectLabel.TrafficSigns: 11,  # COCO: stop sign (traffic signをstop signとして扱う)
    carla.CityObjectLabel.Vehicles: 2,       # COCO: car
    carla.CityObjectLabel.Pedestrians: 0,    # COCO: person
}