class ExperimentConfig:
    def __init__(self):
        SIZE_THRESHOLD = 200  # バウンディングボックスの最小サイズ
        IoU_THRESHOLD = 0.5  # IoUの閾値
        CONF_THRESHOLD = 0.2  # 信頼度の閾値
        ADAPTIVE_THRESHOLD = 5  # 適応的評価の閾値
        IM_WIDTH = 800
        IM_HEIGHT = 600
        class_Map = {
            0: 0,  # pedestrian
            1: 2,  # bicycle
            2: 2,  # motorcycle
            3: 2,  # car
            5: 2,  # bus
            7: 2,  # truck
            9: 9,  # traffic light
            11: 11,  # stop sign
        }
