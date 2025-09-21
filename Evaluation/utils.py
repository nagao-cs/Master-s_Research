SIZE_THRESHOLD = 100  # バウンディングボックスの最小サイズ
IoU_THRESHOLD = 0.5  # IoUの閾値
CONF_THRESHOLD = 0.2  # 信頼度の閾値
ADAPTIVE_THRESHOLD = 5  # 適応的評価の閾値
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


def iou(box1, box2):
    axmin, axmax, aymin, aymax, _ = box1
    bxmin, bxmax, bymin, bymax, _ = box2
    area_a = (axmax - axmin) * (aymax - aymin)
    area_b = (bxmax - bxmin) * (bymax - bymin)

    abxmin = max(axmin, bxmin)
    abxmax = min(axmax, bxmax)
    abymin = max(aymin, bymin)
    abymax = min(aymax, bymax)
    intersection = max(0, abxmax - abxmin) * max(0, abymax - abymin)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0
