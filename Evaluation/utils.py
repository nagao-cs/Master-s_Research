SIZE_THRESHOLD = 500  # バウンディングボックスの最小サイズ
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
    ax_center, ay_center, a_width, a_height, _ = box1
    bx_center, by_center, b_width, b_height, _ = box2
    axmin = ax_center - a_width / 2
    axmax = ax_center + a_width / 2
    aymin = ay_center - a_height / 2
    aymax = ay_center + a_height / 2
    bxmin = bx_center - b_width / 2
    bxmax = bx_center + b_width / 2
    bymin = by_center - b_height / 2
    bymax = by_center + b_height / 2
    area_a = (axmax - axmin) * (aymax - aymin)
    area_b = (bxmax - bxmin) * (bymax - bymin)

    abxmin = max(axmin, bxmin)
    abxmax = min(axmax, bxmax)
    abymin = max(aymin, bymin)
    abymax = min(aymax, bymax)
    intersection = max(0, abxmax - abxmin) * max(0, abymax - abymin)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0


if __name__ == "__main__":
    # テスト用
    box1 = (0.08739471435546875, 72.09510040283203,
            324.7374572753906, 374.1318359375, 0.44695085287094116)
    box2 = (0.0, 81.1821085767601, 320.76099300798137, 373.40947502363366, 0.0)

    print("Test with debug output:")
    iou_value = iou(box1, box2)
    print(f"Final IoU: {iou_value}")
