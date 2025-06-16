import os 
import cv2

gt_dir = 'C:/CARLA_Latest/WindowsNoEditor/output/label/Town01_Opt'

detection_dir = 'C:/CARLA_Latest/WindowsNoEditor/ObjectDetection/yolov8_results/labels/Town01_Opt'

def iou(box1, box2):
    axmin, axmax, aymin, aymax = box1
    bxmin, bxmax, bymin, bymax = box2
    area_a = (axmax - axmin) * (aymax - aymin)
    area_b = (bxmax - bxmin) * (bymax - bymin)
    
    abxmin = max(axmin, bxmin)
    abxmax = min(axmax, bxmax)
    abymin = max(aymin, bymin)
    abymax = min(aymax, bymax)
    intersection = max(0, abxmax - abxmin) * max(0, abymax - abymin)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0

def unanimous_detections():
    pass

def all_detections():
   pass