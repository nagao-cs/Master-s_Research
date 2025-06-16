import os 
import cv2



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

class Dataset:
    def __init__(self, gt_dir, detection_dir):
        self.gt = self.get_gt(gt_dir)
        self.detections = self.get_detections(detection_dir)
        
    def get_gt(self, gt_dir):
        gt = list()
        for filename in os.listdir(gt_dir):
            filepath = os.path.join(gt_dir, filename)
            with open(filepath, 'r') as f:
                lines = f.readlines()
                frame_gt = dict()
                for line in lines[1:]:
                    if not line.strip():
                        continue
                    parts = line.strip().split(',')
                    class_id = int(parts[0])
                    xmin = float(parts[1])
                    xmax = float(parts[2])
                    ymin = float(parts[3])
                    ymax = float(parts[4])
                    distance = float(parts[5])
                    if class_id not in frame_gt:
                        frame_gt[class_id] = list()
                    frame_gt[class_id].append((xmin, xmax, ymin, ymax, distance))
                gt.append(frame_gt)
        return gt

    def get_detections(self, detection_dir):
        cameras = ['front', 'left_1', 'right_1']
        all_detections = dict()
        for camera in cameras:
            camera_detection_dir = os.path.join(detection_dir, camera)
            if not os.path.exists(camera_detection_dir):
                print(f"Detection directory does not exist: {camera_detection_dir}")
                continue
            
            detections = self.get_camera_detections(camera_detection_dir)
            all_detections[camera] = detections
        return all_detections
    
    def get_camera_detections(self, camera_detection_dir):
        detections = list()
        for filename in os.listdir(camera_detection_dir):
            filepath = os.path.join(camera_detection_dir, filename)
            with open(filepath, 'r') as f:
                lines = f.readlines()
                frame_detections = dict()
                for line in lines[1:]:
                    if not line.strip():
                        continue
                    parts = line.strip().split(',')
                    class_id = int(parts[0])
                    xmin = float(parts[1])
                    xmax = float(parts[2])
                    ymin = float(parts[3])
                    ymax = float(parts[4])
                    confidence = float(parts[5])
                    if class_id not in frame_detections:
                        frame_detections[class_id] = list()
                    frame_detections[class_id].append((xmin, xmax, ymin, ymax, confidence))
                detections.append(frame_detections)
        return detections
        
def main():
    gt_dir = 'C:/CARLA_Latest/WindowsNoEditor/output/label/Town01_Opt/front'
    detection_dir = 'C:/CARLA_Latest/WindowsNoEditor/ObjectDetection/yolov8_results/labels/Town01_Opt'
    
    dataset = Dataset(gt_dir, detection_dir)
    print(dataset.gt[0])
    print(dataset.detections['front'][0])
    
    
        
if __name__ == "__main__":
    main()