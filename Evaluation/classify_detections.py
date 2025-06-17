import os 
import cv2



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


def unanimous_detections():
    pass


class Dataset:
    class_Map = {
        0: 0,  #pedestrian
        1: 2,  #bicycle
        2: 2,  #motorcycle
        3: 2,  #car
        5: 2,  #bus
        7: 2,  #truck
        9: 9,  #traffic light
        11: 11, #stop sign
    }   
        
    def __init__(self, gt_dir, detection_dir, cameras:list):
        self.cameras = cameras
        self.gt = self.get_gt(gt_dir)
        self.detections = self.get_detections(detection_dir)
        self.num_frames = len(self.gt)
        
    def get_gt(self, gt_dir) -> list:
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

    def get_detections(self, detection_dir) -> dict:
        all_detections = dict()
        for camera in self.cameras:
            camera_detection_dir = os.path.join(detection_dir, camera)
            detections = self.get_camera_detections(camera_detection_dir)
            all_detections[camera] = detections
        return all_detections
    
    def get_camera_detections(self, camera_detection_dir) -> list:
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
                    class_id = self.class_Map.get(int(parts[0]), -1)  #-1（無視するクラス）
                    if class_id == -1:
                        continue
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
    
    def unanimous_detections(self, detectinos) -> list:
        unanimous_detections = list()
        for i in range(self.num_frames):
            frame_unanimous = dict()
            base_detecions = self.detections[self.cameras[0]][i]
            for class_id, bboxes in base_detecions.items():
                for base_bbox in bboxes:
                    unanimous = True
                    for camera in self.cameras[1:]:
                        mathced = False
                        other_bboxes = self.detections[camera][i].get(class_id, [])
                        for other_bbox in other_bboxes:
                            if iou(base_bbox, other_bbox) > 0.5:
                                mathced = True
                                break
                        if not mathced:
                            print(f"not matched: {camera} {class_id} {base_bbox}")
                            unanimous = False
                            break
                    if unanimous:
                        if class_id not in frame_unanimous:
                            frame_unanimous[class_id] = list()
                        frame_unanimous[class_id].append(base_bbox)
            unanimous_detections.append(frame_unanimous)
            break
            
        return unanimous_detections
    
def main():
    gt_dir = 'C:/CARLA_Latest/WindowsNoEditor/output/label/Town01_Opt/front'
    detection_dir = 'C:/CARLA_Latest/WindowsNoEditor/ObjectDetection/yolov8_results/labels/Town01_Opt'
    cameras = os.listdir(detection_dir)
    
    dataset = Dataset(gt_dir, detection_dir, cameras)
    print(f"gt:{dataset.gt[0]}")
    print(f"front: {dataset.detections['front'][0]}")
    print(f"left_1: {dataset.detections['left_1'][0]}")
    print(f"right_1: {dataset.detections['right_1'][0]}")
    unanimous = dataset.unanimous_detections(dataset.detections)
    print(f"unanimous detections: {unanimous}")
    
    
        
if __name__ == "__main__":
    main()