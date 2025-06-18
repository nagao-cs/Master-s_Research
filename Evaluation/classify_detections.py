import os 
import pickle

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

class Evaluation:
    def __init__(self, dataset):
        self.dataset = dataset
        
    def cov_od(self):
        total_obj = self.dataset.total_objects()
        common_fp = self.dataset.common_false_positives()
        common_fn = self.dataset.common_false_negatives()
        cov_od = 0
        for frame in range(self.dataset.num_frames):
            frame_obj = total_obj[frame]
            num_obj = sum(len(bboxes) for bboxes in frame_obj.values())
            
            frame_fp = common_fp[frame]
            num_fp = sum(len(bboxes) for bboxes in frame_fp.values())
            
            frame_fn = common_fn[frame]
            num_fn = sum(len(bboxes) for bboxes in frame_fn.values())

            if num_obj == 0:
                cov_od += 0
            else:
                cov_od += (num_fp+num_fn)/(num_obj)
        return 1-(cov_od/self.dataset.num_frames)
    
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
                    xmin = int(float(parts[1]))
                    xmax = int(float(parts[2]))
                    ymin = int(float(parts[3]))
                    ymax = int(float(parts[4]))
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
                    if camera_detection_dir.endswith('front'):
                        offset = 0
                    elif camera_detection_dir.endswith('left_1'):
                        offset = -22
                    elif camera_detection_dir.endswith('right_1'):
                        offset = 22
                    xmin = int(float(parts[1])) + offset
                    xmax = int(float(parts[2])) + offset
                    ymin = int(float(parts[3]))
                    ymax = int(float(parts[4]))
                    confidence = float(parts[5])
                    if class_id not in frame_detections:
                        frame_detections[class_id] = list()
                    frame_detections[class_id].append((xmin, xmax, ymin, ymax, confidence))
                detections.append(frame_detections)
        return detections
    
    def affirmative_detections(self) -> list:
        affirmative_detections = list()
        for i in range(self.num_frames):
            frame_affirmative = dict()
            for camera in self.cameras:
                buffer = dict()
                for class_id, bboxes in self.detections[camera][i].items():
                    if class_id not in frame_affirmative:
                        buffer[class_id] = bboxes
                        continue
                    for bbox in bboxes:
                        matched = False
                        for other_bbox in frame_affirmative[class_id]:
                            if iou(bbox, other_bbox) > 0.5:
                                matched = True
                                break
                        if not matched:
                            if class_id not in buffer:
                                buffer[class_id] = list()
                            buffer[class_id].append(bbox)
                for class_id, bboxes in buffer.items():
                    if class_id not in frame_affirmative:
                        frame_affirmative[class_id] = list()
                    for bbox in bboxes:
                        frame_affirmative[class_id].append(bbox)
            affirmative_detections.append(frame_affirmative)
        return affirmative_detections
    
    def unanimous_detections(self) -> list:
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
                            unanimous = False
                            break
                    if unanimous:
                        if class_id not in frame_unanimous:
                            frame_unanimous[class_id] = list()
                        frame_unanimous[class_id].append(base_bbox)
            unanimous_detections.append(frame_unanimous)
            
        return unanimous_detections
    
    def common_false_positives(self) -> list:
        common_fp = list()
        for i in range(self.num_frames):
            frame_fp = dict()
            camera_fps = [self.camera_false_positives(camera, i) for camera in self.cameras]
            if min(len(camera_fp) for camera_fp in camera_fps) == 0:
                common_fp.append(frame_fp)
                continue
            base_fps = camera_fps[0]
            for class_id, bboxes in base_fps.items():
                for bbox in bboxes:
                    unanimous = True
                    for other_camera_fns in camera_fps[1:]:
                        matched = False
                        other_bboxes = other_camera_fns.get(class_id, [])
                        for other_bbox in other_bboxes:
                            if iou(bbox, other_bbox) > 0.5:
                                matched = True
                                break
                        if not matched:
                            unanimous = False
                            break
                    if unanimous:
                        if class_id not in frame_fp:
                            frame_fp[class_id] = list()
                        frame_fp[class_id].append(bbox)    
            common_fp.append(frame_fp)
        return common_fp
    
    def affirmative_false_positives(self) -> list:
        affirmative_fps = list()
        for frame in range(self.num_frames):
            frame_fp = dict()
            camera_fps = [self.camera_false_positives(camera, frame) for camera in self.cameras]
            for i in range(len(self.cameras)):
                buffer = dict()
                for class_id, bboxes in camera_fps[i].items():
                    if class_id not in frame_fp:
                        buffer[class_id] = bboxes
                        continue
                    for bbox in bboxes:
                        matched = False
                        for other_bbox in frame_fp[class_id]:
                            if iou(bbox, other_bbox) > 0.5:
                                matched = True
                                break
                        if not matched:
                            if class_id not in buffer:
                                buffer[class_id] = list()
                            buffer[class_id].append(bbox)
                for class_id, bboxes in buffer.items():
                    if class_id not in frame_fp:
                        frame_fp[class_id] = list()
                    for bbox in bboxes:
                        frame_fp[class_id].append(bbox)
            affirmative_fps.append(frame_fp)
        
        return affirmative_fps
    
    def camera_false_positives(self, camera, frame) -> dict:
        frame_detections = self.detections[camera][frame]
        frame_gt = self.gt[frame]
        frame_fp = dict()
        for class_id, bboxes in frame_detections.items():
            if class_id not in frame_gt:
                frame_fp[class_id] = bboxes
            else:
                for bbox in bboxes:
                    matched = False
                    for gt_bbox in frame_gt[class_id]:
                        if iou(bbox, gt_bbox) > 0.5:
                            matched = True
                            break
                    if not matched:
                        if class_id not in frame_fp:
                            frame_fp[class_id] = list()
                        frame_fp[class_id].append(bbox)
        return frame_fp
    
    def common_false_negatives(self) -> list:
        common_fn = list()
        for i in range(self.num_frames):
            frame_fn = dict()
            camera_fps = [self.camera_false_negatives(camera, i) for camera in self.cameras]
            if min(len(camera_fp) for camera_fp in camera_fps) == 0:
                common_fn.append(frame_fn)
                continue
            base_fns = camera_fps[0]
            for class_id, bboxes in base_fns.items():
                for bbox in bboxes:
                    unanimous = True
                    for other_camera_fns in camera_fps[1:]:
                        matched = False
                        other_bboxes = other_camera_fns.get(class_id, [])
                        for other_bbox in other_bboxes:
                            if iou(bbox, other_bbox) > 0.5:
                                matched = True
                                break
                        if not matched:
                            unanimous = False
                            break
                    if unanimous:
                        if class_id not in frame_fn:
                            frame_fn[class_id] = list()
                        frame_fn[class_id].append(bbox)    
            common_fn.append(frame_fn)
        return common_fn
    
    def camera_false_negatives(self, camera, frame) -> dict:
        frame_detections = self.detections[camera][frame]
        frame_gt = self.gt[frame]
        frame_fn = dict()
        for class_id, bboxes in frame_gt.items():
            if class_id not in frame_detections:
                frame_fn[class_id] = bboxes
            else:
                for gt_bbox in bboxes:
                    matched = False
                    for bbox in frame_detections[class_id]:
                        if iou(gt_bbox, bbox) > 0.5:
                            matched = True
                            break
                    if not matched:
                        if class_id not in frame_fn:
                            frame_fn[class_id] = list()
                        frame_fn[class_id].append(gt_bbox)
        return frame_fn

    def total_objects(self) -> list:
        total_objects = list()
        fps = self.affirmative_false_positives()
        for frame in range(self.num_frames):
            frame_fp = fps[frame]
            gt = self.gt[frame]
            frame_object = dict()
            for class_id, bboxes in frame_fp.items():
                frame_object[class_id] = bboxes
            for class_id, bboxes in gt.items():
                if class_id not in frame_object:
                    frame_object[class_id] = list()
                frame_object[class_id].extend(bboxes)
            total_objects.append(frame_object)
        return total_objects
            
    
def main():
    gt_dir = 'C:/CARLA_Latest/WindowsNoEditor/output/label/Town01_Opt/front'
    detection_dir = 'C:/CARLA_Latest/WindowsNoEditor/ObjectDetection/yolov8_results/labels/Town01_Opt'
    num_versions = [1, 2, 3]
    for num_version in num_versions:
        # データセットを一度だけロードし、キャッシュする
        cameras = os.listdir(detection_dir)[:num_version]
        cache_file = f'dataset_{len(cameras)}_cache.pkl'
        if os.path.exists(cache_file):
            print(f"Loading dataset from cache: {cache_file}")
            with open(cache_file, 'rb') as f:
                dataset = pickle.load(f)
        else:
            print("Loading dataset from raw files...")
            dataset = Dataset(gt_dir, detection_dir, cameras)
            with open(cache_file, 'wb') as f:
                pickle.dump(dataset, f)
            print(f"Dataset cached to {cache_file}")
        dataset = Dataset(gt_dir, detection_dir, cameras)    
        eval = Evaluation(dataset)
        print(f"{num_version} version")
        print(f"    {eval.cov_od()}")
    
if __name__ == "__main__":
    main()