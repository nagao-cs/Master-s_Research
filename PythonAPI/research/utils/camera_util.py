import carla
from queue import Queue
import math
import numpy as np
import cv2
import os
from utils.config import IM_WIDTH, IM_HEIGHT, FOV, VALID_DISTANCE, CLASS_MAPPING, XMIN, XMAX, YMIN, YMAX, DIST, SIZE_THRESHOLD


def setting_camera(world, bp_library, ego_vehicle, im_width, im_height, fov, num_camera):
    cameras = list()
    queues = [Queue() for _ in range(num_camera)]
    for i in range(num_camera):
        camera_bp = bp_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(im_width))
        camera_bp.set_attribute('image_size_y', str(im_height))
        camera_bp.set_attribute('fov', str(fov))
        # camera_bp.set_attribute('sensor_tick', '1.0')
        if i == 0:
            camera_bp.set_attribute('role_name', 'front')
            camera_transform = carla.Transform(
                carla.Location(x=1.5, y=0.0, z=2.0))
        elif i % 2 == 1:
            number = math.ceil(i / 2)
            camera_bp.set_attribute('role_name', f'right_{number}')
            camera_transform = carla.Transform(
                carla.Location(x=1.5, y=0.3*(number), z=2.0))
        else:
            number = math.ceil(i / 2)
            camera_bp.set_attribute('role_name', f'left_{number}')
            camera_transform = carla.Transform(
                carla.Location(x=1.5, y=-0.3*(number), z=2.0))
        camera = world.spawn_actor(
            camera_bp, camera_transform, attach_to=ego_vehicle)
        que = queues[i]
        camera.listen(que.put)
        cameras.append(camera)
    print(f"{len(cameras)} 台のカメラをスポーン")
    return cameras, queues


def setting_depth_camera(world, bp_library, ego_vehicle, im_width, im_height, fov, num_camera):
    depth_cameras = list()
    depth_ques = list()
    depth_bp = bp_library.find('sensor.camera.depth')
    depth_bp.set_attribute('image_size_x', str(im_width))
    depth_bp.set_attribute('image_size_y', str(im_height))
    depth_bp.set_attribute('fov', str(fov))
    # depth_bp.set_attribute('sensor_tick', '1.0')
    for i in range(num_camera):
        if i == 0:
            depth_transform = carla.Transform(
                carla.Location(x=1.5, y=0.0, z=2.0))
        elif i % 2 == 1:
            number = math.ceil(i / 2)
            depth_transform = carla.Transform(
                carla.Location(x=1.5, y=0.3*(number), z=2.0))
        else:
            number = math.ceil(i / 2)
            depth_transform = carla.Transform(
                carla.Location(x=1.5, y=-0.3*(number), z=2.0))
        depth_camera = world.spawn_actor(
            depth_bp, depth_transform, attach_to=ego_vehicle)
        q = Queue()
        depth_camera.listen(q.put)
        depth_cameras.append(depth_camera)
        depth_ques.append(q)
    print(f"{len(depth_cameras)} 台の深度カメラをスポーン")
    return depth_cameras, depth_ques


def create_save_queues(num_camera):
    row_image_ques = [Queue() for _ in range(num_camera)]
    bbox_image_ques = [Queue() for _ in range(num_camera)]
    label_ques = [Queue() for _ in range(num_camera)]
    return row_image_ques, bbox_image_ques, label_ques


def build_projection_matrix(w, h, fov, is_behind_camera=False):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)

    if is_behind_camera:
        K[0, 0] = K[1, 1] = -focal
    else:
        K[0, 0] = K[1, 1] = focal

    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K


def get_image_point(loc, K, w2c):
    # 3D座標の2D投影を計算
    point = np.array([loc.x, loc.y, loc.z, 1])
    # カメラ座標に変換
    point_camera = np.dot(w2c, point)

    # UE4の座標系から「標準」の座標系 (x, y ,z) -> (y, -z, x) に変更
    point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

    # カメラ行列を使用して3D->2Dに投影
    point_img = np.dot(K, point_camera)
    # 正規化
    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]

    return point_img[0:2]


def calculate_yolo_bbox(points_2d, img_width, img_height):
    if not points_2d:
        return None

    points_2d = np.array(points_2d)

    # x, y 座標を画像の範囲内にクリップ
    xmin = np.clip(np.min(points_2d[:, 0]), 0, img_width - 1)
    ymin = np.clip(np.min(points_2d[:, 1]), 0, img_height - 1)
    xmax = np.clip(np.max(points_2d[:, 0]), 0, img_width - 1)
    ymax = np.clip(np.max(points_2d[:, 1]), 0, img_height - 1)

    # クリッピング後も有効なボックスか確認
    if xmax <= xmin or ymax <= ymin:
        return None

    # 正規化
    xmin, xmax, ymin, ymax = xmin/img_width, xmax / \
        img_width, ymin/img_height, ymax/img_height

    x_center = (xmin + xmax) / 2.0
    y_center = (ymin + ymax) / 2.0
    width = xmax - xmin
    height = ymax - ymin

    return (x_center, y_center, width, height)


def process_camera_data(image, camera_actor, world, K, K_b, display_window_name):
    # RGB配列に整形
    img_display = image.copy()  # バウンディングボックスを描画用

    # ワールドからカメラへの変換行列を取得
    world_to_camera = np.array(
        camera_actor.get_transform().get_inverse_matrix())

    camera_transform = camera_actor.get_transform()
    camera_location = camera_transform.location
    camera_forward_vec = camera_transform.get_forward_vector()

    # 現在のフレームのラベルデータを格納するリスト
    frame_labels = list()

    # CityObjectLabels (TrafficLight, TrafficSigns) の処理
    city_object_categories = [
        carla.CityObjectLabel.TrafficLight,
        carla.CityObjectLabel.TrafficSigns,
        carla.CityObjectLabel.Vehicles,
        carla.CityObjectLabel.Pedestrians,
        # carla.CityObjectLabel.Buildings,
        # carla.CityObjectLabel.Vegetation,
        # carla.CityObjectLabel.Walls
    ]
    for category_label in city_object_categories:
        boundingboxes = world.get_level_bbs(category_label)
        for bbox in boundingboxes:
            dist = bbox.location.distance(camera_location)

            if dist < VALID_DISTANCE:
                ray = bbox.location - camera_location

                if camera_forward_vec.dot(ray) > 0:  # カメラの視野角内（前方）にあるかを確認
                    verts = [v for v in bbox.get_world_vertices(
                        carla.Transform())]
                    points_2d_on_image = []

                    for vert in verts:
                        ray_vert = vert - camera_location
                        if camera_forward_vec.dot(ray_vert) > 0:
                            p = get_image_point(vert, K, world_to_camera)
                            points_2d_on_image.append(p)

                    yolo_bbox = calculate_yolo_bbox(
                        points_2d_on_image, IM_WIDTH, IM_HEIGHT)

                    if yolo_bbox:
                        xmin, xmax, ymin, ymax = yolo_bbox
                        size = (xmax - xmin) * (ymax - ymin)
                        if size > SIZE_THRESHOLD:
                            # print(size)
                            class_id = CLASS_MAPPING[category_label]
                            frame_labels.append(
                                [class_id, xmin, xmax, ymin, ymax, dist])

    # # bboxをdist,xminとyminでソート
    frame_labels.sort(key=lambda bbox: (bbox[XMIN], bbox[YMIN]))
    # # ほかのbboxに隠れるbboxを除外
    frame_labels = remove_overlapping_bboxes(frame_labels)
    # # bboxを描画
    for bbox in frame_labels:
        class_id, xmin, xmax, ymin, ymax, dist = bbox
        xmin_draw = int(xmin)
        xmax_draw = int(xmax)
        ymin_draw = int(ymin)
        ymax_draw = int(ymax)
        color = (0, 255, 0) if class_id in [CLASS_MAPPING[carla.CityObjectLabel.TrafficLight], CLASS_MAPPING[carla.CityObjectLabel.TrafficSigns]] else (
            255, 0, 0) if class_id == CLASS_MAPPING[carla.CityObjectLabel.Vehicles] else (0, 0, 255)
        cv2.rectangle(img_display, (xmin_draw, ymin_draw),
                      (xmax_draw, ymax_draw), color, 2)
        cv2.putText(img_display, f"{int(class_id)}", (xmin_draw,
                    ymin_draw - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    # 画像を表示
    cv2.imshow(display_window_name, img_display)

    return frame_labels, img_display


def is_contained(inner, outer):
    # inner, outer: [class_id, xmin, xmax, ymin, ymax, dist]
    return (outer[1] <= inner[1] and outer[2] >= inner[2] and
            outer[3] <= inner[3] and outer[4] >= inner[4])


def remove_overlapping_bboxes(bboxes):
    if not bboxes:
        return []
    filtered = []
    for i, bbox in enumerate(bboxes):
        contained = False
        for j, other in enumerate(bboxes):
            if i == j:
                continue
            # より手前（distが小さい）で完全に覆われている場合
            if other[DIST] < bbox[DIST] and is_contained(bbox, other):
                contained = True
                break
        if not contained and bbox[0] != -1:  # -1は無視するクラス
            filtered.append(bbox)
    return filtered


def save_images(image_queues, cameras, output_dir, suffix=''):
    for i, cam_q in enumerate(image_queues):
        camera = cameras[i]
        image_dir = f"{output_dir}/{camera.attributes['role_name']}{suffix}"
        os.makedirs(image_dir, exist_ok=True)
        print(f"{camera.attributes['role_name']} の画像を保存しています...")
        num_frame = 0
        while not cam_q.empty():
            image = cam_q.get()
            image_path = os.path.join(image_dir, f"{num_frame:06d}.png")
            image.save_to_disk(image_path)
            num_frame += 1
    print("すべての画像を保存しました。")
