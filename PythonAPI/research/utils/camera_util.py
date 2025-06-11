import carla
from queue import Queue
import math
import numpy as np
import cv2
from utils.config import IM_WIDTH, IM_HEIGHT, FOV, VALID_DISTANCE, CLASS_MAPPING

def setting_camera(world, bp_library, ego_vehicle, im_width, im_height, fov, num_camera):
    cameras = list()
    ques = list()
    for i in range(num_camera):
        camera_bp = bp_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(im_width))
        camera_bp.set_attribute('image_size_y', str(im_height))
        camera_bp.set_attribute('fov', str(fov))
        if i == 0:
            camera_bp.set_attribute('role_name', 'front')
            camera_transform = carla.Transform(carla.Location(x=1.5, y=0.0, z=2.0))
        elif i % 2 == 1:
            number = math.ceil(i / 2)
            camera_bp.set_attribute('role_name', f'right_{number}')
            camera_transform = carla.Transform(carla.Location(x=1.5, y=0.3*(number), z=2.0))
        else:
            number = math.ceil(i / 2)
            camera_bp.set_attribute('role_name', f'left_{number}')
            camera_transform = carla.Transform(carla.Location(x=1.5, y=-0.3*(number), z=2.0))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=ego_vehicle)
        q = Queue()
        camera.listen(q.put)
        cameras.append(camera)
        ques.append(q)
        print(f"{camera_bp.get_attribute('role_name')} カメラをスポーン")
    print(f"{len(cameras)} 台のカメラをスポーン")
    #debug
    for camera in cameras:
        print(f"role:{camera.attributes['role_name']}, Location: {camera.get_location()}")
    return cameras, ques

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

    # bbox_width_pixels = xmax - xmin
    # bbox_height_pixels = ymax - ymin

    # center_x_pixels = (xmin + xmax) / 2.0
    # center_y_pixels = (ymin + ymax) / 2.0

    # # 正規化
    # center_x = center_x_pixels / img_width
    # center_y = center_y_pixels / img_height
    # bbox_width = bbox_width_pixels / img_width
    # bbox_height = bbox_height_pixels / img_height

    return (xmin, xmax, ymin, ymax)

def process_camera_data(image, camera_actor, world, K, K_b, ego_vehicle,display_window_name):
    # RGB配列に整形
    img_raw = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
    img_display = img_raw.copy() # デバッグ表示用 (バウンディングボックスを描画)

    # ワールドからカメラへの変換行列を取得
    world_to_camera = np.array(camera_actor.get_transform().get_inverse_matrix())
    
    camera_transform = camera_actor.get_transform()
    camera_location = camera_transform.location
    camera_forward_vec = camera_transform.get_forward_vector()
    
    # 現在のフレームのラベルデータを格納するリスト
    frame_labels = list()

    # CityObjectLabels (TrafficLight, TrafficSigns) の処理
    city_object_categories = [carla.CityObjectLabel.TrafficLight, carla.CityObjectLabel.TrafficSigns, carla.CityObjectLabel.Pedestrians]
    for category_label in city_object_categories:
        boundingboxes = world.get_level_bbs(category_label)
        for bbox in boundingboxes:
            dist = bbox.location.distance(camera_location)

            if dist < VALID_DISTANCE:
                ray = bbox.location - camera_location
                
                if camera_forward_vec.dot(ray) > 0: # カメラの視野角内（前方）にあるかを確認
                    verts = [v for v in bbox.get_world_vertices(carla.Transform())]
                    points_2d_on_image = []

                    for vert in verts:
                        ray_vert = vert - camera_location
                        if camera_forward_vec.dot(ray_vert) > 0: # 頂点がカメラの前にあれば通常の投影
                            p = get_image_point(vert, K, world_to_camera)
                        # else: # 頂点がカメラの後ろにあれば反転行列で投影（完全なバウンディングボックスのため）
                        #     p = get_image_point(vert, K_b, world_to_camera)
                        
                        points_2d_on_image.append(p)
                    
                    yolo_bbox = calculate_yolo_bbox(points_2d_on_image, IM_WIDTH, IM_HEIGHT)
                    
                    if yolo_bbox:
                        xmin, xmax, ymin, ymax = yolo_bbox
                        class_id = CLASS_MAPPING[category_label]
                        frame_labels.append([class_id, xmin, xmax, ymin, ymax])

    # VehicleとPedestrianの処理
    for actor in world.get_actors().filter('*vehicle*'):
        if actor.id == ego_vehicle.id:
            continue # 自車はスキップ
        
        bb = actor.bounding_box
        dist = actor.get_transform().location.distance(camera_location)
        
        if dist < VALID_DISTANCE:
            ray = actor.get_transform().location - camera_location
            
            if camera_forward_vec.dot(ray) > 0: # カメラの視野角内（前方）にあるかを確認
                verts = [v for v in bb.get_world_vertices(actor.get_transform())]
                points_2d_on_image = []

                for vert in verts:
                    ray_vert = vert - camera_location
                    if camera_forward_vec.dot(ray_vert) > 0:
                        p = get_image_point(vert, K, world_to_camera)
                    else:
                        p = get_image_point(vert, K_b, world_to_camera)
                    points_2d_on_image.append(p)
                
                yolo_bbox = calculate_yolo_bbox(points_2d_on_image, IM_WIDTH, IM_HEIGHT)

                if yolo_bbox:
                    xmin, xmax, ymin, ymax = yolo_bbox
                    
                    if 'vehicle' in actor.type_id:
                        class_id = CLASS_MAPPING[carla.CityObjectLabel.Vehicles]
                    elif 'walker' in actor.type_id:
                        class_id = CLASS_MAPPING[carla.CityObjectLabel.Pedestrians]
                    else:
                        print(f"Unknown actor type: {actor.type_id}")
                    
                    frame_labels.append([class_id, xmin, xmax, ymin, ymax])
    # bboxをxminとyminでソート
    frame_labels.sort(key=lambda x: (x[1], x[3]))
    # ほかのbboxに隠れるbboxを除外
    frame_labels = remove_overlapping_bboxes(frame_labels)
    # bboxを描画
    for bbox in frame_labels:
        class_id, xmin, xmax, ymin, ymax = bbox
        # xmin_draw = int((center_x - bbox_width / 2) * IM_WIDTH)
        # ymin_draw = int((center_y - bbox_height / 2) * IM_HEIGHT)
        # xmax_draw = int((center_x + bbox_width / 2) * IM_WIDTH)
        # ymax_draw = int((center_y + bbox_height / 2) * IM_HEIGHT)
        xmin_draw = int(xmin)
        xmax_draw = int(xmax)
        ymin_draw = int(ymin)
        ymax_draw = int(ymax)
        color = (0, 255, 0) if class_id in [CLASS_MAPPING[carla.CityObjectLabel.TrafficLight], CLASS_MAPPING[carla.CityObjectLabel.TrafficSigns]] else (255, 0, 0) if class_id == CLASS_MAPPING[carla.CityObjectLabel.Vehicles] else (0, 0, 255)
        cv2.rectangle(img_display, (xmin_draw, ymin_draw), (xmax_draw, ymax_draw), color, 2)
        cv2.putText(img_display, f"{int(class_id)}", (xmin_draw, ymin_draw - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    # 画像を表示
    cv2.imshow(display_window_name, img_display)
    
    return frame_labels, img_display

def remove_overlapping_bboxes(bboxes):
    if not bboxes:
        return []
    # bboxes[1]=xmin, bboxes[2]=xmax, bboxes[3]=ymin, bboxes[4]=ymax
    filtered_bboxes = [bboxes[0]]  # 最初のbboxはxminが最小なのでほかのbboxにたぶん隠れないと仮定
    prev = bboxes[0]
    for bbox in bboxes[1:]:
        # まずx座標についてチェック
        if bbox[2] <= prev[2]:
            # 現在のbboxがprev_xmin < bbox_xmin < bbox_xmax < prev_xmaxである
            # 次にy座標についてチェック
            if prev[3] < bbox[3] < prev[4] or prev[3] < bbox[4] < prev[4]:
                # 現在のbboxがprev_ymin < bbox_ymin < bbox_ymax < prev_ymaxである
                continue
            else:
                # 現在のbboxはprev_bboxに隠れていないので追加
                filtered_bboxes.append(bbox)
        else:
            # 現在のbboxはprev_xmax < bbox_xminである
            filtered_bboxes.append(bbox)
    
    return filtered_bboxes