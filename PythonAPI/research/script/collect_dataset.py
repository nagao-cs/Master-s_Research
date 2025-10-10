
import numpy as np
import csv
import cv2
import sys
import os
import glob
from queue import Queue
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../')))


def image_to_depth(depth_image):
    array = np.frombuffer(depth_image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (depth_image.height, depth_image.width, 4))[
        :, :, :3]  # B, G, R
    B = array[:, :, 0].astype(np.uint32)
    G = array[:, :, 1].astype(np.uint32)
    R = array[:, :, 2].astype(np.uint32)

    normalized = (R + G * 256 + B * 256 * 256) / float(256**3 - 1)
    depth_in_meters = normalized * 1000.0  # convert to meters
    return depth_in_meters


def project_point(vert, K, w2c):
    point = np.array([vert.x, vert.y, vert.z, 1.0])
    point_camera = np.dot(w2c, point)
    # Unreal Engine座標系→OpenCV座標系
    point_camera = [point_camera[1], -point_camera[2], point_camera[0]]
    point_img = np.dot(K, point_camera)
    u = int(round(point_img[0] / point_img[2]))
    v = int(round(point_img[1] / point_img[2]))
    if not (0 <= u < IM_WIDTH and 0 <= v < IM_HEIGHT):
        return None
    dist = point_img[2]
    return (u, v, dist)


def is_visible_bbox(bbox, camera, K, world_2_camera, depth_map, threshold_visible=2, eps=1.0):
    verts = [vert for vert in bbox.get_world_vertices(carla.Transform())]
    visible_count = 0

    for vert in verts:
        result = project_point(vert, K, world_2_camera)
        if result is None:
            continue
        u, v, dist = result
        # print(f"u:{u}, v:{v}, dist:{dist}")
        # print(f"depth_map[v, u]:{depth_map[v][u]}")
        if dist < depth_map[v][u]+eps:
            visible_count += 1

    return visible_count >= threshold_visible


def save_images(images, dir_path):
    os.makedirs(dir_path, exist_ok=True)
    last_index = len(os.listdir(dir_path))
    for i, image in enumerate(images):
        index = last_index + i
        image_path = os.path.join(dir_path, f"{index:06d}.png")
        cv2.imwrite(image_path, image)


def save_labels(labels, dir_path):
    os.makedirs(dir_path, exist_ok=True)
    last_index = len(os.listdir(dir_path))
    for i, label in enumerate(labels):
        index = last_index + i
        label_path = os.path.join(dir_path, f"{index:06d}.txt")
        with open(label_path, 'w') as f:
            for bbox in label:
                class_id, x_center, y_center, width, height = bbox
                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")


def save_dataset(images, labels, image_dir, label_dir):
    save_images(images, image_dir)
    save_labels(labels, label_dir)
    print(
        f"Saved {len(images)} images and {len(labels)} labels to {image_dir} and {label_dir}")


# === Carla Egg のパス設定 ===
try:
    sys.path.append(glob.glob('C:/CARLA_Latest/WindowsNoEditor/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64'))[0])
except IndexError:
    pass


def main():
    # === サーバに接続し、基本的なセッティング ===
    client = carla_util.connect_to_server(CARLA_HOST, CARLA_PORT, TIMEOUT)
    world, blueprint_library = carla_util.load_map(client, MAP)
    world = carla_util.apply_settings(
        world, synchronous_mode=True, fixed_delta_seconds=FIXED_DELTA_SECONDS)
    traffic_manager, tm_port = carla_util.setting_traffic_manager(
        client, synchronous_mode=True)
    spawn_points = world.get_map().get_spawn_points()
    print(f"spawn_points:{len(spawn_points)}")

    # === 天候の設定 ===
    weather = world.get_weather()
    print(f"Current weather: {weather}")
    # weather.cloudiness = 20.0
    # weather.precipitation = 0.0
    # weather.precipitation_deposits = 0.0
    # weather.wind_intensity = 10.0
    # weather.fog_density = 0.0
    # weather.fog_distance = 0.0
    # weather.fog_falloff = 0.0
    # weather.wetness = 0.0
    # weather.sun_azimuth_angle = 180.0
    # weather.sun_altitude_angle = 70.0
    # world.set_weather(weather)

    # === NPC車両スポーン ===
    vehicles = carla_util.spawn_npc_vehicles(
        world, blueprint_library, traffic_manager, spawn_points, CAR_RATIO)

    # === 歩行者スポーン ===
    all_actors = carla_util.spawn_npc_pedestrians(
        world, client, blueprint_library, NUM_WALKERS)

    # === Ego車両スポーン（最後） ===
    ego_bp = blueprint_library.find('vehicle.lincoln.mkz_2020')
    ego_vehicle = carla_util.spawn_Ego_vehicles(
        client, world, ego_bp, spawn_points)

    # === カメラセンサの設定 ===
    cameras, camera_image_queues = camera_util.setting_camera(
        world, blueprint_library, ego_vehicle, IM_WIDTH, IM_HEIGHT, FOV, NUM_CAMERA=1)

    # 深度センサの設定
    depth_cameras, depth_queues = camera_util.setting_depth_camera(
        world, blueprint_library, ego_vehicle, IM_WIDTH, IM_HEIGHT, FOV, NUM_CAMERA=1)

    # カメラ行列の計算
    K = camera_util.build_projection_matrix(IM_WIDTH, IM_HEIGHT, FOV)

    # === 保存用のキューを作成 ===
    original_images = list()
    labels = list()

    # === ターゲットオブジェクトの設定 ===
    target_objects = [
        carla.CityObjectLabel.Vehicles,
        carla.CityObjectLabel.Pedestrians,
        carla.CityObjectLabel.TrafficSigns,
        carla.CityObjectLabel.TrafficLight,
    ]

    # === シミュレーション開始 ===
    ego_vehicle.set_autopilot(True)
    import time
    start = time.time()
    try:
        duration_sec = TIME_DURATION
        num_frames = int(duration_sec / FIXED_DELTA_SECONDS)
        for frame_idx in range(num_frames):
            world.tick()  # シミュレーションを進める

            idx = 0
            # === RGBカメラと深度カメラを取得 ===
            camera = cameras[idx]
            depth_camera = depth_cameras[idx]
            camera_queue = camera_image_queues[idx]
            depth_queue = depth_queues[idx]

            # === RGB画像と深度画像を取得 ===
            original_image = camera_queue.get()
            depth_image = depth_queue.get()

            # === RGB画像を変換 ===
            # image_array = np.frombuffer(original_image.raw_data, dtype=np.uint8)
            # original_image = image_array.reshape((original_image.height, original_image.width, 4))[:, :, :4]
            original_image = np.reshape(np.copy(
                original_image.raw_data), (original_image.height, original_image.width, 4))
            bbox_image = original_image.copy()

            # === 深度画像を距離マップに変換 ===
            depth_map = image_to_depth(depth_image)
            # depth_show  = depth_map.copy()
            # 最大表示距離を設定（例: 50m）
            # max_depth = 150
            # depth_vis = np.clip(depth_map, 0, max_depth)
            # depth_vis = (depth_vis / max_depth * 255).astype(np.uint8)
            # cv2.imshow(f'Depth Map {camera.attributes["role_name"]}', depth_vis)

            # === カメラの位置と向きを取得 ===
            camera_transform = camera.get_transform()
            camera_location = camera_transform.location
            camera_forward_vector = camera_transform.get_forward_vector()

            # === カメラのワールド座標系からカメラ座標系への変換行列を取得 ===
            world_to_camera = np.array(
                camera.get_transform().get_inverse_matrix())

            # === 距離マップとbboxの距離から視認可能なbboxを抽出 ===
            visible_bboxes = list()
            for target in target_objects:
                bboxes = world.get_level_bbs(target)
                for bbox in bboxes:
                    ray = bbox.location - camera_location
                    if camera_forward_vector.dot(ray) < 0:
                        continue
                    if bbox.location.distance(camera.get_location()) > 100.0:
                        continue
                    if is_visible_bbox(bbox, camera, K, world_to_camera, depth_map, eps=0.3):
                        verts = bbox.get_world_vertices(carla.Transform())
                        points_2d_on_image = []
                        for vert in verts:
                            p = camera_util.get_image_point(
                                vert, K, world_to_camera)
                            if p is not None:
                                points_2d_on_image.append(p)
                        yolo_bbox = camera_util.calculate_yolo_bbox(
                            points_2d_on_image, IM_WIDTH, IM_HEIGHT)
                        if yolo_bbox:
                            # xmin, xmax, ymin, ymax = yolo_bbox
                            # class_id = CLASS_MAPPING.get(target, -1)
                            # size = (xmax - xmin) * (ymax - ymin)
                            # if size < SIZE_THRESHOLD:
                            #     continue
                            # visible_bboxes.append(
                            #     [class_id, xmin, xmax, ymin, ymax])
                            x_center, y_center, width, height = yolo_bbox
                            class_id = camera_util.CLASS_MAPPING.get(
                                target, -1)
                            visible_bboxes.append(
                                [class_id, x_center, y_center, width, height])

            # === 画像に視認可能なbboxを描画 ===
            for bbox in visible_bboxes:
                class_id, x_center, y_center, width, height = bbox
                xmin = int((x_center - width / 2) * IM_WIDTH)
                xmax = int((x_center + width / 2) * IM_WIDTH)
                ymin = int((y_center - height / 2) * IM_HEIGHT)
                ymax = int((y_center + height / 2) * IM_HEIGHT)
                cv2.rectangle(bbox_image, (xmin, ymin),
                              (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(bbox_image, f'{class_id}', (xmin, ymin - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # === 画像を表示 ===
            display_name = f'{camera.attributes["role_name"]} with Bounding Boxes'
            cv2.imshow(display_name, bbox_image)

            # 元画像、バウンディングボックス画像、ラベルを保存用キューに追加
            original_images.append(original_image)
            labels.append(visible_bboxes)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("ユーザーによりシミュレーションが停止されました。")
                break
    finally:
        print("シミュレーションが終了しました。")
        os.makedirs("C:\CARLA_Latest\WindowsNoEditor\\raw_dataset",
                    exist_ok=True)
        raw_image_dir = "C:\CARLA_Latest\WindowsNoEditor\\raw_dataset\images"
        raw_label_dir = "C:\CARLA_Latest\WindowsNoEditor\\raw_dataset\labels"
        save_dataset(original_images, labels,
                     raw_image_dir, raw_label_dir)

        # === クリーンアップ ===
        carla_util.cleanup(client, world, vehicles,
                           all_actors, cameras, depth_cameras)
        print("シミュレーションが終了しました。")
        cv2.destroyAllWindows()
        end = time.time()
        print(f"シミュレーションにかかった時間: {end - start:.2f}秒")

        # print(f"画像保存にかかった時間: {save_end - save_start:.2f}秒")
if __name__ == '__main__':
    from utils.config import *
    from utils import carla_util, camera_util
    main()
