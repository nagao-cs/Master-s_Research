import glob
import os
import sys
import cv2
import csv
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from utils import carla_util, camera_util
from utils.config import *
from queue import Queue

def image_to_depth(depth_image):
    array = np.frombuffer(depth_image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (depth_image.height, depth_image.width, 4))[:, :, :3]  # B, G, R
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
    u = point_img[0] / point_img[2]
    v = point_img[1] / point_img[2]
    if not (0 <= u < IM_WIDTH and 0 <= v < IM_HEIGHT):
        return None
    dist = point_img[2]
    return (u, v, dist)

def is_visible_bbox(bbox, camera, K, world_2_camera, depth_map, threshold_visible=1, eps=1.0):
    verts = [vert for vert in bbox.get_world_vertices(carla.Transform())]
    visible_count = 0

    for vert in verts:
        result = project_point(vert, K, world_2_camera)
        if result is None:
            continue
        u, v, dist = result
        print(f"u:{u}, v:{v}, dist:{dist}")
        print(f"depth_map[v, u]:{depth_map[v][u]}")
        print(abs(dist - depth_map[v][u]))
        if abs(dist - depth_map[v][u]) < eps:
            visible_count += 1
        
    return visible_count >= threshold_visible


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
    world = carla_util.apply_settings(world, synchronous_mode=True, fixed_delta_seconds=FIXED_DELTA_SECONDS)
    traffic_manager, tm_port = carla_util.setting_traffic_manager(client, synchronous_mode=True)
    spawn_points = world.get_map().get_spawn_points()
    print(f"spawn_points:{len(spawn_points)}")
    
    # === NPC車両スポーン ===
    vehicles = carla_util.spawn_npc_vehicles(world, blueprint_library, traffic_manager, spawn_points, CAR_RATIO)

    # === 歩行者スポーン ===
    pedestrians, walker_controllers = carla_util.spawn_npc_pedestrians(world, blueprint_library, NUM_WALKERS)

    # === Ego車両スポーン（最後） ===
    ego_bp = blueprint_library.find('vehicle.lincoln.mkz_2020')
    ego_vehicle = carla_util.spawn_Ego_vehicles(client, world, ego_bp, spawn_points)
    
    # === カメラセンサの設定 ===
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', str(IM_WIDTH))
    camera_bp.set_attribute('image_size_y', str(IM_HEIGHT))
    camera_bp.set_attribute('fov', str(FOV))
    camera_transform = carla.Transform(carla.Location(x=1.5, z=2.0))
    camera_queue = Queue()
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=ego_vehicle)
    camera.listen(camera_queue.put)
    print("Camera sensor is set up.")
    
    # 深度センサの設定
    depth_bp = blueprint_library.find('sensor.camera.depth')
    depth_bp.set_attribute('image_size_x', str(IM_WIDTH))
    depth_bp.set_attribute('image_size_y', str(IM_HEIGHT))
    depth_bp.set_attribute('fov', str(FOV))
    depth_transform = carla.Transform(carla.Location(x=1.5, z=2.0))
    depth_queue = Queue()
    depth_camera = world.spawn_actor(depth_bp, depth_transform, attach_to=ego_vehicle)
    depth_camera.listen(depth_queue.put)
    print("Depth sensor is set up.")
    
    # カメラ行列の計算
    K = camera_util.build_projection_matrix(IM_WIDTH, IM_HEIGHT, FOV)
    
    # === シミュレーション開始 ===
    ego_vehicle.set_autopilot(True)
    try:
        duration_sec = TIME_DURATION
        num_frames = int(duration_sec / FIXED_DELTA_SECONDS)
        for frame_idx in range(num_frames):
            world.tick() # シミュレーションを進める

            # === 深度画像を取得して距離マップに変換 ===
            depth_image = depth_queue.get()
            depth_map = image_to_depth(depth_image)
            
            # === カメラ画像を取得 ===
            camera_image = camera_queue.get()
            camera_image_array = np.frombuffer(camera_image.raw_data, dtype=np.uint8)
            camera_image_array = camera_image_array.reshape((camera_image.height, camera_image.width, 4))[:, :, :4]

            world_to_camera = np.array(camera.get_transform().get_inverse_matrix())

            # === 深度データとbboxの距離から視認できるbboxを抽出 ===
            visible_bboxes = []
            camera_transform = camera.get_transform()
            camera_location = camera_transform.location
            camera_forward_vector = camera_transform.get_forward_vector()
            for bbox in world.get_level_bbs(carla.CityObjectLabel.Vehicles):
                ray = bbox.location - camera_location
                if camera_forward_vector.dot(ray) < 0:
                    continue
                if bbox.location.distance(camera.get_location()) > 50.0:
                    continue
                if is_visible_bbox(bbox, camera, K, world_to_camera, depth_map, eps=0.3):
                    # print(bbox)
                    verts = bbox.get_world_vertices(carla.Transform())
                    points_2d_on_image = []
                    for vert in verts:
                        p = camera_util.get_image_point(vert, K, world_to_camera)
                        if p is not None:
                            points_2d_on_image.append(p)
                    yolo_bbox = camera_util.calculate_yolo_bbox(points_2d_on_image, IM_WIDTH, IM_HEIGHT)
                    if yolo_bbox:
                        xmin, xmax, ymin, ymax = yolo_bbox
                        visible_bboxes.append([xmin, ymin, xmax, ymax])
            
            # === 画像に視認可能なbboxを描画 ===
            for bbox in visible_bboxes:
                xmin, ymin, xmax, ymax = bbox
                xmin = int(xmin)
                ymin = int(ymin)
                xmax = int(xmax)
                ymax = int(ymax)
                cv2.rectangle(camera_image_array, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.imshow('Camera View', camera_image_array)
            cv2.imshow('Depth View', depth_map.astype(np.uint8))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("ユーザーによりシミュレーションが停止されました。")
                break
    finally:
        # アクターの破棄
        camera.stop()
        camera.destroy()
        depth_camera.stop()
        depth_camera.destroy()
        ego_vehicle.destroy()
        for vehicle in vehicles:
            vehicle.destroy()
        for pedestrian in pedestrians:
            pedestrian.destroy()
        for controller in walker_controllers:
            controller.stop()
            controller.destroy()
        print("シミュレーションが終了しました。")
        cv2.destroyAllWindows()
if __name__ == '__main__':
    main()