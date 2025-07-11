import glob
import os
import sys
import cv2
import csv
from queue import Queue
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from utils import carla_util, camera_util
from utils.config import *


# === Carla Egg のパス設定 ===
try:
    sys.path.append(glob.glob('C:/CARLA_Latest/WindowsNoEditor/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64'))[0])
except IndexError:
    pass

def get_bbox_world_vertices(actor):
    bbox = actor.bounding_box
    transform = actor.get_transform()
    extent = bbox.extent

    # 8頂点（ローカル座標系）
    corners = [
        carla.Location(x, y, z)
        for x in [-extent.x, extent.x]
        for y in [-extent.y, extent.y]
        for z in [-extent.z, extent.z]
    ]

    # BBoxの中心位置をワールド座標に変換
    bbox_center_world = transform.transform(bbox.location)

    # ローカル→ワールド座標への変換を定義
    bbox_transform = carla.Transform(bbox_center_world, transform.rotation)

    # 各頂点をワールド座標に変換
    return [bbox_transform.transform(corner) for corner in corners]

def draw_2d_bbox_on_image(image, vertices, K, w2c):
    points_2d = []
    for v in vertices:
        p = camera_util.get_image_point(v, K, w2c)
        if p is not None:
            points_2d.append((int(p[0]), int(p[1])))
        else:
            return  # 1点でも変換失敗したらスキップ

    yolo_bbox = camera_util.calculate_yolo_bbox(points_2d, image.shape[1], image.shape[0])
    if yolo_bbox is not None:
        xmin, xmax, ymin, ymax = yolo_bbox
        cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 2)



def get_image_point(loc, K, world_2_camera):
    point = np.array([loc.x, loc.y, loc.z, 1.0])
    camera_coords = np.dot(world_2_camera, point)

    if camera_coords[2] <= 0:
        return None

    point_img = np.dot(K, camera_coords[:3] / camera_coords[2])
    return int(point_img[0]), int(point_img[1])


def draw_actor_bounding_box(world, actor, color=carla.Color(255, 0, 0), life_time=0.1):
    """
    CARLAのアクターのBoundingBoxをデバッグ描画する

    Parameters:
        world: carla.World
        actor: carla.Actor
        color: carla.Color (default 赤)
        life_time: float (表示時間秒, 0なら瞬間のみ)
    """
    bbox = actor.bounding_box
    bbox_transform = actor.get_transform().transform(bbox.location)
    world.debug.draw_box(
        box=bbox,
        rotation=actor.get_transform().rotation,
        thickness=0.1,
        life_time=life_time,
        color=color,
    )

def get_world_to_camera_matrix(camera):
    transform = camera.get_transform()
    matrix = transform.get_inverse_matrix()
    world_2_camera = np.array([
        [matrix[0][0], matrix[0][1], matrix[0][2], matrix[0][3]],
        [matrix[1][0], matrix[1][1], matrix[1][2], matrix[1][3]],
        [matrix[2][0], matrix[2][1], matrix[2][2], matrix[2][3]],
        [matrix[3][0], matrix[3][1], matrix[3][2], matrix[3][3]]
    ])
    return world_2_camera


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
    ego_id = ego_vehicle.id
    
    # === カメラセンサの設定 ===
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', str(IM_WIDTH))
    camera_bp.set_attribute('image_size_y', str(IM_HEIGHT))
    camera_bp.set_attribute('fov', str(FOV))
    camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=ego_vehicle)
    image_queue = Queue()
    camera.listen(image_queue.put)
    print("カメラセンサをスポーン")
    
    # === semantic lidarセンサの設定 ===
    lidar_bp = blueprint_library.find('sensor.lidar.ray_cast_semantic')
    # 水平視野角を360度にする
    lidar_bp.set_attribute('horizontal_fov', '180.0')
    # 回転周波数をFPSと同じにする (1ステップで360度データを得るため)
    lidar_bp.set_attribute('rotation_frequency', '20.0')  # 20Hz
    # センサーティックを0.0に設定し、可能な限り高速でデータを取得する (各シミュレーションステップ)
    lidar_bp.set_attribute('sensor_tick', '0.0')
    # レーザーの数（チャネル数、デフォルト32)
    lidar_bp.set_attribute('channels', '64')
    # 測定範囲（メートル単位、デフォルト10.0） [8]
    lidar_bp.set_attribute('range', '100.0')
    # 1秒あたりの点群数 (デフォルト56000)
    lidar_bp.set_attribute('points_per_second', '56000')
    lidar_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=ego_vehicle)
    lidar_queue = Queue()
    lidar.listen(lidar_queue.put)
    print("Semantic Lidarセンサをスポーン")
    
    # カメラ行列の計算
    K = camera_util.build_projection_matrix(IM_WIDTH, IM_HEIGHT, FOV)
    K_b = camera_util.build_projection_matrix(IM_WIDTH, IM_HEIGHT, FOV, is_behind_camera=True)
    
    # === シミュレーション開始 ===
    ego_vehicle.set_autopilot(True)
    try:
        duration_sec = TIME_DURATION
        num_frames = int(duration_sec / FIXED_DELTA_SECONDS)
        for frame_idx in range(num_frames):
            world.tick() # シミュレーションを進める

            image = image_queue.get()
            rgb_image = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
            # rgb_image = rgb_image[:, :, :3]
            lidar_data = lidar_queue.get()
            lidar_array = np.frombuffer(lidar_data.raw_data, dtype=np.dtype([
                ('x', np.float32), ('y', np.float32), ('z', np.float32),
                ('cos_angle', np.float32), ('ObjIdx', np.uint32), ('ObjTag', np.uint32)
            ]))
            target_tag = {4, 10, 12, 18} # 歩行者、車両、標識、信号機
            filtered_points = dict()
            for point in lidar_array:
                if point['ObjTag'] in target_tag and point['x'] > 0:
                    ObjIdx, ObjTag = int(point['ObjIdx']), int(point['ObjTag'])
                    x, y, z = float(point['x']), float(point['y']), float(point['z'])
                    key = (ObjIdx, ObjTag)

                    if key not in filtered_points:
                        filtered_points[key] = [(x, y, z)]
                    else:
                        filtered_points[key].append((x, y, z))

            world_2_camera = get_world_to_camera_matrix(camera)
            camera_transform = camera.get_transform()
            camera_location = camera_transform.location
            camera_forward_vec = camera_transform.get_forward_vector()
            actors = world.get_actors()
            frame_labels = list()
            for ObjIdx, ObjTag in filtered_points.keys():
                if ObjIdx == ego_id:
                    continue
                print(ObjIdx, ObjTag)
                actor = actors.find(ObjIdx)
                if actor:
                    # print(actor.id, actor.semantic_tags ,actor.bounding_box)
                    bbox = actor.bounding_box
                    verts = [v for v in bbox.get_world_vertices(actor.get_transform())]
                    points_2d_on_image = []

                    for vert in verts:
                        ray_vert = vert - camera_location
                        if camera_forward_vec.dot(ray_vert) > 0: 
                            p = camera_util.get_image_point(vert, K, world_2_camera)
                            points_2d_on_image.append(p)
                    
                    yolo_bbox = camera_util.calculate_yolo_bbox(points_2d_on_image, IM_WIDTH, IM_HEIGHT)
                    if yolo_bbox:
                        xmin, xmax, ymin, ymax = yolo_bbox
                        size = (xmax - xmin) * (ymax - ymin)
                        if size > SIZE_THRESHOLD:
                            # print(size)
                            class_id = ObjTag
                            frame_labels.append([class_id, xmin, xmax, ymin, ymax, size])
        
            for bbox in frame_labels:
                class_id, xmin, xmax, ymin, ymax, size = bbox
                xmin_draw = int(xmin)
                xmax_draw = int(xmax)
                ymin_draw = int(ymin)
                ymax_draw = int(ymax)
                print(f"Class ID: {class_id}, BBox: ({xmin_draw}, {ymin_draw}), ({xmax_draw}, {ymax_draw}), Size: {size}")
                color = (0, 255, 0)
                cv2.rectangle(rgb_image, (xmin_draw, ymin_draw), (xmax_draw, ymax_draw), color, 2)
                cv2.putText(rgb_image, f"{int(class_id)}", (xmin_draw, ymin_draw - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            cv2.imshow("Semantic LiDAR BBoxes", rgb_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("ユーザーによりシミュレーションが停止されました。")
                break
            
    finally:
        print("シミュレーションが終了しました。")
        if camera:
            camera.stop()
            camera.destroy()
        print("カメラをクリーンアップしました。")
        for vehicle in vehicles:
            if vehicle:
                vehicle.destroy()
        print(f"{len(vehicles)} 台のNPC車両を破棄")
        for pedestrian in pedestrians:
            if pedestrian:
                pedestrian.destroy()
        print(f"{len(pedestrians)} 人のNPC歩行者を破棄")
        for controller in walker_controllers:
            if controller:
                controller.stop()
                controller.destroy()
        print("歩行者コントローラを破棄")
        if ego_vehicle:
            ego_vehicle.destroy()
            print("Ego車両を破棄")
        if lidar:
            lidar.stop()
            lidar.destroy()
        print("センサをクリーンアップしました。")
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        print("ワールドの設定を非同期モードに戻しました。")
        traffic_manager.set_synchronous_mode(False)
        print("トラフィックマネージャーを非同期モードに戻しました。")
        cv2.destroyAllWindows()
        print("ウィンドウを閉じました。")

if __name__ == '__main__':
    main()