import glob
import os
import sys
import carla
import random
import time
import queue
import numpy as np
import cv2
import csv

# === Carla Egg のパス設定 ===
try:
    sys.path.append(glob.glob('C:/CARLA_Latest/WindowsNoEditor/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64'))[0])
except IndexError:
    pass

# === グローバル変数の設定 ===
CARLA_HOST = 'localhost'
CARLA_PORT = 2000

MAP = "Town01_Opt"
TIME_DURATION = 1000
VALID_DISTANCE = 50
FIXED_DELTA_SECONDS = 0.05

CAR_RATIO = 0.5

IM_WIDTH = 800
IM_HEIGHT = 600
FOV = 60

OUTPUT_IMG_DIR = "C:\CARLA_Latest\WindowsNoEditor\output\image"
OUTPUT_LABEL_DIR = "C:\CARLA_Latest\WindowsNoEditor\output\label"

EDGES = [[0,1], [1,3], [3,2], [2,0], [0,4], [4,5], [5,1], [5,7], [7,6], [6,4], [6,2], [7,3]]

# === ヘルパー関数 ===
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
        # Calculate 2D projection of 3D coordinate

        # Format the input coordinate (loc is a carla.Position object)
        point = np.array([loc.x, loc.y, loc.z, 1])
        # transform to camera coordinates
        point_camera = np.dot(w2c, point)

        # New we must change from UE4's coordinate system to an "standard"
        # (x, y ,z) -> (y, -z, x)
        # and we remove the fourth componebonent also
        point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

        # now project 3D->2D using the camera matrix
        point_img = np.dot(K, point_camera)
        # normalize
        point_img[0] /= point_img[2]
        point_img[1] /= point_img[2]

        return point_img[0:2]

def point_in_canvas(pos, img_h, img_w):
    """Return true if point is in canvas"""
    if (pos[0] >= 0) and (pos[0] < img_w) and (pos[1] >= 0) and (pos[1] < img_h):
        return True
    return False

# === Carlaサーバに接続 ===
client = carla.Client(CARLA_HOST, CARLA_PORT)
client.set_timeout(10.0)

# マップ変更
print(client.get_available_maps())
client.load_world(MAP)
world = client.get_world()
blueprint_library = world.get_blueprint_library()
print(f"Loaded world '{MAP}' with specific layers. Parked vehicles should be gone.")

blueprint_library = world.get_blueprint_library()

# === 同期モード設定 ===
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = FIXED_DELTA_SECONDS  # シミュレーション1ステップ = 0.05秒
world.apply_settings(settings)

# === トラフィックマネージャも同期モードに ===
traffic_manager = client.get_trafficmanager()
traffic_manager.set_synchronous_mode(True)
tm_port = traffic_manager.get_port()

# === NPC車両スポーン ===
spawn_points = world.get_map().get_spawn_points()
vehicles = []
num_npc = int(len(spawn_points) * CAR_RATIO)
for i in range(num_npc):
    npc_bp = random.choice(blueprint_library.filter('vehicle.*'))
    transform = spawn_points[(i+1) % len(spawn_points)]
    npc = world.try_spawn_actor(npc_bp, transform)
    if npc:
        npc.set_autopilot(True)
        vehicles.append(npc)

print(f"{num_npc}npc vehicle spawned")
# === 歩行者スポーン ===
pedestrians = []
walker_controllers = []
n_walker = 10
for _ in range(n_walker):
    walker_bp = random.choice(blueprint_library.filter('walker.pedestrian.*'))
    loc = world.get_random_location_from_navigation()
    if loc:
        walker = world.try_spawn_actor(walker_bp, carla.Transform(loc))
        if walker:
            ctrl_bp = blueprint_library.find('controller.ai.walker')
            ctrl = world.spawn_actor(ctrl_bp, carla.Transform(), attach_to=walker)
            ctrl.start()
            ctrl.go_to_location(world.get_random_location_from_navigation())
            ctrl.set_max_speed(1.0 + random.random())  # 1–2 m/s
            pedestrians.append(walker)
            walker_controllers.append(ctrl)
print("npc walker spawned")

# === Ego車両スポーン（最後） ===
ego_bp = blueprint_library.find('vehicle.lincoln.mkz_2020')
ego_transform = spawn_points[0]
ego_vehicle = world.try_spawn_actor(ego_bp, ego_transform)
print("Ego車両スポーン")

# === カメラセンサの設定 ===
camera_bp = blueprint_library.find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', str(IM_WIDTH))
camera_bp.set_attribute('image_size_y', str(IM_HEIGHT))
camera_bp.set_attribute('fov', str(FOV))
# 例1: 車両の少し前、中央、ルーフの高さあたりから前方を向くカメラ
camera_transform = carla.Transform(
    carla.Location(x=1.5, y=0.0, z=1.4),  # 前方1.5m, 横方向中央, 上方1.4m
    carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0) # 回転なし (車両と同じ向き)
)

# 例2: ダッシュボードカメラ風 (少し前、中央、低め)
# camera_transform = carla.Transform(
#     carla.Location(x=0.8, y=0.0, z=1.3),
#     carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)
# )

# 例3: 後方確認用カメラ (車両後方、少し上、後ろを向く)
# camera_transform = carla.Transform(
#     carla.Location(x=-2.0, y=0.0, z=1.0),  # 後方2.0m
#     carla.Rotation(pitch=0.0, yaw=180.0, roll=0.0) # 180度回転して後方を向く
# )

# 例4: ドライブレコーダー風に少し下を向ける
# camera_transform = carla.Transform(
#     carla.Location(x=1.2, y=0.0, z=1.35),
#     carla.Rotation(pitch=-10.0, yaw=0.0, roll=0.0) # 10度下を向ける
# )
# カメラをego_vehicleにアタッチ
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=ego_vehicle)
image_queue = queue.Queue()
print("Camera attached")

# 保存用ディレクトリ作成
os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)

# カメラ行列の計算
K = build_projection_matrix(IM_WIDTH, IM_HEIGHT, FOV)

# === シミュレーション開始 ===
ego_vehicle.set_autopilot(True)

try:
    print("Running simulation... Press 'q' to stop.")
    duration_sec = TIME_DURATION
    num_frames = int(duration_sec / FIXED_DELTA_SECONDS)

    # カメラのリスナーを設定
    camera.listen(image_queue.put)

    for frame_idx in range(num_frames):
        world.tick() # シミュレーションを進める
        
        # カメラからの画像をキューから取得
        image = image_queue.get()

        # 生の画像データを保存
        image_path = os.path.join(OUTPUT_IMG_DIR, f"{image.frame:06d}.png")
        image.save_to_disk(image_path)

        # RGB配列に整形
        img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))

        # ワールドからカメラへの変換行列を取得
        world_to_camera = np.array(camera.get_transform().get_inverse_matrix())
        image_w = IM_HEIGHT
        image_h = IM_HEIGHT
        fov = FOV

        # Calculate the camera projection matrix to project from 3D -> 2D
        K = build_projection_matrix(image_w, image_h, fov)
        K_b = build_projection_matrix(image_w, image_h, fov, is_behind_camera=True)

        # carla.CityObjectLabel.Vehicles, carla.CityObjectLabel.Pedestrians などでフィルタリング可能
        bounding_boxes = world.get_level_bbs(carla.CityObjectLabel.TrafficLight)
        bounding_boxes.extend(world.get_level_bbs(carla.CityObjectLabel.TrafficSigns))
        # bounding_boxes.extend(world.get_level_bbs(carla.CityObjectLabel.Vehicles))
        

        # #ego_vehicle前方50m以内のbboxにフィルタリング
        # ego_transform = ego_vehicle.get_transform()
        # ego_location = ego_transform.location
        # ego_forward_vec = ego_transform.get_forward_vector()
        # print(f"ego_location:{ego_location}")

        # bounding_boxes = [
        #     bbox for bbox in bounding_boxes
        #     if 1 < bbox.location.distance(ego_location) < VALID_DISTANCE and \
        #     ego_forward_vec.dot(bbox.location - ego_location) > 0
        # ]
        # for bbox in bounding_boxes:
        #     if hasattr(bbox, 'actor_id'):
        #         print(f"bbox has actor_id attribute] {bbox.actor_id}")
        #     print(bbox.location)
        # break
        # bounding boxの描画
        for bbox in bounding_boxes:
            if bbox.location == ego_vehicle.bounding_box.location:
                continue
            # Ego車両からの距離でフィルタリング（例: 50m以内）
            if 1 < bbox.location.distance(ego_vehicle.get_transform().location) < VALID_DISTANCE:
                forward_vec = ego_vehicle.get_transform().get_forward_vector()
                ray = bbox.location - ego_vehicle.get_transform().location

                if forward_vec.dot(ray) > 0:
                    # Cycle through the vertices
                    verts = [v for v in bbox.get_world_vertices(carla.Transform())]
                    for edge in EDGES:
                        # Join the vertices into edges
                        p1 = get_image_point(verts[edge[0]], K, world_to_camera)
                        p2 = get_image_point(verts[edge[1]],  K, world_to_camera)
                        # Draw the edges into the camera output
                        cv2.line(img, (int(p1[0]),int(p1[1])), (int(p2[0]),int(p2[1])), (0,0,255, 255), 1)

        for npc in world.get_actors().filter('*vehicle*' or '*pedestrian*'):
            # Filter out the ego vehicle
            if npc.id != ego_vehicle.id:

                bb = npc.bounding_box
                dist = npc.get_transform().location.distance(ego_vehicle.get_transform().location)

                if dist < VALID_DISTANCE:
                    forward_vec = ego_vehicle.get_transform().get_forward_vector()
                    ray = npc.get_transform().location - ego_vehicle.get_transform().location

                    if forward_vec.dot(ray) > 0:
                        verts = [v for v in bb.get_world_vertices(npc.get_transform())]
                        for edge in EDGES:
                            p1 = get_image_point(verts[edge[0]], K, world_to_camera)
                            p2 = get_image_point(verts[edge[1]],  K, world_to_camera)

                            p1_in_canvas = point_in_canvas(p1, image_h, image_w)
                            p2_in_canvas = point_in_canvas(p2, image_h, image_w)

                            if not p1_in_canvas and not p2_in_canvas:
                                continue

                            ray0 = verts[edge[0]] - camera.get_transform().location
                            ray1 = verts[edge[1]] - camera.get_transform().location
                            cam_forward_vec = camera.get_transform().get_forward_vector()

                            # One of the vertex is behind the camera
                            if not (cam_forward_vec.dot(ray0) > 0):
                                p1 = get_image_point(verts[edge[0]], K_b, world_to_camera)
                            if not (cam_forward_vec.dot(ray1) > 0):
                                p2 = get_image_point(verts[edge[1]], K_b, world_to_camera)

                            cv2.line(img, (int(p1[0]),int(p1[1])), (int(p2[0]),int(p2[1])), (255,0,0, 255), 1)        

        # 画像を表示
        cv2.namedWindow('Carla Camera with Bounding Boxes', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Carla Camera with Bounding Boxes', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    print("Cleaning up...")

    # アクターの破棄
    if camera:
        camera.stop()
        camera.destroy()
    if ego_vehicle:
        ego_vehicle.destroy()
    for v in vehicles:
        if v:
            v.destroy()
    for c in walker_controllers:
        if c:
            c.stop()
            c.destroy()
    for p in pedestrians:
        if p:
            p.destroy()

    # シミュレーション設定を元に戻す
    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)

    traffic_manager.set_synchronous_mode(False)
    cv2.destroyAllWindows() # OpenCVウィンドウを閉じる

    print("Done.")