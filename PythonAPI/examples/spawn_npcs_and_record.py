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
X_MAX = -10000
X_MIN = 10000
Y_MAX = -10000
Y_MIN = 10000

# === クラスIDマッピングの定義 ===
CLASS_MAPPING = {
    carla.CityObjectLabel.TrafficLight: 0,
    carla.CityObjectLabel.TrafficSigns: 1,
    carla.CityObjectLabel.Vehicles: 2,
    carla.CityObjectLabel.Pedestrians: 3,
}

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

def calculate_yolo_bbox(points_2d, img_width, img_height):
    """
    2D頂点からYOLO形式のバウンディングボックス (center_x, center_y, width, height) を計算する。
    Args:
        points_2d (list): [(x1, y1), (x2, y2), ...] の形式の2D頂点リスト。
        img_width (int): 画像の幅。
        img_height (int): 画像の高さ。
    Returns:
        tuple: (class_id, center_x, center_y, bbox_width, bbox_height) または None（ボックスが画像外の場合）。
    """
    if not points_2d:
        return None

    # 画像の範囲でクリッピング
    # numpy配列に変換してmin/maxを計算
    points_2d = np.array(points_2d)
    
    # x, y 座標を画像の範囲内にクリップ
    xmin = np.clip(np.min(points_2d[:, 0]), 0, img_width - 1)
    ymin = np.clip(np.min(points_2d[:, 1]), 0, img_height - 1)
    xmax = np.clip(np.max(points_2d[:, 0]), 0, img_width - 1)
    ymax = np.clip(np.max(points_2d[:, 1]), 0, img_height - 1)

    # クリッピング後も有効なボックスか確認
    if xmax <= xmin or ymax <= ymin:
        return None # 無効なボックスはスキップ

    bbox_width_pixels = xmax - xmin
    bbox_height_pixels = ymax - ymin

    center_x_pixels = (xmin + xmax) / 2.0
    center_y_pixels = (ymin + ymax) / 2.0

    # 正規化
    center_x = center_x_pixels / img_width
    center_y = center_y_pixels / img_height
    bbox_width = bbox_width_pixels / img_width
    bbox_height = bbox_height_pixels / img_height

    return (center_x, center_y, bbox_width, bbox_height)
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
        # image.save_to_disk(image_path)

        # RGB配列に整形
        img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))

        # ワールドからカメラへの変換行列を取得
        world_to_camera = np.array(camera.get_transform().get_inverse_matrix())
        image_w = camera_bp.get_attribute("image_size_x").as_int()
        image_h = camera_bp.get_attribute("image_size_y").as_int()
        fov = camera_bp.get_attribute("fov").as_float()

        # Calculate the camera projection matrix to project from 3D -> 2D
        K = build_projection_matrix(image_w, image_h, fov)
        K_b = build_projection_matrix(image_w, image_h, fov, is_behind_camera=True)

        # 現在のフレームのラベルデータを格納するリスト
        frame_labels = []

        # 検出対象のオブジェクトリストを生成
        # TrafficLight, TrafficSigns, Vehicles, Pedestrians を対象にする
        objects_to_detect = []
        objects_to_detect.extend(world.get_actors().filter('traffic.traffic_light'))
        objects_to_detect.extend(world.get_actors().filter('traffic.traffic_sign'))
        # objects_to_detect.extend(world.get_actors().filter('vehicle.*'))
        # objects_to_detect.extend(world.get_actors().filter('walker.*'))
        
        camera_transform = camera.get_transform()
        camera_location = camera_transform.location
        camera_forward_vec = camera_transform.get_forward_vector()
        
        # bounding boxの描画
        for obj in objects_to_detect:
            if obj.id == ego_vehicle.id: # Ego車両は検出対象から除外
                continue

            # オブジェクトのBoundingBoxを取得
            bb = obj.bounding_box
            
            # オブジェクトのタイプを判断し、クラスIDを取得
            obj_class_id = None
            if 'traffic_light' in obj.type_id:
                obj_class_id = CLASS_MAPPING.get(carla.CityObjectLabel.TrafficLight)
            elif 'traffic_sign' in obj.type_id:
                obj_class_id = CLASS_MAPPING.get(carla.CityObjectLabel.TrafficSigns)
            elif 'vehicle' in obj.type_id:
                obj_class_id = CLASS_MAPPING.get(carla.CityObjectLabel.Vehicles)
            elif 'walker' in obj.type_id:
                obj_class_id = CLASS_MAPPING.get(carla.CityObjectLabel.Pedestrians)
            
            if obj_class_id is None: # マッピングされていないオブジェクトはスキップ
                continue

            dist = obj.get_transform().location.distance(camera_location)

            if dist < VALID_DISTANCE:
                ray = obj.get_transform().location - camera_location
                
                # カメラの視野角内（前方）にあるかを確認
                if camera_forward_vec.dot(ray) > 0:
                    verts = [v for v in bb.get_world_vertices(obj.get_transform())]
                    points_2d_on_image = []

                    # 2D投影された頂点を収集し、画像内に存在するかを確認
                    all_points_in_canvas = True # 全ての点が画像内にあるかフラグ
                    for vert in verts:
                        # 頂点がカメラの後ろにある場合は、別のK_bを使うべき
                        # このロジックは3D BBOX描画で必要だが、2D BBOX抽出では複雑になりがち
                        # まずは通常のKで試し、問題があれば調整
                        ray_vert = vert - camera_location
                        if camera_forward_vec.dot(ray_vert) > 0: # 頂点がカメラの前にいる場合
                            p = get_image_point(vert, K, world_to_camera)
                        else: # 頂点がカメラの後ろにいる場合
                            # カメラの後ろの頂点を考慮すると、2D BBOX計算が非常に複雑になる
                            # 簡単な方法としては、そのオブジェクトを検出対象から除外するか、
                            # 少なくとも描画はスキップする
                            # 今回は単純化のため、全頂点がカメラの前にないとスキップする、というロジックは採用しない
                            # ただし、描画時にK_bを使うことで、クリッピングはできる
                            p = get_image_point(vert, K_b, world_to_camera) # 仮に後ろの頂点も投影
                        
                        points_2d_on_image.append(p)
                        # 一つでも点が完全に画像外の場合、描画は省略する
                        # if not point_in_canvas(p, IM_WIDTH, IM_HEIGHT):
                        #     all_points_in_canvas = False
                        #     break # 描画はしない

                    # if not all_points_in_canvas:
                    #     continue # 描画はスキップ

                    # 2Dバウンディングボックスの計算とYOLO形式への変換
                    yolo_bbox = calculate_yolo_bbox(points_2d_on_image, IM_WIDTH, IM_HEIGHT)
                    
                    if yolo_bbox:
                        center_x, center_y, bbox_width, bbox_height = yolo_bbox
                        frame_labels.append(f"{obj_class_id} {center_x:.6f} {center_y:.6f} {bbox_width:.6f} {bbox_height:.6f}")

                        # デバッグ用に画像にバウンディングボックスを描画 (確認用)
                        # OpenCVはBGR形式なのでimg_displayを使う
                        # 描画は非正規化座標で行う
                        xmin_draw = int((center_x - bbox_width / 2) * IM_WIDTH)
                        ymin_draw = int((center_y - bbox_height / 2) * IM_HEIGHT)
                        xmax_draw = int((center_x + bbox_width / 2) * IM_WIDTH)
                        ymax_draw = int((center_y + bbox_height / 2) * IM_HEIGHT)
                        cv2.rectangle(img, (xmin_draw, ymin_draw), (xmax_draw, ymax_draw), (0, 255, 0), 2) # 緑色で描画
                        cv2.putText(img, f"{obj_class_id}", (xmin_draw, ymin_draw - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        # ラベルファイルを保存
        label_path = os.path.join(OUTPUT_LABEL_DIR, f"{image.frame:06d}.txt")
        with open(label_path, 'w') as f:
            for label_line in frame_labels:
                f.write(label_line + '\n')

        for npc in world.get_actors().filter('*vehicle*' or '*pedestrian*'):
            # Filter out the ego vehicle
            if npc.id != ego_vehicle.id:

                bb = npc.bounding_box
                dist = npc.get_transform().location.distance(camera_location)

                if dist < VALID_DISTANCE:
                    camera_forward_vec = camera_transform.get_forward_vector()
                    ray = npc.get_transform().location - camera_location

                    if camera_forward_vec.dot(ray) > 0:
                        p1 = get_image_point(bb.location, K, world_to_camera)
                        verts = [v for v in bb.get_world_vertices(npc.get_transform())]
                        x_max = -10000
                        x_min = 10000
                        y_max = -10000
                        y_min = 10000

                        for vert in verts:
                            p = get_image_point(vert, K, world_to_camera)
                            # Find the rightmost vertex
                            if p[0] > x_max:
                                x_max = p[0]
                            # Find the leftmost vertex
                            if p[0] < x_min:
                                x_min = p[0]
                            # Find the highest vertex
                            if p[1] > y_max:
                                y_max = p[1]
                            # Find the lowest  vertex
                            if p[1] < y_min:
                                y_min = p[1]

                        cv2.line(img, (int(x_min),int(y_min)), (int(x_max),int(y_min)), (255,0,0, 255), 1)
                        cv2.line(img, (int(x_min),int(y_max)), (int(x_max),int(y_max)), (255,0,0, 255), 1)
                        cv2.line(img, (int(x_min),int(y_min)), (int(x_min),int(y_max)), (255,0,0, 255), 1)
                        cv2.line(img, (int(x_max),int(y_min)), (int(x_max),int(y_max)), (255,0,0, 255), 1)


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