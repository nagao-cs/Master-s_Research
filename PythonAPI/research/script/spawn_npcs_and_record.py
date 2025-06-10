import glob
import os
import sys
import carla
import random
import time
import queue
import numpy as np
import cv2
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from utils import carla_util
from utils.config import *

# === Carla Egg のパス設定 ===
try:
    sys.path.append(glob.glob('C:/CARLA_Latest/WindowsNoEditor/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64'))[0])
except IndexError:
    pass


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
    """
    2D頂点からYOLO形式のバウンディングボックス (center_x, center_y, width, height) を計算する。
    Args:
        points_2d (list): [(x1, y1), (x2, y2), ...] の形式の2D頂点リスト。
        img_width (int): 画像の幅。
        img_height (int): 画像の高さ。
    Returns:
        tuple: (center_x, center_y, bbox_width, bbox_height) または None（ボックスが画像外の場合）。
    """
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

def process_camera_data(image, camera_actor, world, K, K_b, ego_vehicle, output_img_dir, output_label_dir, display_window_name):
    """
    単一のカメラ画像を処理し、バウンディングボックスを計算して画像とラベルを保存する。
    """
    # RGB配列に整形
    img_raw = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
    img_display = img_raw.copy() # デバッグ表示用 (バウンディングボックスを描画)

    # ワールドからカメラへの変換行列を取得
    world_to_camera = np.array(camera_actor.get_transform().get_inverse_matrix())
    
    camera_transform = camera_actor.get_transform()
    camera_location = camera_transform.location
    camera_forward_vec = camera_transform.get_forward_vector()
    
    # 現在のフレームのラベルデータを格納するリスト
    frame_labels = []

    # CityObjectLabels (TrafficLight, TrafficSigns) の処理
    city_object_categories = [carla.CityObjectLabel.TrafficLight, carla.CityObjectLabel.TrafficSigns]
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
                        else: # 頂点がカメラの後ろにあれば反転行列で投影（完全なバウンディングボックスのため）
                            p = get_image_point(vert, K_b, world_to_camera)
                        
                        points_2d_on_image.append(p)
                    
                    yolo_bbox = calculate_yolo_bbox(points_2d_on_image, IM_WIDTH, IM_HEIGHT)
                    
                    if yolo_bbox:
                        center_x, center_y, bbox_width, bbox_height = yolo_bbox
                        class_id = CLASS_MAPPING[category_label]
                        frame_labels.append(f"{class_id} {center_x:.6f} {center_y:.6f} {bbox_width:.6f} {bbox_height:.6f}")

                        # デバッグ用に画像にバウンディングボックスを描画 (緑色)
                        xmin_draw = int((center_x - bbox_width / 2) * IM_WIDTH)
                        ymin_draw = int((center_y - bbox_height / 2) * IM_HEIGHT)
                        xmax_draw = int((center_x + bbox_width / 2) * IM_WIDTH)
                        ymax_draw = int((center_y + bbox_height / 2) * IM_HEIGHT)
                        cv2.rectangle(img_display, (xmin_draw, ymin_draw), (xmax_draw, ymax_draw), (0, 255, 0), 2)
                        cv2.putText(img_display, f"{class_id}", (xmin_draw, ymin_draw - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

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
                    center_x, center_y, bbox_width, bbox_height = yolo_bbox
                    
                    if 'vehicle' in actor.type_id:
                        class_id = CLASS_MAPPING[carla.CityObjectLabel.Vehicles]
                        color = (255, 0, 0) # 車両は青色
                    elif 'walker' in actor.type_id:
                        class_id = CLASS_MAPPING[carla.CityObjectLabel.Pedestrians]
                        color = (0, 0, 255) # 歩行者は赤色
                    else:
                        continue # 現在のフィルターでは発生しないはず

                    frame_labels.append(f"{class_id} {center_x:.6f} {center_y:.6f} {bbox_width:.6f} {bbox_height:.6f}")

                    # デバッグ用に画像にバウンディングボックスを描画
                    xmin_draw = int((center_x - bbox_width / 2) * IM_WIDTH)
                    ymin_draw = int((center_y - bbox_height / 2) * IM_HEIGHT)
                    xmax_draw = int((center_x + bbox_width / 2) * IM_WIDTH)
                    ymax_draw = int((center_y + bbox_height / 2) * IM_HEIGHT)
                    cv2.rectangle(img_display, (xmin_draw, ymin_draw), (xmax_draw, ymax_draw), color, 2)
                    cv2.putText(img_display, f"{class_id}", (xmin_draw, ymin_draw - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # 画像とラベルを保存
    # カメラのロール名（例: 'camera_1'）からIDを抽出
    camera_id_str = camera_actor.attributes['role_name'].split('_')[-1] 
    label_path = os.path.join(output_label_dir, f"{image.frame:06d}_{camera_id_str}.txt")
    
    with open(label_path, 'w') as f:
        for label_line in frame_labels:
            f.write(label_line + '\n')

    # 画像を表示
    cv2.imshow(display_window_name, img_display)


# === Carlaサーバに接続 ===
client = carla_util.connect_to_server(CARLA_HOST, CARLA_PORT, TIMEOUT)

# マップ変更
world, blueprint_library = carla_util.load_map(client, MAP)

# === 同期モード設定 ===
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
world.apply_settings(settings)

# === トラフィックマネージャも同期モードに ===
traffic_manager = client.get_trafficmanager()
traffic_manager.set_synchronous_mode(True)
tm_port = traffic_manager.get_port()

# === NPC車両スポーン ===
vehicles = carla_util.spawn_npc_vehicles(world, blueprint_library, traffic_manager, CAR_RATIO)

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
print("NPC歩行者をスポーンしました。")
exit()
# === Ego車両スポーン（最後） ===
ego_bp = blueprint_library.find('vehicle.lincoln.mkz_2020')
spawn_point = world.get_map().get_spawn_points()[0]
ego_transform = spawn_point
ego_vehicle = world.try_spawn_actor(ego_bp, ego_transform)
if ego_vehicle:
    print("Ego車両をスポーンしました。")
else:
    print("Ego車両のスポーンに失敗しました。")
    # エラー処理としてクリーンアップして終了
    client.apply_batch([carla.command.DestroyActor(x) for x in vehicles + pedestrians + walker_controllers])
    world.apply_settings(world.get_settings()) # Synchronous mode解除
    sys.exit(1)


# === カメラセンサの設定 ===
camera_bp = blueprint_library.find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', str(IM_WIDTH))
camera_bp.set_attribute('image_size_y', str(IM_HEIGHT))
camera_bp.set_attribute('fov', str(FOV))

# カメラのリストとそれぞれのキュー
cameras = []
image_queues = []

# カメラ1: 車両の少し前、中央、ルーフの高さあたりから前方を向くカメラ
# ロール名をブループリントに設定
camera_bp_1 = blueprint_library.find('sensor.camera.rgb')
camera_bp_1.set_attribute('image_size_x', str(IM_WIDTH))
camera_bp_1.set_attribute('image_size_y', str(IM_HEIGHT))
camera_bp_1.set_attribute('fov', str(FOV))
camera_bp_1.set_attribute('role_name', 'camera_1') # ここでブループリントにロール名を設定

camera_transform_1 = carla.Transform(carla.Location(x=1.5, y=0.0, z=1.4))
camera_1 = world.spawn_actor(camera_bp_1, camera_transform_1, attach_to=ego_vehicle, attachment_type=carla.AttachmentType.Rigid)
if camera_1:
    cameras.append(camera_1)
    q1 = queue.Queue()
    image_queues.append(q1)
    camera_1.listen(q1.put)
    print("Camera 1 をアタッチしました。")

# カメラ2: 車両の少し前、右寄りの位置から前方を向くカメラ
# ロール名をブループリントに設定
camera_bp_2 = blueprint_library.find('sensor.camera.rgb')
camera_bp_2.set_attribute('image_size_x', str(IM_WIDTH))
camera_bp_2.set_attribute('image_size_y', str(IM_HEIGHT))
camera_bp_2.set_attribute('fov', str(FOV))
camera_bp_2.set_attribute('role_name', 'camera_2') # ここでブループリントにロール名を設定

camera_transform_2 = carla.Transform(carla.Location(x=1.5, y=0.5, z=1.4))
camera_2 = world.spawn_actor(camera_bp_2, camera_transform_2, attach_to=ego_vehicle, attachment_type=carla.AttachmentType.Rigid)
if camera_2:
    cameras.append(camera_2)
    q2 = queue.Queue()
    image_queues.append(q2)
    camera_2.listen(q2.put)
    print("Camera 2 をアタッチしました。")


# 保存用ディレクトリ作成
os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)

# カメラ行列の計算 (すべてのカメラで同じ設定なので一度でOK)
K = build_projection_matrix(IM_WIDTH, IM_HEIGHT, FOV)
K_b = build_projection_matrix(IM_WIDTH, IM_HEIGHT, FOV, is_behind_camera=True)

# 後でまとめて保存するためのキュー（必要であれば）
que_save_image_later_1 = queue.Queue()
que_save_image_later_2 = queue.Queue()
later_save_que = [que_save_image_later_1, que_save_image_later_2]

# === シミュレーション開始 ===
ego_vehicle.set_autopilot(True, tm_port) # TrafficManagerのポートを指定

try:
    print("シミュレーションを実行中... 'q' キーを押すと停止します。")
    duration_sec = TIME_DURATION
    num_frames = int(duration_sec / FIXED_DELTA_SECONDS)

    for frame_idx in range(num_frames):
        world.tick() # シミュレーションを進める
        
        # 各カメラの画像をキューから取得し、処理
        for i, cam_q in enumerate(image_queues):
            image = cam_q.get()
            camera_actor = cameras[i]
            display_name = f'Carla Camera {i+1} with Bounding Boxes'
            
            process_camera_data(image, camera_actor, world, K, K_b, ego_vehicle, OUTPUT_IMG_DIR, OUTPUT_LABEL_DIR, display_name)
            save_que = later_save_que[i]
            save_que.put(image)       
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("ユーザーによりシミュレーションが停止されました。")
            break

finally:
    print("シミュレーションが終了しました。")
    #　画像を保存
    for i in range(len(cameras)):
        save_que = later_save_que[i]
        camera = cameras[i]
        image_dir = OUTPUT_IMG_DIR + f"/{camera.attributes['role_name']}"
        os.makedirs(image_dir, exist_ok=True)
        print(f"Camera {camera.attributes['role_name']} の画像を保存しています...")
        while not save_que.empty():
            image = save_que.get()
            image_path = os.path.join(image_dir, f"{image.frame:06d}.png")
            image.save_to_disk(image_path)
    print("すべての画像を保存しました。")
    print("\nクリーンアップ中...")

    # アクターの破棄
    # すべてのカメラのリスナーを停止し、破棄
    for cam in cameras:
        if cam:
            cam.stop()
            cam.destroy()
            print(f"Camera {cam.attributes['role_name']} を破棄しました。")

    if ego_vehicle:
        ego_vehicle.destroy()
        print("Ego車両を破棄しました。")

    for v in vehicles:
        if v:
            v.destroy()
    print(f"{len(vehicles)} NPC車両を破棄しました。")

    for c in walker_controllers:
        if c:
            c.stop()
            c.destroy()
    for p in pedestrians:
        if p:
            p.destroy()
    print(f"{len(pedestrians)} NPC歩行者を破棄しました。")

    # シミュレーション設定を元に戻す
    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)
    print("シミュレーション設定を非同期モードに戻しました。")

    traffic_manager.set_synchronous_mode(False)
    cv2.destroyAllWindows() # OpenCVウィンドウを閉じる

    print("完了しました。")