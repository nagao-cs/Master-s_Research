import glob
import os
import sys
import carla
import random
import time
import queue
import numpy as np
import cv2

# === Carla Egg のパス設定 ===
try:
    sys.path.append(glob.glob('C:/CARLA_Latest/WindowsNoEditor/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64'))[0])
except IndexError:
    pass

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

# === Carlaサーバに接続 ===
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

# マップ変更
print(client.get_available_maps())
map = 'Town10HD_Opt'
client.load_world(map)
world = client.get_world()
blueprint_library = world.get_blueprint_library()

# === 同期モード設定 ===
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05  # シミュレーション1ステップ = 0.05秒
world.apply_settings(settings)

# === トラフィックマネージャも同期モードに ===
traffic_manager = client.get_trafficmanager()
traffic_manager.set_synchronous_mode(True)
tm_port = traffic_manager.get_port()
print(f"tm_port:{tm_port}")


# === NPC車両スポーン ===
spawn_points = world.get_map().get_spawn_points()
print(len(spawn_points))
# random.shuffle(spawn_points)

vehicles = []
n_npc = (len(spawn_points)*2)//3
for i in range(n_npc):
    npc_bp = random.choice(blueprint_library.filter('vehicle.*'))
    transform = spawn_points[(i+1) % len(spawn_points)]
    npc = world.try_spawn_actor(npc_bp, transform)
    if npc:
        npc.set_autopilot(True)
        vehicles.append(npc)

print("npc vehicle spawned")
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
ego_bp = blueprint_library.find('vehicle.tesla.model3')
ego_transform = spawn_points[0]
ego_vehicle = world.try_spawn_actor(ego_bp, ego_transform)
print("hero appeared")

# === カメラセンサの設定 ===
IM_WIDTH = 800
IM_HEIGHT = 600
FOV = 60
camera_bp = blueprint_library.find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', str(IM_WIDTH))
camera_bp.set_attribute('image_size_y', str(IM_HEIGHT))
camera_bp.set_attribute('fov', str(FOV))
camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=ego_vehicle)

image_queue = queue.Queue()
print("Camera attached")

# 保存用ディレクトリ作成
os.makedirs("output/images", exist_ok=True)

# カメラ行列の計算
K = build_projection_matrix(IM_WIDTH, IM_HEIGHT, FOV)

def save_image(image):
    image.save_to_disk(f"output/images/{image.frame:06d}.png")
    print(f"[Saved] Frame {image.frame}")

# === シミュレーション開始 ===
ego_vehicle.set_autopilot(True)
print("Befor running Simulation")

TIME_DURATION = 1000
VALID_DISTANCE = 50
try:
    print("Running simulation... Press 'q' to stop.")

    duration_sec = TIME_DURATION

    # カメラのリスナーを設定
    camera.listen(image_queue.put)

    for _ in range(duration_sec):
        world.tick() # シミュレーションを進める
        
        # カメラからの画像をキューから取得
        image = image_queue.get()

        # RGB配列に整形
        img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))

        # ワールドからカメラへの変換行列を取得
        world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

        # すべてのオブジェクトのbounding boxを取得
        # carla.CityObjectLabel.Vehicles, carla.CityObjectLabel.Pedestrians などでフィルタリング可能
        bounding_boxes = world.get_level_bbs(carla.CityObjectLabel.Vehicles) 
        edges = [[0,1], [1,3], [3,2], [2,0], [0,4], [4,5], [5,1], [5,7], [7,6], [6,4], [6,2], [7,3]]

        # # bounding boxの描画
        # for bbox in bounding_boxes:
        #     # Ego車両からの距離でフィルタリング（例: 50m以内）
        #     if bbox.location.distance(ego_vehicle.get_transform().location) < 50:
        #         forward_vec = ego_vehicle.get_transform().get_forward_vector()
        #         ray = bbox.location - ego_vehicle.get_transform().location

        #         if forward_vec.dot(ray) > 0:
        #             # Cycle through the vertices
        #             verts = [v for v in bbox.get_world_vertices(carla.Transform())]
        #             for edge in edges:
        #                 # Join the vertices into edges
        #                 p1 = get_image_point(verts[edge[0]], K, world_2_camera)
        #                 p2 = get_image_point(verts[edge[1]],  K, world_2_camera)
        #                 # Draw the edges into the camera output
        #                 cv2.line(img, (int(p1[0]),int(p1[1])), (int(p2[0]),int(p2[1])), (0,0,255, 255), 1)

        for npc in world.get_actors().filter('*vehicle*'):
            # Filter out the ego vehicle
            if npc.id != ego_vehicle.id:

                bb = npc.bounding_box
                dist = npc.get_transform().location.distance(ego_vehicle.get_transform().location)

                if dist < VALID_DISTANCE:
                    forward_vec = ego_vehicle.get_transform().get_forward_vector()
                    ray = npc.get_transform().location - ego_vehicle.get_transform().location

                    if forward_vec.dot(ray) > 0:
                        p1 = get_image_point(bb.location, K, world_2_camera)
                        verts = [v for v in bb.get_world_vertices(npc.get_transform())]
                        x_max = -10000
                        x_min = 10000
                        y_max = -10000
                        y_min = 10000

                        for vert in verts:
                            p = get_image_point(vert, K, world_2_camera)
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

                        cv2.line(img, (int(x_min),int(y_min)), (int(x_max),int(y_min)), (0,0,255, 255), 1)
                        cv2.line(img, (int(x_min),int(y_max)), (int(x_max),int(y_max)), (0,0,255, 255), 1)
                        cv2.line(img, (int(x_min),int(y_min)), (int(x_min),int(y_max)), (0,0,255, 255), 1)
                        cv2.line(img, (int(x_max),int(y_min)), (int(x_max),int(y_max)), (0,0,255, 255), 1)

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