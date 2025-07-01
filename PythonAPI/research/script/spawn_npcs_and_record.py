import glob
import os
import sys
import cv2
import csv
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
    cameras, image_queues = camera_util.setting_camera(world, blueprint_library, ego_vehicle, IM_WIDTH, IM_HEIGHT, FOV, NUM_CAMERA)

    # 保存用ディレクトリ作成
    os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
    image_dir = OUTPUT_IMG_DIR + f"/{MAP}"
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)
    label_dir = OUTPUT_LABEL_DIR + f"/{MAP}"
    os.makedirs(label_dir, exist_ok=True)

    # カメラ行列の計算
    K = camera_util.build_projection_matrix(IM_WIDTH, IM_HEIGHT, FOV)
    K_b = camera_util.build_projection_matrix(IM_WIDTH, IM_HEIGHT, FOV, is_behind_camera=True)

    row_image_ques, bbox_image_ques, label_ques = camera_util.create_save_queues(NUM_CAMERA)

    # === シミュレーション開始 ===
    ego_vehicle.set_autopilot(True)
    try:
        print(SIZE_THRESHOLD)
        print("シミュレーションを実行中... 'q' キーを押すと停止します。")
        duration_sec = TIME_DURATION
        num_frames = int(duration_sec / FIXED_DELTA_SECONDS)

        for frame_idx in range(num_frames):
            world.tick() # シミュレーションを進める
            
            # 各カメラの画像をキューから取得し、処理
            for i, cam_q in enumerate(image_queues):
                image = cam_q.get()
                # RGB配列に整形
                image = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
                camera_actor = cameras[i]
                display_name = f'Carla Camera {i+1} with Bounding Boxes'
                
                frame_labels, bbox_image = camera_util.process_camera_data(image, camera_actor, world, K, K_b, display_name, size_threshold=SIZE_THRESHOLD)
                img_save_que = row_image_ques[i]
                img_save_que.put(image)
                bbox_save_que = bbox_image_ques[i]
                bbox_save_que.put(bbox_image)
                label_save_que = label_ques[i]
                label_save_que.put(frame_labels)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("ユーザーによりシミュレーションが停止されました。")
                break

    finally:
        print("シミュレーションが終了しました。")
        #　画像を保存
        # print("生の画像を保存中")
        # for i in range(len(cameras)):
        #     img_save_que = row_image_ques[i]
        #     camera = cameras[i]
        #     image_dir = OUTPUT_IMG_DIR + f"/{MAP}" + f"/{camera.attributes['role_name']}"
        #     os.makedirs(image_dir, exist_ok=True)
        #     print(f"{camera.attributes['role_name']} の画像を保存しています...")
        #     num_frame = 0
        #     while not img_save_que.empty():
        #         image = img_save_que.get()
        #         image_path = os.path.join(image_dir, f"{num_frame:06d}.png")
        #         cv2.imwrite(image_path, image)
        #         num_frame += 1
        # print("生のすべての画像を保存しました。")
        # print("バウンディングボックスを描画した画像を保存中")
        # for i in range(len(cameras)):
        #     bbox_save_que = bbox_image_ques[i]
        #     camera = cameras[i]
        #     bbox_dir = OUTPUT_IMG_DIR + f"/{MAP}" + f"/{camera.attributes['role_name']}_bbox"
        #     os.makedirs(bbox_dir, exist_ok=True)
        #     print(f"{camera.attributes['role_name']} のバウンディングボックスを描画した画像を保存しています...")
        #     num_frame = 0
        #     while not bbox_save_que.empty():
        #         bbox_image = bbox_save_que.get()
        #         bbox_image_path = os.path.join(bbox_dir, f"{num_frame:06d}.png")
        #         cv2.imwrite(bbox_image_path, bbox_image)
        #         num_frame += 1
        # # ラベルを保存
        # print("ラベルを保存中") 
        # for i in range(len(cameras)):
        #     label_save_que = label_ques[i]
        #     camera = cameras[i]
        #     label_dir = OUTPUT_LABEL_DIR + f"/{MAP}" + f"/{camera.attributes['role_name']}"
        #     os.makedirs(label_dir, exist_ok=True)
        #     print(f"{camera.attributes['role_name']} のラベルを保存しています...")
        #     num_frame = 0
        #     while not label_save_que.empty():
        #         labels = label_save_que.get()
        #         label_path = os.path.join(label_dir, f"{num_frame:06d}.csv")
        #         with open(label_path, 'w') as f:
        #             writer = csv.writer(f)
        #             writer.writerow(['class_id', 'xmin', 'xmax', 'ymin', 'ymax', 'distance'])
        #             for label in labels:
        #                 writer.writerow(label)
        #         num_frame += 1
        # print("すべてのラベルを保存しました。")
        
        carla_util.cleanup(client, world, vehicles, pedestrians, walker_controllers, cameras)
        cv2.destroyAllWindows() # OpenCVウィンドウを閉じる

        print("シミュレーション終了")

if __name__ == '__main__':
    main()