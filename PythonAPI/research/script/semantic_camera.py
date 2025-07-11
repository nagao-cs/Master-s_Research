import carla
import numpy as np
import cv2
import random
from queue import Queue

def semantic_callback(image, queue):
    queue.put(image)

def main():
    # CARLAクライアント接続
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    blueprint_library = world.get_blueprint_library()

    # Ego車両のスポーン
    vehicle_bp = blueprint_library.find('vehicle.lincoln.mkz_2020')
    spawn_point = random.choice(world.get_map().get_spawn_points())
    ego_vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    ego_vehicle.set_autopilot(True)

    # セマンティックセグメンテーションカメラの設定
    camera_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
    camera_bp.set_attribute('image_size_x', '800')
    camera_bp.set_attribute('image_size_y', '600')
    camera_bp.set_attribute('fov', '90')

    camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))  # 車両の上部前方
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=ego_vehicle)

    # キューで画像取得
    image_queue = Queue()
    camera.listen(lambda image: semantic_callback(image, image_queue))

    try:
        for _ in range(10000):  # 100フレーム取得して終了
            world.tick()

            if image_queue.empty():
                continue

            image = image_queue.get()

            image.convert(carla.ColorConverter.CityScapesPalette)
            img_array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
            sem_image = img_array[:, :, :3]

            cv2.imshow('Semantic Segmentation Camera', sem_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        print("クリーンアップ中...")
        camera.stop()
        camera.destroy()
        ego_vehicle.destroy()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
