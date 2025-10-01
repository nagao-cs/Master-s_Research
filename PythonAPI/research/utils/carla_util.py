import carla
import random
import os
import cv2
import csv


def connect_to_server(host, port, timeout):
    client = carla.Client(host, port)
    client.set_timeout(timeout)
    return client


def load_map(client, map_name):
    world = client.load_world(map_name)
    print(client.get_available_maps())
    bp = world.get_blueprint_library()
    print(f"{map_name} loaded")
    return world, bp


def apply_settings(world, synchronous_mode, fixed_delta_seconds):
    settings = world.get_settings()
    settings.synchronous_mode = synchronous_mode
    settings.fixed_delta_seconds = fixed_delta_seconds
    world.apply_settings(settings)
    print(
        f"World settings applied: Synchronous mode: {synchronous_mode}, Fixed delta seconds: {fixed_delta_seconds}")
    return world


def setting_traffic_manager(client, synchronous_mode):
    traffic_manager = client.get_trafficmanager()
    traffic_manager.set_synchronous_mode(synchronous_mode)
    tm_port = traffic_manager.get_port()
    print(
        f"Traffic manager settings applied: Synchronous mode: {synchronous_mode}, Port: {tm_port}")
    return traffic_manager, tm_port


def spawn_npc_vehicles(world, bp, traffic_manager, spawn_points, car_ratio):
    tm_port = traffic_manager.get_port()
    num_spawn_points = len(spawn_points)
    vehicles = list()
    num_vehicles = int(num_spawn_points * car_ratio)
    car_bps = [v for v in bp.filter(
        'vehicle.*') if 'harley-davidson' not in v.tags and 'yamaha' not in v.tags and 'kawasaki' not in v.tags and 'crossbike' not in v.tags and 'omafiets' not in v.tags and 'vespa' not in v.tags]
    for i in range(num_vehicles):
        vehicle_bp = random.choice(car_bps)
        transform = spawn_points[i+1]
        npc = world.try_spawn_actor(vehicle_bp, transform)
        if npc:
            npc.set_autopilot(True, tm_port)
            vehicles.append(npc)
    print(f"{len(vehicles)} 台のNPC車両をスポーン")
    return vehicles


def spawn_npc_pedestrians(world, client, bp, num_walkers):
    SpawnActor = carla.command.SpawnActor
    # 初期化
    walkers_list = []
    controllers = []
    Running_ratio = 0.0

    # 1. スポーン位置の収集
    spawn_points = []
    for i in range(num_walkers):
        spawn_point = carla.Transform()
        loc = world.get_random_location_from_navigation()
        if (loc != None):
            spawn_point.location = loc
            spawn_points.append(spawn_point)
    print(f"収集した歩行者のスポーンポイント: {len(spawn_points)}")

    # 2. 歩行者のスポーン
    walker_bps = bp.filter('walker.pedestrian.*')
    batch = list()
    for spawn_point in spawn_points:
        walker_bp = random.choice(walker_bps)
        batch.append(SpawnActor(walker_bp, spawn_point))
    results = client.apply_batch_sync(batch, True)
    walker_ids = []
    for i in range(len(results)):
        if results[i].error:
            print(f"Walker {i} のスポーンに失敗: {results[i].error}")
        else:
            walker_ids.append(results[i].actor_id)
    print(f"{len(walker_ids)} 人のNPC歩行者をスポーン")

    # 3. 歩行者コントローラのスポーン
    batch = list()
    walker_controller_bp = bp.find('controller.ai.walker')
    for i in range(len(walker_ids)):
        batch.append(SpawnActor(walker_controller_bp,
                     carla.Transform(), walker_ids[i]))
    results = client.apply_batch_sync(batch, True)
    walker_controller_ids = list()
    for i in range(len(results)):
        if results[i].error:
            print(f"Walker Controller {i} のスポーンに失敗: {results[i].error}")
        else:
            walker_controller_ids.append(results[i].actor_id)
    print(f"{len(walker_controller_ids)} 人のNPC歩行者コントローラをスポーン")

    # 4. 歩行者とコントローラの紐付けと設定
    all_ids = list()
    for i in range(len(walker_ids)):
        all_ids.append(walker_ids[i])
        all_ids.append(walker_controller_ids[i])
    all_actors = world.get_actors(all_ids)
    for i in range(0, len(all_actors), 2):
        walker = all_actors[i]
        controller = all_actors[i+1]
        controller.start()
        controller.go_to_location(world.get_random_location_from_navigation())
    return all_actors


def spawn_Ego_vehicles(client, world, bp, spawn_points):
    spawn_point = spawn_points[-2]
    ego_vehicle = world.try_spawn_actor(bp, spawn_point)
    if ego_vehicle:
        ego_vehicle.set_autopilot(True)
        print("Ego vehicle spawned")
    else:
        print("Failed to spawn Ego vehicle")

    return ego_vehicle


def show_queue_content(queue, display_name):
    for i in range(queue.qsize()):
        image = queue[i]
        cv2.imshow(display_name, image)
        cv2.waitKey(1)
    else:
        print(f"{display_name} is empty")


def save_images(image_queues, cameras, output_dir):
    for i, camera in enumerate(cameras):
        image_queue = image_queues[i]
        camera_name = camera.attributes['role_name']
        num_images = image_queue.qsize()
        print(f"Saving {num_images} images from {camera_name}...")
        save_dir = f"{output_dir}/{camera_name}"
        os.makedirs(save_dir, exist_ok=True)
        num_frame = 0
        while not image_queue.empty():
            image = image_queue.get()
            image_path = f"{save_dir}/{num_frame:06d}.png"
            cv2.imwrite(image_path, image)
            cv2.imshow(camera_name, image)
            num_frame += 1


def save_labels(label_queues, cameras, output_dir):
    for i, camera in enumerate(cameras):
        label_queue = label_queues[i]
        camera_name = camera.attributes['role_name']
        print(f"Saving labels from {camera_name}...")
        save_dir = f"{output_dir}/{camera_name}"
        os.makedirs(save_dir, exist_ok=True)
        num_frame = 0
        while not label_queue.empty():
            labels = label_queue.get()
            label_path = f"{save_dir}/{num_frame:06d}.csv"
            with open(label_path, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(
                    ['class_id', 'xmin', 'xmax', 'ymin', 'ymax', 'dist'])
                for label in labels:
                    writer.writerow(label)
            num_frame += 1


def cleanup(client, world, vehicles, all_actors, cameras, depth_cameras):
    print("クリーンアップを開始")
    for camera in cameras:
        if camera:
            camera.stop()
            camera.destroy()
            print(f"{camera.attributes['role_name']} を破棄")
    print(f"{len(cameras)} 台のカメラを破棄")
    for depth_camera in depth_cameras:
        if depth_camera:
            depth_camera.stop()
            depth_camera.destroy()
            print(f"{depth_camera.attributes['role_name']} を破棄")
    print(f"{len(depth_cameras)} 台の深度カメラを破棄")
    for vehicle in vehicles:
        if vehicle:
            vehicle.destroy()
    print(f"{len(vehicles)} 台のNPC車両を破棄")
    for i in range(0, len(all_actors), 2):
        all_actors[i+1].stop()  # コントローラ停止
    client.apply_batch([carla.command.DestroyActor(x)
                        for x in all_actors])  # 歩行者とコントローラをまとめて破棄
    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)
    traffic_manager = client.get_trafficmanager()
    traffic_manager.set_synchronous_mode(False)
    print("シミュレーションを非同期モードに設定")
    print("クリーンアップが完了")
