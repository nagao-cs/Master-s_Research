import carla
import random
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
    print(f"World settings applied: Synchronous mode: {synchronous_mode}, Fixed delta seconds: {fixed_delta_seconds}")
    return world

def setting_traffic_manager(client, synchronous_mode):
    traffic_manager = client.get_trafficmanager()
    traffic_manager.set_synchronous_mode(synchronous_mode)
    tm_port = traffic_manager.get_port()
    print(f"Traffic manager settings applied: Synchronous mode: {synchronous_mode}, Port: {tm_port}")
    return traffic_manager, tm_port

def spawn_npc_vehicles(world, bp, traffic_manager, spawn_points, car_ratio):
    tm_port = traffic_manager.get_port()
    num_spawn_points = len(spawn_points)
    vehicles = list()
    num_vehicles = int(num_spawn_points * car_ratio)
    car_bps = [v for v in bp.filter('vehicle.*') if 'bike' not in v.id and 'bicycle' not in v.id and 'motorcycle' not in v.id and 'vespa' not in v.id]
    for i in range(num_vehicles):
        vehicle_bp = random.choice(car_bps)
        transform = spawn_points[i+1]
        npc = world.try_spawn_actor(vehicle_bp, transform)
        if npc:
            npc.set_autopilot(True, tm_port)
            vehicles.append(npc)
    print(f"{len(vehicles)} 台のNPC車両をスポーン")
    return vehicles

def spawn_npc_pedestrians(world, bp, num_walkers):
    pedestrians = list()
    walker_controllers = list()
    for i in range(num_walkers):
        walker_bp = random.choice(bp.filter('walker.pedestrian.*'))
        loc = world.get_random_location_from_navigation()
        if loc:
            walker = world.try_spawn_actor(walker_bp, carla.Transform(loc))
            if walker:
                ctrl_bp = bp.find('controller.ai.walker')
                ctrl = world.try_spawn_actor(ctrl_bp, carla.Transform(), walker)
                if ctrl:
                    ctrl.start()
                    ctrl.go_to_location(world.get_random_location_from_navigation())
                    ctrl.set_max_speed(1.0 + random.random())
                    walker_controllers.append(ctrl)
                pedestrians.append(walker)
    print(f"{len(pedestrians)} 人のNPC歩行者をスポーン")    
    return pedestrians, walker_controllers

def spawn_Ego_vehicles(client, world, bp, spawn_points):
    spawn_point = spawn_points[-2]
    ego_vehicle = world.try_spawn_actor(bp, spawn_point)
    if ego_vehicle:
        ego_vehicle.set_autopilot(True)
        print("Ego vehicle spawned")
    else:
        print("Failed to spawn Ego vehicle")
        
        
    return ego_vehicle

def cleanup(client, world, vehicles, pedestrians, walker_controllers, cameras):
    print("クリーンアップを開始")
    for camera in cameras:
        if camera:
            camera.stop()
            camera.destroy()
            print(f"{camera.attributes['role_name']} を破棄")
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
    print(f"{len(walker_controllers)} 人のNPC歩行者コントローラを破棄")
    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)
    traffic_manager = client.get_trafficmanager()
    traffic_manager.set_synchronous_mode(False)
    print("シミュレーションを非同期モードに設定")
    print("クリーンアップが完了")