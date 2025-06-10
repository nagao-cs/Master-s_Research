import carla
import random
def connect_to_server(host, port, timeout):
    client = carla.Client(host, port)
    client.set_timeout(timeout) 
    return client

def load_map(client, map_name):
    world = client.load_world(map_name)
    bp = world.get_blueprint_library()
    print(f"{map_name} loaded")
    return world, bp

def spawn_npc_vehicles(world, bp, traffic_manager, car_ratio):
    tm_port = traffic_manager.get_port()
    spawn_points = world.get_map().get_spawn_points()
    num_spawn_points = len(spawn_points)
    vehicles = list()
    num_vehicles = int(num_spawn_points * car_ratio)
    for i in range(num_vehicles):
        vehicle_bp = random.choice(bp.filter('vehicle.*'))
        transform = spawn_points[i+1]
        npc = world.try_spawn_actor(vehicle_bp, transform)
        if npc:
            npc.set_autopilot(True, tm_port)
            vehicles.append(npc)
    print(f"{len(vehicles)} 台のNPC車両をスポーン")
    return vehicles

# def spawn_npc_pedestrians(world, bp, traffic_manager, num_walkers):
#     pedestrians = list()
#     for i in range(num_walkers):
#         walker_bp = random.choice(bp.filter('walker.pedestrian.*'))
#         spawn_points = world.get_map().get_spawn_points()
#         transform = random.choice(spawn_points)
#         npc = world.try_spawn_actor(walker_bp, transform)
#         if npc:
#             npc.set_autopilot(True, traffic_manager.get_port())
#             pedestrians.append(npc)