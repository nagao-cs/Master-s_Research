import carla
from queue import Queue
import math

def setting_camera(world, bp_library, ego_vehicle, im_width, im_height, fov, num_camera):
    cameras = list()
    ques = list()
    for i in range(num_camera):
        camera_bp = bp_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(im_width))
        camera_bp.set_attribute('image_size_y', str(im_height))
        camera_bp.set_attribute('fov', str(fov))
        if i == 0:
            camera_bp.set_attribute('role_name', 'front')
            camera_transform = carla.Transform(carla.Location(x=1.5, y=0.0, z=2.0))
        elif i % 2 == 1:
            number = math.ceil(i / 2)
            camera_bp.set_attribute('role_name', f'right_{number}')
            camera_transform = carla.Transform(carla.Location(x=1.5, y=0.3*(number), z=2.0))
        else:
            number = math.ceil(i / 2)
            camera_bp.set_attribute('role_name', f'left_{number}')
            camera_transform = carla.Transform(carla.Location(x=1.5, y=-0.3*(number), z=2.0))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=ego_vehicle)
        q = Queue()
        camera.listen(q.put)
        cameras.append(camera)
        ques.append(q)
        print(f"{camera_bp.get_attribute('role_name')} カメラをスポーン")
    print(f"{len(cameras)} 台のカメラをスポーン")
    #debug
    for camera in cameras:
        print(f"role:{camera.attributes['role_name']}, Location: {camera.get_location()}")
    return cameras, ques

def create_save_queues(num_camera):
    row_image_ques = [Queue() for _ in range(num_camera)]
    bbox_image_ques = [Queue() for _ in range(num_camera)]
    label_ques = [Queue() for _ in range(num_camera)]
    return row_image_ques, bbox_image_ques, label_ques

