import carla
import math
from queue import Queue
import logging

logger = logging.getLogger(__name__)


class EgoVehicle:
    """自車両と搭載センサーを管理するクラス"""

    def __init__(self, world, config):
        self.world = world
        self.cfg = config
        self.vehicle = None
        # {'sensor': actor, 'queue': queue, 'name': str, 'type': str} のリスト
        self.sensors = []

    def spawn(self, spawn_point):
        """自車両をスポーン"""
        bp_lib = self.world.get_blueprint_library()
        vehicle_bp = bp_lib.find('vehicle.lincoln.mkz_2020')

        self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
        if self.vehicle:
            self.vehicle.set_autopilot(True)
            logger.info("Ego vehicle spawned.")
        else:
            logger.error("Failed to spawn Ego vehicle.")
            raise RuntimeError("Failed to spawn Ego vehicle.")

    def setup_sensors(self):
        """設定に基づいてRGBカメラとDepthカメラを取り付ける"""
        if not self.vehicle:
            raise RuntimeError("Vehicle not spawned yet.")

        bp_lib = self.world.get_blueprint_library()

        # センサー設定の共通化
        # 複数のカメラ配置ロジック（Front, Left, Right...）
        for i in range(self.cfg.num_camera):  # configにnum_cameraがある前提
            # 位置計算 (元のコードのロジックを踏襲)
            if i == 0:
                loc = carla.Location(x=1.5, y=0.0, z=2.0)
                base_name = "front"
            else:
                number = math.ceil(i / 2)
                y_pos = 0.3 * number if i % 2 == 1 else -0.3 * number
                prefix = "right" if i % 2 == 1 else "left"
                loc = carla.Location(x=1.5, y=y_pos, z=2.0)
                base_name = f"{prefix}_{number}"

            transform = carla.Transform(loc)

            # --- RGB Camera ---
            self._attach_sensor(bp_lib, 'sensor.camera.rgb',
                                transform, base_name)

            # --- Depth Camera ---
            self._attach_sensor(
                bp_lib, 'sensor.camera.depth', transform, base_name)

    def _attach_sensor(self, bp_lib, type_id, transform, role_name):
        """センサー生成・取付・リスナー登録のヘルパー"""
        bp = bp_lib.find(type_id)
        bp.set_attribute('image_size_x', str(self.cfg.im_width))
        bp.set_attribute('image_size_y', str(self.cfg.im_height))
        bp.set_attribute('fov', str(self.cfg.fov))
        bp.set_attribute('role_name', role_name)

        sensor_actor = self.world.spawn_actor(
            bp, transform, attach_to=self.vehicle)
        q = Queue()
        sensor_actor.listen(q.put)

        sensor_type = "rgb" if "rgb" in type_id else "depth"

        self.sensors.append({
            'actor': sensor_actor,
            'queue': q,
            'name': role_name,
            'type': sensor_type
        })

    def get_sensor_data(self, target_frame):  # 引数に target_frame を追加
        """現在のフレーム(target_frame)と一致するセンサーデータを取得する"""
        data_packet = {}

        # 全センサーについて同期確認
        for sensor_info in self.sensors:
            name = sensor_info['name']
            s_type = sensor_info['type']
            q = sensor_info['queue']

            # 正しいフレームのデータが取れるまでループ
            while True:
                if q.empty():
                    # データがまだ来ていない場合は少し待つ必要があるかも知れませんが、
                    # 同期モード(Sync)ならtick完了時点で入っているはずです。
                    # 万が一空ならブロッキングgetで待ちます。
                    data = q.get(timeout=2.0)
                else:
                    data = q.get()  # キューから取り出す

                # フレーム番号のチェック
                if data.frame == target_frame:
                    # 正解：データを採用してループを抜ける
                    if name not in data_packet:
                        data_packet[name] = {
                            'actor_rgb': None, 'actor_depth': None}

                    data_packet[name][s_type] = data

                    # Actor参照の保存（座標計算用）
                    if s_type == 'rgb':
                        data_packet[name]['actor_rgb'] = sensor_info['actor']
                    else:
                        data_packet[name]['actor_depth'] = sensor_info['actor']
                    break

                elif data.frame < target_frame:
                    # 古いデータ：捨てて次を取りに行く
                    continue

                else:
                    # 未来のデータ：ここに来ることは通常あり得ないが、万が一の場合はエラーかbreak
                    # (同期ズレが深刻な場合)
                    print(
                        f"Error: Future frame received? Target:{target_frame}, Got:{data.frame}")
                    break

        return data_packet

    def destroy(self):
        """センサーと車両を破棄"""
        for s in self.sensors:
            if s['actor']:
                s['actor'].stop()
                s['actor'].destroy()
        self.sensors = []

        if self.vehicle:
            self.vehicle.destroy()
            self.vehicle = None
        logger.info("Ego vehicle and sensors destroyed.")
