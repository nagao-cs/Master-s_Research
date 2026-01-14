import carla
import random
import logging

# ログ設定（printの代わりにloggingを使うと管理しやすいです）
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CarlaWorldManager:
    """CARLAサーバーへの接続とワールド設定を管理するクラス"""

    def __init__(self, config):
        self.cfg = config
        self.client = None
        self.world = None
        self.bp_lib = None
        self.traffic_manager = None
        self.tm_port = None

        self._connect()
        self._load_world()
        self._apply_settings()
        self._setup_traffic_manager()

    def _connect(self):
        """サーバーへ接続"""
        self.client = carla.Client(self.cfg.host, self.cfg.port)
        self.client.set_timeout(self.cfg.timeout)
        logger.info(
            f"Connected to CARLA server at {self.cfg.host}:{self.cfg.port}")

    def _load_world(self):
        """マップのロード"""
        # 現在のマップと違う場合のみロード（高速化のため）
        if self.client.get_world().get_map().name.split('/')[-1] != self.cfg.map_name:
            self.world = self.client.load_world(self.cfg.map_name)
        else:
            self.world = self.client.get_world()

        self.bp_lib = self.world.get_blueprint_library()
        logger.info(f"Map '{self.cfg.map_name}' loaded.")

    def _apply_settings(self):
        """同期モードなどの設定適用"""
        settings = self.world.get_settings()
        settings.synchronous_mode = self.cfg.synchronous_mode
        settings.fixed_delta_seconds = self.cfg.fixed_delta_seconds
        self.world.apply_settings(settings)
        logger.info(
            f"World settings: Sync={self.cfg.synchronous_mode}, Delta={self.cfg.fixed_delta_seconds}")

    def _setup_traffic_manager(self):
        """Traffic Managerの設定"""
        self.traffic_manager = self.client.get_trafficmanager()
        self.traffic_manager.set_synchronous_mode(self.cfg.synchronous_mode)
        self.tm_port = self.traffic_manager.get_port()
        logger.info(f"Traffic Manager set up on port {self.tm_port}")

    def get_spawn_points(self):
        return self.world.get_map().get_spawn_points()

    def tick(self):
        """シミュレーションを1ステップ進める"""
        self.world.tick()

    def cleanup(self):
        """設定を元に戻す"""
        if self.world:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)

        if self.traffic_manager:
            self.traffic_manager.set_synchronous_mode(False)

        logger.info("World cleanup completed (Async mode restored).")


class ActorManager:
    """NPC（車両・歩行者）のスポーンと管理を行うクラス"""

    def __init__(self, world_manager):
        self.client = world_manager.client
        self.world = world_manager.world
        self.bp_lib = world_manager.bp_lib
        self.tm_port = world_manager.tm_port

        self.vehicle_list = []
        self.walker_list = []      # 歩行者本体
        self.controller_list = []  # AIコントローラ

    def spawn_vehicles(self, num_vehicles, spawn_points):
        """NPC車両のスポーン"""
        if not spawn_points:
            logger.warning("No spawn points available for vehicles.")
            return

        # バイクなどを除外するフィルタリング
        car_bps = [
            v for v in self.bp_lib.filter('vehicle.*')
            if all(tag not in v.tags for tag in ['harley-davidson', 'yamaha', 'kawasaki', 'crossbike', 'omafiets', 'vespa'])
        ]

        num_spawn = min(num_vehicles, len(spawn_points) - 1)

        for i in range(num_spawn):
            blueprint = random.choice(car_bps)
            if blueprint.has_attribute('color'):
                color = random.choice(
                    blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)

            # spawn_points[0]はEgo車両用に空けておくため +1
            transform = spawn_points[i + 1]
            vehicle = self.world.try_spawn_actor(blueprint, transform)

            if vehicle:
                vehicle.set_autopilot(True, self.tm_port)
                self.vehicle_list.append(vehicle)

        logger.info(f"Spawned {len(self.vehicle_list)} NPC vehicles.")

    def spawn_walkers(self, num_walkers):
        """NPC歩行者のスポーン"""
        # 1. スポーン位置の取得
        spawn_points = []
        for _ in range(num_walkers):
            loc = self.world.get_random_location_from_navigation()
            if loc:
                spawn_point = carla.Transform(location=loc)
                spawn_points.append(spawn_point)

        # 2. 歩行者本体のバッチスポーン
        batch = []
        walker_bps = self.bp_lib.filter('walker.pedestrian.*')
        SpawnActor = carla.command.SpawnActor

        for sp in spawn_points:
            walker_bp = random.choice(walker_bps)
            batch.append(SpawnActor(walker_bp, sp))

        results = self.client.apply_batch_sync(batch, True)
        walker_ids = [r.actor_id for r in results if not r.error]

        # 3. コントローラのバッチスポーン
        batch = []
        controller_bp = self.bp_lib.find('controller.ai.walker')
        for wid in walker_ids:
            batch.append(SpawnActor(controller_bp, carla.Transform(), wid))

        results = self.client.apply_batch_sync(batch, True)
        controller_ids = [r.actor_id for r in results if not r.error]

        # IDリストからアクターオブジェクトを取得してリストに保存
        self.walker_list = self.world.get_actors(walker_ids)
        self.controller_list = self.world.get_actors(controller_ids)

        # 4. コントローラの起動と目的地設定
        self.world.tick()  # コントローラの初期化待ち
        for controller in self.controller_list:
            controller.start()
            controller.go_to_location(
                self.world.get_random_location_from_navigation())
            # ランダムに少し走らせる設定などもここで可能

        logger.info(
            f"Spawned {len(self.walker_list)} walkers and controllers.")

    def destroy_all(self):
        """管理している全アクターを破棄"""
        # コントローラの停止
        for controller in self.controller_list:
            controller.stop()

        # バッチ処理で一括削除
        all_ids = [v.id for v in self.vehicle_list] + \
                  [w.id for w in self.walker_list] + \
                  [c.id for c in self.controller_list]

        if all_ids:
            self.client.apply_batch(
                [carla.command.DestroyActor(x) for x in all_ids])

        logger.info(f"Destroyed {len(all_ids)} actors (vehicles + walkers).")

        # リストをクリア
        self.vehicle_list = []
        self.walker_list = []
        self.controller_list = []
