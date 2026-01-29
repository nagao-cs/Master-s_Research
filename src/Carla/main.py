import argparse
import logging
import time
import cv2
import sys

# モジュールのインポート
try:
    from configuration import SimulationConfig
    from manager import CarlaWorldManager, ActorManager
    from egoVehicle import EgoVehicle
    from labelGenerator import LabelGenerator
    from datasetWriter import DatasetWriter
except ImportError as e:
    print(f"モジュールのインポートに失敗しました: {e}")
    sys.exit(1)

# ログ設定
logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    level=logging.INFO
)
logger = logging.getLogger("Main")


def parse_arguments():
    """コマンドライン引数の解析"""
    parser = argparse.ArgumentParser(
        description="CARLA Object Detection Dataset Generator")
    parser.add_argument("--map", type=str, default="Town01",
                        help="ロードするマップ名 (例: Town01, Town02)")
    parser.add_argument("--vehicles", type=int, default=50, help="NPC車両の数")
    parser.add_argument("--walkers", type=int, default=50, help="NPC歩行者の数")
    parser.add_argument("--duration", type=float,
                        default=100.0, help="シミュレーション実行時間(秒)")
    parser.add_argument("--debug", action="store_true",
                        help="保存せずに画面表示のみ行うデバッグモード")
    return parser.parse_args()


def main():
    args = parse_arguments()

    # 1. 設定の初期化と上書き
    cfg = SimulationConfig()
    cfg.map_name = args.map
    cfg.num_vehicles = args.vehicles
    cfg.num_walkers = args.walkers
    cfg.time_duration = args.duration

    # 保存ディレクトリの準備（デバッグモード以外）
    if not args.debug:
        logger.info("データセット保存モード")
        cfg.create_directories()
    else:
        logger.info("デバッグモード")

    # 2. 環境（World）のセットアップ
    world_mgr = CarlaWorldManager(cfg)

    # 3. アクター（NPC）のセットアップ
    actor_mgr = ActorManager(world_mgr)

    # 4. 自車両（Ego）のセットアップ
    ego = EgoVehicle(world_mgr.world, cfg)

    # 5. データ処理・保存クラスのセットアップ
    label_gen = LabelGenerator(cfg)
    writer = DatasetWriter(cfg)

    try:
        # --- シミュレーション準備フェーズ ---

        # NPCのスポーン
        spawn_points = world_mgr.get_spawn_points()
        actor_mgr.spawn_vehicles(cfg.num_vehicles, spawn_points)
        actor_mgr.spawn_walkers(cfg.num_walkers)

        # 自車両のスポーン（spawn_points[0]を使用）
        if not spawn_points:
            raise RuntimeError("スポーンポイントが見つかりません。")
        ego.spawn(spawn_points[0])
        ego.setup_sensors()

        logger.info("シミュレーションを開始します...")

        # --- メインループ ---
        total_frames = int(cfg.time_duration / cfg.fixed_delta_seconds)
        start_time = time.time()

        for frame_idx in range(total_frames):
            # 1. 時間を進め、新しいフレームIDを取得
            # world.tick() は新しいフレームのIDを返します
            target_frame_id = world_mgr.world.tick()

            # 2. センサーデータ取得（フレームIDを指定して同期）
            sensor_data = ego.get_sensor_data(target_frame_id)

            if not sensor_data:
                continue

            # 3. ラベル生成と画像処理
            # process_frame は {camera_name: {'image':..., 'labels':..., 'bbox_image':...}} を返す
            results = label_gen.process_frame(world_mgr.world, sensor_data)

            # 4. データの保存
            if not args.debug:
                writer.save_frame(frame_idx, results)

            # 5. 画面表示（進捗確認用）
            for cam_name, res in results.items():
                if 'bbox_image' in res:
                    cv2.imshow(f"Preview: {cam_name}", res['bbox_image'])

            # 'q'キーで中断
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("ユーザーにより中断されました。")
                break

            # 簡易プログレスログ (100フレームごと)
            if frame_idx % 100 == 0:
                logger.info(
                    f"Progress: {frame_idx}/{total_frames} frames processed.")

    except Exception as e:
        logger.error(f"予期せぬエラーが発生しました: {e}", exc_info=True)

    finally:
        # --- クリーンアップフェーズ ---
        logger.info("クリーンアップを開始します...")

        # 画像ウィンドウを閉じる
        cv2.destroyAllWindows()

        # アクターの破棄（生成された順の逆が安全）
        if 'ego' in locals():
            ego.destroy()

        if 'actor_mgr' in locals():
            actor_mgr.destroy_all()

        if 'world_mgr' in locals():
            world_mgr.cleanup()

        elapsed_time = time.time() - start_time
        logger.info(f"完了。経過時間: {elapsed_time:.2f}秒")


if __name__ == "__main__":
    main()
