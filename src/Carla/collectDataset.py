import argparse
import logging
import time
import sys
import os
import cv2

# 自作モジュールのインポート
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
logger = logging.getLogger("Collector")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Multi-Map Dataset Collector")
    # 複数のマップをリストとして受け取る
    parser.add_argument("--maps", nargs='+', default=["Town01", "Town02", "Town03", "Town04", "Town05"],
                        help="データ収集を行うマップのリスト (スペース区切り)")
    parser.add_argument("--duration", type=float,
                        default=60.0, help="各マップでの収集時間(秒)")
    parser.add_argument("--vehicles", type=int, default=50, help="NPC車両の数")
    parser.add_argument("--walkers", type=int, default=50, help="NPC歩行者の数")
    parser.add_argument("--start_frame", type=int,
                        default=0, help="保存ファイル名の開始番号")
    parser.add_argument("--ratio", type=float, default=0.7,
                        help="全スポーンポイントに対する車両の割合 (0.0 - 1.0)")
    return parser.parse_args()


def collect_on_single_map(map_name, args, start_frame_idx):
    """
    1つのマップでシミュレーションを実行し、データを保存する関数
    Returns: 次のマップに引き継ぐフレーム番号
    """
    logger.info(f"=== マップ開始: {map_name} ===")

    # 1. 設定の更新
    cfg = SimulationConfig()
    cfg.map_name = map_name
    cfg.num_walkers = args.walkers
    cfg.time_duration = args.duration
    cfg.base_output_dir = "C:\\CARLA_Latest\\WindowsNoEditor\\trainDataset"
    cfg.create_directories()

    # 2. マネージャーの初期化
    world_mgr = CarlaWorldManager(cfg)
    actor_mgr = ActorManager(world_mgr)
    ego = EgoVehicle(world_mgr.world, cfg)
    label_gen = LabelGenerator(cfg)
    writer = DatasetWriter(cfg)

    current_frame_idx = start_frame_idx
    frames_per_map = int(cfg.time_duration / cfg.fixed_delta_seconds)

    try:
        # --- シミュレーション準備 ---
        spawn_points = world_mgr.get_spawn_points()

        if not spawn_points:
            logger.error(f"{map_name} にスポーンポイントがありません。スキップします。")
            return current_frame_idx

        # === 最大数の7割を計算 ===
        total_points = len(spawn_points)
        num_vehicles = int(total_points * args.ratio)

        logger.info(
            f"Spawn Points: {total_points} -> Spawning {num_vehicles} vehicles ({args.ratio*100:.0f}%)")

        # NPCスポーン (計算した車両数を渡す)
        actor_mgr.spawn_vehicles(num_vehicles, spawn_points)
        actor_mgr.spawn_walkers(cfg.num_walkers)

        # 自車スポーン
        if not spawn_points:
            logger.error(f"{map_name} にスポーンポイントがありません。スキップします。")
            return current_frame_idx

        # 必要に応じてランダム化しても良い: random.choice(spawn_points)
        ego.spawn(spawn_points[0])
        ego.setup_sensors()

        # ウォーミングアップ（物理安定化）
        for _ in range(20):
            world_mgr.tick()

        logger.info(f"{map_name} での収集を開始します ({frames_per_map} フレーム予定)")

        # --- メインループ ---
        for i in range(frames_per_map):
            # 1. 時間を進める (Tick)
            target_frame_id = world_mgr.world.tick()

            # 進捗表示
            if i % 50 == 0:
                print(
                    f"  Map: {map_name} | Progress: {i}/{frames_per_map} | Total Saved: {current_frame_idx}")

                # 2. センサーデータ取得 (同期処理付き)
                sensor_data = ego.get_sensor_data(target_frame_id)
                if not sensor_data:
                    continue

                # 3. ラベル生成
                results = label_gen.process_frame(world_mgr.world, sensor_data)

                # 4. 保存（ファイル名は通し番号 current_frame_idx を使用）
                writer.save_frame(current_frame_idx, results)

                # プレビュー表示（負荷軽減のため時々表示）
                for cam_name, res in results.items():
                    if 'bbox_image' in res:
                        cv2.imshow(f"Preview", res['bbox_image'])
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("ユーザーによる中断")
                    raise KeyboardInterrupt

            current_frame_idx += 1

    except KeyboardInterrupt:
        raise  # 外側のループも止めるために再送出
    except Exception as e:
        logger.error(f"{map_name} でエラーが発生しました: {e}", exc_info=True)
    finally:
        # --- クリーンアップ ---
        logger.info(f"=== マップ終了処理: {map_name} ===")
        cv2.destroyAllWindows()
        if 'ego' in locals():
            ego.destroy()
        if 'actor_mgr' in locals():
            actor_mgr.destroy_all()
        if 'world_mgr' in locals():
            world_mgr.cleanup()

    return current_frame_idx


def main():
    args = parse_arguments()

    total_start_time = time.time()
    current_global_frame = args.start_frame

    logger.info(f"収集対象マップ: {args.maps}")
    logger.info(f"各マップの所要時間: {args.duration}秒")

    try:
        for map_name in args.maps:
            # メモリリーク防止のため、PythonのGCを明示的に呼ぶ手もありますが、
            # クラス再生成で参照が切れるため基本的には不要です。
            current_global_frame = collect_on_single_map(
                map_name, args, current_global_frame)

            # マップ切り替えの間に少し待機（サーバーの負荷軽減）
            time.sleep(2.0)

    except KeyboardInterrupt:
        logger.info("プロセスが中断されました。")
    finally:
        elapsed = time.time() - total_start_time
        logger.info(
            f"全工程終了。総経過時間: {elapsed:.2f}秒, 総保存フレーム数: {current_global_frame}")


if __name__ == "__main__":
    main()
