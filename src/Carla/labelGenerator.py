import numpy as np
import cv2
import carla


class LabelGenerator:
    """画像の変換、BBox計算、オクルージョン判定を行うクラス"""

    def __init__(self, config):
        self.cfg = config
        # カメラ行列Kは解像度とFOVが変わらなければ固定なので初期化時に計算
        self.K = self._build_projection_matrix(
            config.im_width, config.im_height, config.fov)

    def process_frame(self, world, sensor_data_packet):
        """
        1フレーム分のセンサーデータを受け取り、ラベルとBBox描画画像を生成する
        Returns: {camera_name: {'image': np_img, 'labels': list, 'bbox_image': np_img}}
        """
        results = {}

        # 検出対象のアクターを一括取得（最適化）
        target_actors = []
        # CityObjectLabelに対応するラベルIDを取得してループ
        # ここでは簡易化のため、Configのclass_mappingのキー（CityObjectLabel）を使用
        for label_enum in [carla.CityObjectLabel.Vehicles, carla.CityObjectLabel.Pedestrians,
                           carla.CityObjectLabel.TrafficSigns, carla.CityObjectLabel.TrafficLight]:
            target_actors.extend(world.get_level_bbs(label_enum))

        for cam_name, data in sensor_data_packet.items():
            rgb_raw = data.get('rgb')
            depth_raw = data.get('depth')
            camera_actor = data.get('actor_rgb')

            if not (rgb_raw and depth_raw and camera_actor):
                continue

            # 1. 画像変換
            rgb_img = self._to_numpy_img(rgb_raw)
            depth_meters = self._to_depth_meters(depth_raw)

            # 2. カメラ行列・位置取得
            world_2_camera = np.array(
                camera_actor.get_transform().get_inverse_matrix())
            cam_loc = camera_actor.get_transform().location
            cam_forward = camera_actor.get_transform().get_forward_vector()

            visible_bboxes = []

            # 3. BBox計算とオクルージョン判定
            for bbox in target_actors:
                # 距離フィルタ
                dist = bbox.location.distance(cam_loc)
                if dist > self.cfg.valid_distance:
                    continue

                # 前方フィルタ
                ray = bbox.location - cam_loc
                if cam_forward.dot(ray) < 0:
                    continue

                # オクルージョン判定 (Depth Map使用)
                if self._is_visible(bbox=bbox, world_2_camera=world_2_camera, depth_map=depth_meters, threshold_visible=2, eps=0.3) == False:
                    continue
                # 2D投影
                yolo_bbox = self._get_2d_bbox(bbox, world_2_camera)

                if yolo_bbox is None:
                    continue
                xc, yc, w, h = yolo_bbox
                # サイズフィルタ
                if (w * self.cfg.im_width) * (h * self.cfg.im_height) > self.cfg.size_threshold:
                    # クラスID取得 (bboxにはtypeがないので、元のlogic同様リスト管理か、
                    # あるいはget_level_bbsの呼び出し元でクラスIDを付与する工夫が必要。
                    # ここでは簡略化のため、bbox自体からはクラスが取れないCARLAの仕様に対し、
                    # 上記target_actors取得ループを分け、クラスIDをセットで渡す構造にするのが理想)

                    # ※実装上の修正: world.get_level_bbs()の戻り値は単なるBoundingBoxでクラス情報を持たないため、
                    # 呼び出しループをここで行う形に修正します。
                    pass

            # リファクタリング: ループ構造の適正化
            visible_bboxes = self._compute_bboxes(
                world, cam_loc, cam_forward, world_2_camera, depth_meters)

            # 4. 描画用画像の作成
            bbox_img = rgb_img.copy()
            self._draw_bboxes(bbox_img, visible_bboxes)

            results[cam_name] = {
                'original_image': rgb_img,
                # [[class_id, xc, yc, w, h, dist], ...]
                'labels': visible_bboxes,
                'bbox_image': bbox_img
            }

        return results

    def _compute_bboxes(self, world, cam_loc, cam_forward, world_2_camera, depth_map):
        """クラスごとのBBox取得とフィルタリング"""
        frame_labels = []

        # ターゲットごとに取得
        targets = [
            (carla.CityObjectLabel.Vehicles,
             self.cfg.class_mapping[carla.CityObjectLabel.Vehicles]),
            (carla.CityObjectLabel.Pedestrians,
             self.cfg.class_mapping[carla.CityObjectLabel.Pedestrians]),
            (carla.CityObjectLabel.TrafficSigns,
             self.cfg.class_mapping[carla.CityObjectLabel.TrafficSigns]),
            (carla.CityObjectLabel.TrafficLight,
             self.cfg.class_mapping[carla.CityObjectLabel.TrafficLight]),
        ]

        for city_label, class_id in targets:
            bboxes = world.get_level_bbs(city_label)
            for bbox in bboxes:
                dist = bbox.location.distance(cam_loc)
                if dist > self.cfg.valid_distance:
                    continue

                ray = bbox.location - cam_loc
                if cam_forward.dot(ray) < 0:
                    continue

                # Depth判定
                if self._is_visible(bbox, world_2_camera, depth_map):
                    yolo_bbox = self._get_2d_bbox(bbox, world_2_camera)
                    if yolo_bbox:
                        xc, yc, w, h = yolo_bbox
                        # サイズチェック
                        if (w * self.cfg.im_width * h * self.cfg.im_height) >= self.cfg.size_threshold:
                            frame_labels.append([class_id, xc, yc, w, h, dist])

        # 重なり除去などは必要に応じて追加（元のコードのremove_overlapping_bboxes）
        return frame_labels

    def _build_projection_matrix(self, w, h, fov):
        focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
        K = np.identity(3)
        K[0, 0] = K[1, 1] = focal
        K[0, 2] = w / 2.0
        K[1, 2] = h / 2.0
        return K

    def _to_numpy_img(self, carla_img):
        array = np.frombuffer(carla_img.raw_data, dtype=np.uint8)
        array = np.reshape(array, (carla_img.height, carla_img.width, 4))
        return array[:, :, :3]  # Alpha除去

    def _to_depth_meters(self, depth_img):
        array = np.frombuffer(depth_img.raw_data, dtype=np.uint8)
        array = np.reshape(array, (depth_img.height, depth_img.width, 4))[
            :, :, :3]
        B = array[:, :, 0].astype(np.float32)
        G = array[:, :, 1].astype(np.float32)
        R = array[:, :, 2].astype(np.float32)
        normalized = (R + G * 256 + B * 256 * 256) / (256**3 - 1)
        return normalized * 1000.0

    def _project_point(self, location, world_2_camera):
        point = np.array([location.x, location.y, location.z, 1])
        point_camera = np.dot(world_2_camera, point)
        # UE4 (x,y,z) -> Camera (y, -z, x)
        point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

        point_img = np.dot(self.K, point_camera)
        if point_img[2] == 0:
            return None
        u = point_img[0] / point_img[2]
        v = point_img[1] / point_img[2]
        dist = point_img[2]
        return (u, v, dist)

    def _get_2d_bbox(self, bbox, world_2_camera):
        verts = [v for v in bbox.get_world_vertices(carla.Transform())]
        points_2d = []
        for v in verts:
            res = self._project_point(v, world_2_camera)
            if res:
                points_2d.append(res[:2])  # u, v

        if not points_2d:
            return None

        points_2d = np.array(points_2d)
        img_w, img_h = self.cfg.im_width, self.cfg.im_height

        xmin = np.clip(np.min(points_2d[:, 0]), 0, img_w - 1)
        ymin = np.clip(np.min(points_2d[:, 1]), 0, img_h - 1)
        xmax = np.clip(np.max(points_2d[:, 0]), 0, img_w - 1)
        ymax = np.clip(np.max(points_2d[:, 1]), 0, img_h - 1)

        if xmax <= xmin or ymax <= ymin:
            return None

        # YOLO format (normalized center x, center y, w, h)
        w_norm = (xmax - xmin) / img_w
        h_norm = (ymax - ymin) / img_h
        xc_norm = (xmin + xmax) / 2.0 / img_w
        yc_norm = (ymin + ymax) / 2.0 / img_h

        return (xc_norm, yc_norm, w_norm, h_norm)

    def _is_visible(self, bbox, world_2_camera, depth_map, threshold_visible=3, eps=0.3):
        """Depthマップと比較して、BBoxの頂点が手前にある（隠れていない）か判定"""
        verts = bbox.get_world_vertices(carla.Transform())
        visible_count = 0
        h, w = depth_map.shape

        for v in verts:
            res = self._project_point(v, world_2_camera)
            if res is None:
                continue
            u, v_coord, dist = res

            u_int, v_int = int(u), int(v_coord)
            if 0 <= u_int < w and 0 <= v_int < h:
                depth_val = depth_map[v_int, u_int]
                # 実際の距離がDepthマップの値とほぼ同じ（手前にある）なら可視
                if dist < depth_val + eps:
                    visible_count += 1

        return visible_count >= threshold_visible

    def _draw_bboxes(self, image, bboxes):
        h, w, _ = image.shape
        for cls, xc, yc, bw, bh, dist in bboxes:
            xmin = int((xc - bw/2) * w)
            xmax = int((xc + bw/2) * w)
            ymin = int((yc - bh/2) * h)
            ymax = int((yc + bh/2) * h)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(image, str(cls), (xmin, ymin-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
