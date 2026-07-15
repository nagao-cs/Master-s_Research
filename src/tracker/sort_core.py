"""
sort_core.py
------------
SORT アルゴリズムのコア実装。
オリジナル実装（Bewley et al. 2016）から可視化・デモ用コードを除去し、
ライブラリとして使いやすい形に整理したもの。

依存ライブラリ:
    pip install filterpy scipy
    # lap が入っていれば高速なハンガリアン法を使用（任意）
    pip install lapjv  # または lap

オリジナル:
    https://github.com/abewley/sort
    Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai (GPL-3.0)
"""

from __future__ import annotations
import numpy as np
from filterpy.kalman import KalmanFilter


# ────────────────────────────────────────────
# ハンガリアン法（lap があれば高速版、なければ scipy）
# ────────────────────────────────────────────

def _linear_assignment(cost_matrix: np.ndarray) -> np.ndarray:
    """
    コスト行列を最小化するマッチングを返す。
    Returns: shape (N, 2) の配列 [[det_idx, trk_idx], ...]
    """
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


# ────────────────────────────────────────────
# IoU バッチ計算
# ────────────────────────────────────────────

def iou_batch(bb_test: np.ndarray, bb_gt: np.ndarray) -> np.ndarray:
    """
    2つのbboxセット間の IoU 行列を計算する。

    Args:
        bb_test: shape (M, 4)  [x1, y1, x2, y2]
        bb_gt:   shape (N, 4)  [x1, y1, x2, y2]
    Returns:
        shape (M, N) の IoU 行列
    """
    bb_gt   = np.expand_dims(bb_gt,   0)   # (1, N, 4)
    bb_test = np.expand_dims(bb_test, 1)   # (M, 1, 4)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])

    w  = np.maximum(0.0, xx2 - xx1)
    h  = np.maximum(0.0, yy2 - yy1)
    inter = w * h

    area_test = (bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
    area_gt   = (bb_gt[...,  2] - bb_gt[...,  0]) * (bb_gt[...,  3] - bb_gt[...,  1])

    return inter / (area_test + area_gt - inter)


# ────────────────────────────────────────────
# 座標変換ユーティリティ
# ────────────────────────────────────────────

def bbox_to_z(bbox: np.ndarray) -> np.ndarray:
    """
    [x1, y1, x2, y2] → [cx, cy, s, r] (center, scale=area, aspect ratio)
    SORTの状態ベクトル観測値に変換する。
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.0
    y = bbox[1] + h / 2.0
    s = w * h
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def z_to_bbox(x: np.ndarray, score: float | None = None) -> np.ndarray:
    """
    [cx, cy, s, r, ...] → [x1, y1, x2, y2] または [x1, y1, x2, y2, score]
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    x1 = x[0] - w / 2.0
    y1 = x[1] - h / 2.0
    x2 = x[0] + w / 2.0
    y2 = x[1] + h / 2.0
    if score is None:
        return np.array([x1, y1, x2, y2]).reshape((1, 4))
    else:
        return np.array([x1, y1, x2, y2, score]).reshape((1, 5))


# ────────────────────────────────────────────
# 個別トラックのカルマンフィルタ
# ────────────────────────────────────────────

class KalmanBoxTracker:
    """
    1物体のトラックを等速モデルのカルマンフィルタで管理する。

    状態ベクトル x = [cx, cy, s, r, vcx, vcy, vs]^T  (7次元)
    観測ベクトル z = [cx, cy, s, r]^T                  (4次元)
    ※ アスペクト比 r の速度は状態に含まない（定数と仮定）
    """

    _instance_count = 0  # クラスレベルのID採番

    def __init__(self, bbox: np.ndarray):
        """
        Args:
            bbox: [x1, y1, x2, y2, score] または [x1, y1, x2, y2]
        """
        # カルマンフィルタの定義（dim_x=7, dim_z=4）
        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        # 状態遷移行列 F（等速モデル）
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ], dtype=float)

        # 観測行列 H（位置のみ観測）
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ], dtype=float)

        # 観測ノイズ R（スケール・アスペクト比の観測を少し緩める）
        self.kf.R[2:, 2:] *= 10.0

        # 初期共分散 P（速度成分は未観測なので大きな不確実性）
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P         *= 10.0

        # プロセスノイズ Q（速度変化を小さく）
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        # 初期状態を設定
        self.kf.x[:4] = bbox_to_z(bbox[:4])

        # トラック管理用の変数
        self.id             = KalmanBoxTracker._instance_count
        KalmanBoxTracker._instance_count += 1
        self.time_since_update = 0   # 最後にマッチしてからのフレーム数
        self.hits           = 0      # マッチ総数
        self.hit_streak     = 0      # 連続マッチ数
        self.age            = 0      # 総フレーム数
        self._history: list[np.ndarray] = []
        self.confidence_score = bbox[4] if len(bbox) > 4 else 1.0

    def update(self, bbox: np.ndarray) -> None:
        """検出結果でカルマンフィルタを更新する。"""
        self.time_since_update = 0
        self._history = []
        self.hits       += 1
        self.hit_streak += 1
        self.kf.update(bbox_to_z(bbox[:4]))
        self.confidence_score = bbox[4] if len(bbox) > 4 else 1.0

    def predict(self) -> np.ndarray:
        """
        1ステップ予測を進め、予測 bbox [x1, y1, x2, y2] を返す。
        スケールが負にならないようにガード付き。
        """
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] = 0.0

        self.kf.predict()
        self.age += 1

        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        pred_bbox = z_to_bbox(self.kf.x)
        self._history.append(pred_bbox)
        return self._history[-1]

    def get_state(self) -> np.ndarray:
        """現在の推定 bbox [x1, y1, x2, y2] を返す。"""
        return z_to_bbox(self.kf.x)
    
    def get_state_with_score(self) -> tuple[np.ndarray, float]:
        """現在の推定 bbox と confidence を返す。"""
        return z_to_bbox(self.kf.x), self.confidence_score
    
    @classmethod
    def reset_count(cls) -> None:
        """IDカウンタをリセット（テスト・シーン切り替え用）"""
        cls._instance_count = 0


# ────────────────────────────────────────────
# 検出↔トラック対応付け
# ────────────────────────────────────────────

def associate_detections_to_trackers(
    detections: np.ndarray,
    trackers:   np.ndarray,
    iou_threshold: float = 0.3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    検出結果と既存トラックをIoUベースでマッチングする。

    Args:
        detections:    shape (M, 4+) [x1, y1, x2, y2, ...]
        trackers:      shape (N, 5)  [x1, y1, x2, y2, track_id]
        iou_threshold: この IoU 以上でのみマッチとみなす

    Returns:
        matches:              shape (K, 2) [[det_idx, trk_idx], ...]
        unmatched_detections: shape (L,)   マッチしなかった検出インデックス
        unmatched_trackers:   shape (M,)   マッチしなかったトラックインデックス
    """
    if len(trackers) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections)),
            np.empty((0, 5), dtype=int),
        )

    iou_matrix = iou_batch(detections, trackers)

    if min(iou_matrix.shape) > 0:
        # IoU > threshold の組み合わせが一意なら直接マッチ、そうでなければハンガリアン
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = _linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty((0, 2), dtype=int)

    # マッチ候補のフィルタリング（IoU 閾値以下は除外）
    unmatched_dets  = []
    unmatched_trks  = []
    matches         = []

    det_indices_matched = set(matched_indices[:, 0]) if len(matched_indices) else set()
    trk_indices_matched = set(matched_indices[:, 1]) if len(matched_indices) else set()

    for d in range(len(detections)):
        if d not in det_indices_matched:
            unmatched_dets.append(d)

    for t in range(len(trackers)):
        if t not in trk_indices_matched:
            unmatched_trks.append(t)

    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_dets.append(m[0])
            unmatched_trks.append(m[1])
        else:
            matches.append(m.reshape(1, 2))

    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_dets), np.array(unmatched_trks)


# ────────────────────────────────────────────
# SORT メインクラス
# ────────────────────────────────────────────

class Sort:
    """
    SORT (Simple Online and Realtime Tracker) の本体。

    Args:
        max_age:       検出なしで生存させるフレーム数の上限。
                       1 = 1フレームでも消えたら即削除（論文デフォルト）
        min_hits:      トラックを「確立済み」とするための最低連続マッチ数。
                       この数に達する前はトラックを出力しない。
        iou_threshold: マッチングの IoU 下限。
    """

    def __init__(
        self,
        max_age:       int   = 1,
        min_hits:      int   = 3,
        iou_threshold: float = 0.3,
    ):
        self.max_age       = max_age
        self.min_hits      = min_hits
        self.iou_threshold = iou_threshold
        self.trackers:     list[KalmanBoxTracker] = []
        self.frame_count:  int = 0

    def update(self, dets: np.ndarray = np.empty((0, 5))) -> np.ndarray:
        """
        1フレーム分の検出結果でトラッカーを更新する。
        フレームに検出がない場合は np.empty((0, 5)) を渡す。

        Args:
            dets: shape (M, 5) [[x1, y1, x2, y2, score], ...]

        Returns:
            shape (K, 5) [[x1, y1, x2, y2, track_id], ...]
            ※ min_hits 未満のトラックは含まれない
        """
        self.frame_count += 1

        # ── 予測フェーズ ──────────────────────────────────────
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        for t, trk_arr in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk_arr[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)

        # NaN が出たトラックを削除
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        # ── マッチングフェーズ ────────────────────────────────
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            dets, trks, self.iou_threshold
        )

        # マッチしたトラックを更新
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        # 新規トラックを作成（unmatched_dets）
        for i in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(dets[i, :]))

        # ── 出力フェーズ ──────────────────────────────────────
        ret = []
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            conf = trk.confidence_score
            i -= 1
            # min_hits に達した「確立済み」トラックのみ出力
            confirmed = (trk.hit_streak >= self.min_hits) or (self.frame_count <= self.min_hits)
            if (trk.time_since_update < 1) and confirmed:
                ret.append(np.concatenate((d, [trk.id + 1, conf])).reshape(1, -1))
            # max_age を超えたトラックを削除
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        return np.concatenate(ret) if ret else np.empty((0, 6)) 

    def reset(self) -> None:
        """トラッカーをリセットする（シーン切り替え時など）。"""
        self.trackers    = []
        self.frame_count = 0
        KalmanBoxTracker.reset_count()