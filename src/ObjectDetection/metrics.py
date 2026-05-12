# src/ObjectDetection/metrics.py
from dataclasses import dataclass, field
from typing import Optional
import torch
import psutil
import time
import numpy as np
from thop import profile, clever_format


@dataclass
class PerformanceMetrics:
    """推論パフォーマンスメトリクス"""
    inference_time_ms: float          # 推論時間 (ミリ秒)
    memory_peak_mb: float             # ピークメモリ使用量 (MB)
    flops_giga: Optional[float] = None  # FLOPs (G単位)
    
    def __str__(self) -> str:
        flops_str = f", FLOPs: {self.flops_giga:.2f}G" if self.flops_giga else ""
        return f"Time: {self.inference_time_ms:.2f}ms, Memory: {self.memory_peak_mb:.2f}MB{flops_str}"


class MetricsCollector:
    """推論パフォーマンスメトリクス計測"""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.process = psutil.Process()
    
    def measure_inference(
        self,
        model_fn,
        image,
        image_height,
        image_width,
        compute_flops: bool = True,
        flops_inputs: Optional[tuple] = None,
        **kwargs
    ) -> tuple:
        """
        推論を実行し、パフォーマンスメトリクスを計測
        
        Args:
            model_fn: 推論関数
            *args, **kwargs: model_fnの引数
            compute_flops: FLOPs計算を行うか
            flops_inputs: FLOPs計算用の入力（Tensorのタプル）
        
        Returns:
            (model_output, metrics)
        """
        # メモリ計測開始
        torch.cuda.reset_peak_memory_stats() if self.device == "cuda" else None
        mem_before = self._get_memory_usage()
        
        # 推論時間計測
        start_time = time.perf_counter()
        with torch.no_grad():
            output = model_fn(image)
        inference_time_ms = (time.perf_counter() - start_time) * 1000
        
        # メモリ計測終了
        mem_after = self._get_memory_usage()
        memory_peak_mb = max(mem_after - mem_before, 0.0)
        
        # GPU使用時のピークメモリ
        if self.device == "cuda":
            gpu_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
            memory_peak_mb = gpu_memory_mb
        
        # FLOPs計算（オプション）
        flops_giga = None
        if compute_flops and flops_inputs is not None:
            flops_giga = self._calculate_flops(model_fn, flops_inputs)
        
        metrics = PerformanceMetrics(
            inference_time_ms=inference_time_ms,
            memory_peak_mb=memory_peak_mb,
            flops_giga=flops_giga
        )
        
        return output, metrics
    
    def _get_memory_usage(self) -> float:
        """現在のメモリ使用量をMB単位で返す"""
        return self.process.memory_info().rss / 1024 / 1024
    
    def _calculate_flops(
        self,
        model,
        input_tensor: tuple
    ) -> float:
        """FLOPs計算（単位: Giga）"""
        try:
            flops, params = profile(model, inputs=input_tensor, verbose=False)
            flops_giga = flops / 1e9
            return flops_giga
        except Exception as e:
            print(f"Warning: FLOPs calculation failed: {e}")
            return None