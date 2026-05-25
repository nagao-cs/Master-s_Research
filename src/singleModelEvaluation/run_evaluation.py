"""
Single Model Evaluation - 統合実行スクリプト
全モデルのメトリクス計算と可視化を一度に実行
"""

import argparse
from pathlib import Path
import os
import sys

# パスの追加
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.singleModelEvaluation.computeSingleModelMetrics import (
    compute_all_single_models,
    save_results_to_csv
)
from src.singleModelEvaluation.visualizeSingleModelMetrics import create_model_comparison_plots


def run_full_evaluation(map_name: str = "Town02", iou_threshold: float = 0.5) -> None:
    """
    フル評価パイプライン（計算 → 保存 → 可視化）
    
    Args:
        map_name: マップ名
        iou_threshold: IoU閾値
    """
    
    print("="*80)
    print("SINGLE MODEL EVALUATION PIPELINE")
    print("="*80)
    print(f"Map: {map_name}")
    print(f"IoU Threshold: {iou_threshold}")
    print()
    
    # ステップ1: メトリクス計算
    print("[Step 1/3] Computing metrics for all models...")
    print("-" * 80)
    try:
        results_df = compute_all_single_models(
            map_name=map_name,
            iou_threshold=iou_threshold
        )
        print(f"✓ Successfully computed metrics for {len(results_df)} models")
    except Exception as e:
        print(f"✗ Error computing metrics: {e}")
        return
    
    print()
    
    # ステップ2: 結果を保存
    print("[Step 2/3] Saving results to CSV...")
    print("-" * 80)
    try:
        output_dir = Path(__file__).parent / "results" / map_name
        save_results_to_csv(results_df, output_dir, map_name)
        print(f"✓ Results saved to {output_dir}")
    except Exception as e:
        print(f"✗ Error saving results: {e}")
        return
    
    print()
    
    # ステップ3: 可視化
    print("[Step 3/3] Creating visualization plots...")
    print("-" * 80)
    try:
        summary_csv = output_dir / "summary_metrics.csv"
        create_model_comparison_plots(
            results_csv_path=str(summary_csv),
            output_dir=output_dir
        )
        print(f"✓ Visualizations saved to {output_dir}/figures")
    except Exception as e:
        print(f"✗ Error creating visualizations: {e}")
        return
    
    print()
    print("="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)
    print()
    print("📊 Results Summary:")
    print("-" * 80)
    summary_df = results_df[['model', 'mAP', 'F1', 'Precision', 'Recall']]
    print(summary_df.to_string(index=False))
    print("-" * 80)
    print()
    print(f"📁 Output Directory: {output_dir}")
    print(f"   - summary_metrics.csv")
    print(f"   - detailed_metrics.csv")
    print(f"   - ap_by_class_metrics.csv")
    print(f"   - figures/")
    print()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        description="Run complete single model evaluation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_evaluation.py --map Town02
  python run_evaluation.py --map Town03 --iou_th 0.75
  python run_evaluation.py --map Town05
        """
    )
    argparser.add_argument(
        "--map",
        type=str,
        default="Town02",
        choices=["Town02", "Town03", "Town05"],
        help="Map name (default: Town02)"
    )
    argparser.add_argument(
        "--iou_th",
        type=float,
        default=0.5,
        help="IoU threshold (default: 0.5)"
    )
    
    args = argparser.parse_args()
    
    run_full_evaluation(map_name=args.map, iou_threshold=args.iou_th)