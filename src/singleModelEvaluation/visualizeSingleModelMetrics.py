import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse


def create_model_comparison_plots(
    results_csv_path: str,
    output_dir: Path
) -> None:
    """
    モデル比較プロット を生成
    
    Args:
        results_csv_path: 結果CSVファイルパス
        output_dir: 出力ディレクトリ
    """
    
    os.makedirs(output_dir / "figures", exist_ok=True)
    
    # CSV読み込み
    df = pd.read_csv(results_csv_path)
    models = df['model'].tolist()
    
    # 図1: mAPとF1の比較（バーチャート）
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # mAP
    axes[0].bar(range(len(models)), df['mAP'], color='steelblue', alpha=0.7)
    axes[0].set_xlabel('Model', fontsize=12)
    axes[0].set_ylabel('mAP', fontsize=12)
    axes[0].set_title('Mean Average Precision (mAP) by Model', fontsize=13, fontweight='bold')
    axes[0].set_xticks(range(len(models)))
    axes[0].set_xticklabels(models, rotation=45, ha='right')
    axes[0].grid(True, alpha=0.3)
    
    # F1
    axes[1].bar(range(len(models)), df['F1'], color='coral', alpha=0.7)
    axes[1].set_xlabel('Model', fontsize=12)
    axes[1].set_ylabel('F1 Score', fontsize=12)
    axes[1].set_title('F1 Score by Model', fontsize=13, fontweight='bold')
    axes[1].set_xticks(range(len(models)))
    axes[1].set_xticklabels(models, rotation=45, ha='right')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "figures" / "model_comparison_mAP_F1.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'figures' / 'model_comparison_mAP_F1.png'}")
    plt.close()
    
    # 図2: Precision-Recall 比較
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.scatter(df['Recall'], df['Precision'], s=100, alpha=0.6)
    for i, model in enumerate(models):
        ax.annotate(model, (df['Recall'][i], df['Precision'][i]), 
                   fontsize=9, xytext=(5, 5), textcoords='offset points')
    
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall by Model', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_dir / "figures" / "precision_recall_comparison.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'figures' / 'precision_recall_comparison.png'}")
    plt.close()
    
    # 図3: クラス別AP
    ap_cols = ['AP_pedestrian', 'AP_vehicle', 'AP_traffic_light', 'AP_traffic_sign']
    class_names = ['Pedestrian', 'Vehicle', 'Traffic Light', 'Traffic Sign']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(models))
    width = 0.2
    
    for i, (col, class_name) in enumerate(zip(ap_cols, class_names)):
        offset = width * (i - 1.5)
        ax.bar(x + offset, df[col], width, label=class_name, alpha=0.8)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Average Precision (AP)', fontsize=12)
    ax.set_title('AP by Class and Model', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / "figures" / "ap_by_class.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'figures' / 'ap_by_class.png'}")
    plt.close()
    
    # 図4: モデル性能ランキング
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # mAPでソート
    sorted_indices = np.argsort(df['mAP'].values)[::-1]
    sorted_models = [models[i] for i in sorted_indices]
    sorted_mAP = df['mAP'].values[sorted_indices]
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_models)))
    bars = ax.barh(sorted_models, sorted_mAP, color=colors, alpha=0.7)
    
    ax.set_xlabel('mAP', fontsize=12)
    ax.set_title('Model Ranking by mAP', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # バーの上に値を表示
    for i, (model, mAP_val) in enumerate(zip(sorted_models, sorted_mAP)):
        ax.text(mAP_val + 0.01, i, f'{mAP_val:.4f}', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / "figures" / "model_ranking.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'figures' / 'model_ranking.png'}")
    plt.close()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        description="Visualize single model metrics")
    argparser.add_argument(
        "--results_csv",
        type=str,
        default="singleModelEvaluation/results/Town02/summary_metrics.csv",
        help="Path to summary metrics CSV"
    )
    argparser.add_argument(
        "--output_dir",
        type=str,
        default="singleModelEvaluation/results",
        help="Output directory for figures"
    )
    
    args = argparser.parse_args()
    
    print(f"Creating visualization plots...")
    print(f"Input: {args.results_csv}")
    
    if not os.path.exists(args.results_csv):
        print(f"Error: CSV file not found: {args.results_csv}")
        exit(1)
    
    create_model_comparison_plots(
        results_csv_path=args.results_csv,
        output_dir=Path(args.output_dir)
    )
    
    print("\nVisualization complete!")