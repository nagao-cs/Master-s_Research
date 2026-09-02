#!/usr/bin/env python3
import pandas as pd
import yaml
from pathlib import Path
from itertools import product

def analyze_state_execution():
    """
    config_...ディレクトリ内の全result.csvから
    State1とState2の実行回数を分析
    """
    config_dir = Path(__file__).parent / "config"
    results = []
    
    # config_xxxxxx_* パターンのフォルダを列挙
    for config_folder in sorted(config_dir.glob("config_*")):
        if not config_folder.is_dir():
            continue
        
        yaml_path = config_folder / "default.yaml"
        csv_path = config_folder / "result.csv"
        
        if not yaml_path.exists() or not csv_path.exists():
            continue
        
        try:
            # YAMLからメタデータを読み込む
            with open(yaml_path) as f:
                cfg = yaml.safe_load(f)
            
            # CSVから結果を読み込む
            df_csv = pd.read_csv(csv_path)
            row_data = df_csv.iloc[0]  # 最初の行を取得
            
            # thresholds から theta_track と theta_high を取得
            thresholds = cfg.get("thresholds", {})
            theta_track = thresholds.get("theta_track", "N/A")
            theta_high = thresholds.get("theta_high", "N/A")
            
            row = {
                "Config": config_folder.name,
                "Dataset": cfg.get("dataset", ""),
                "Model_1": cfg.get("model_1", ""),
                "Model_2": cfg.get("model_2", ""),
                "theta_track": theta_track,
                "theta_high": theta_high,
                "State_1": int(row_data.get('state_1', 0)),
                "State_2": int(row_data.get('state_2', 0)),
                "State_3": int(row_data.get('state_3', 0)),
                "F1": f"{row_data['F1']:.4f}",
                "mAP": f"{row_data['mAP']:.4f}",
                "exe_time": f"{row_data['exe_time']:.2f}",
            }
            results.append(row)
            print(f"✓ Loaded: {config_folder.name}")
        
        except Exception as e:
            print(f"✗ Error in {config_folder.name}: {e}")
    
    # DataFrameを作成
    df_results = pd.DataFrame(results)
    
    # 表示用のDataFrameを作成
    display_columns = [
        "Dataset", "Model_1", "Model_2", "theta_track", "theta_high",
        "State_1", "State_2", "State_3", "F1", "mAP", "exe_time"
    ]
    df_display = df_results[display_columns]
    
    print("\n" + "="*120)
    print("State Execution Analysis")
    print("="*120)
    print(df_display.to_string(index=False))
    print("="*120 + "\n")
    
    # ===== 感度分析: theta_track と theta_high による影響 =====
    print("\n" + "="*120)
    print("SENSITIVITY ANALYSIS: State Execution by Threshold Values")
    print("="*120)
    
    for dataset in df_results["Dataset"].unique():
        for model_2 in df_results["Model_2"].unique():
            df_subset = df_results[
                (df_results["Dataset"] == dataset) &
                (df_results["Model_2"] == model_2)
            ]
            
            if len(df_subset) == 0:
                continue
            
            print(f"\n{dataset} + {df_subset['Model_1'].iloc[0]} + {model_2}:")
            print("-" * 100)
            
            # theta_track と theta_high でピボット
            pivot_state1 = df_subset.pivot_table(
                index="theta_track",
                columns="theta_high",
                values="State_1",
                aggfunc='first'
            )
            
            pivot_state2 = df_subset.pivot_table(
                index="theta_track",
                columns="theta_high",
                values="State_2",
                aggfunc='first'
            )
            
            print("\nState_1 Execution Count:")
            print(pivot_state1)
            
            print("\nState_2 Execution Count:")
            print(pivot_state2)
            
            print("\nState_1 / Total Ratio (%):")
            pivot_ratio = (pivot_state1 / (pivot_state1 + pivot_state2) * 100).round(1)
            print(pivot_ratio)
    
    # ===== 統計分析 =====
    print("\n" + "="*120)
    print("STATISTICAL ANALYSIS")
    print("="*120)
    
    print(f"\nTotal configurations: {len(df_results)}")
    print(f"\nState Execution Summary:")
    print(f"  State_1 - Mean: {df_results['State_1'].mean():.1f}, Std: {df_results['State_1'].std():.1f}")
    print(f"  State_2 - Mean: {df_results['State_2'].mean():.1f}, Std: {df_results['State_2'].std():.1f}")
    print(f"  State_3 - Mean: {df_results['State_3'].mean():.1f}, Std: {df_results['State_3'].std():.1f}")
    
    print(f"\nState_1 Ratio (State_1 / Total):")
    total = df_results['State_1'] + df_results['State_2'] + df_results['State_3']
    state1_ratio = (df_results['State_1'] / total * 100)
    print(f"  Mean: {state1_ratio.mean():.1f}%")
    print(f"  Min:  {state1_ratio.min():.1f}%")
    print(f"  Max:  {state1_ratio.max():.1f}%")
    
    # ===== モデル別分析 =====
    print("\n" + "="*120)
    print("ANALYSIS BY MODEL COMBINATION")
    print("="*120)
    
    model_analysis = df_results.groupby(["Model_1", "Model_2"]).agg({
        "State_1": ["mean", "std"],
        "State_2": ["mean", "std"],
        "F1": lambda x: f"{float(x.str.replace('nan', '0').iloc[0]):.4f}",
        "mAP": lambda x: f"{float(x.str.replace('nan', '0').iloc[0]):.4f}",
    }).round(2)
    
    print("\n")
    print(model_analysis)
    
    # CSVで保存
    output_dir = config_dir / "analysis_results"
    output_dir.mkdir(exist_ok=True)
    
    csv_output = output_dir / "state_execution_analysis.csv"
    df_display.to_csv(csv_output, index=False)
    print(f"\n✓ Results saved to: {csv_output}")
    
    # LaTeX形式で保存
    latex_output = output_dir / "state_execution_analysis.tex"
    latex_table = df_display.to_latex(index=False, escape=False)
    with open(latex_output, "w") as f:
        f.write(latex_table)
    print(f"✓ LaTeX table saved to: {latex_output}")

if __name__ == "__main__":
    analyze_state_execution()