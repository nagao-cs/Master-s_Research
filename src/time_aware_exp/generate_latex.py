import pandas as pd
import yaml
from pathlib import Path

def generate_latex_table_from_configs(config_dir_path):
    """
    configディレクトリの各result.csvをLaTeX表にまとめる
    
    Args:
        config_dir_path: configディレクトリのパス
    """
    config_dir = Path(config_dir_path)
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
                "Dataset": cfg.get("dataset", ""),
                "Model_1": cfg.get("model_1", ""),
                "Model_2": cfg.get("model_2", ""),
                r"$\theta_{track}$": f"{theta_track:.2f}",
                r"$\theta_{high}$": f"{theta_high:.2f}",
                "F1": f"{row_data['F1']:.4f}",
                "Precision": f"{row_data['prec']:.4f}",
                "Recall": f"{row_data['rec']:.4f}",
                "mAP": f"{row_data['mAP']:.4f}",
                "Time (s)": f"{row_data['exe_time']:.2f}",
            }
            results.append(row)
            print(f"✓ Loaded: {config_folder.name}")
        
        except Exception as e:
            print(f"✗ Error in {config_folder.name}: {e}")
    
    # DataFrameを作成（カラム順を指定）
    column_order = [
        "Dataset", "Model_1", "Model_2", 
        r"$\theta_{track}$", r"$\theta_{high}$",
        "F1", "Precision", "Recall", "mAP", "Time (s)"
    ]
    df_results = pd.DataFrame(results)
    df_results = df_results[column_order]
    
    # LaTeX表を生成
    latex_table = df_results.to_latex(index=False, escape=False)
    
    return latex_table, df_results

# 実行
config_path = r"/mnt/c/CARLA_Latest/WindowsNoEditor/src/time_aware_exp/config"
latex_output, df = generate_latex_table_from_configs(config_path)

# 結果を表示
print("\n" + "="*80)
print("LaTeX Table:")
print("="*80)
print(latex_output)

# 結果をファイルに保存（オプション）
output_file = Path(config_path) / "results_table.tex"
with open(output_file, "w") as f:
    f.write(latex_output)
print(f"\n✓ LaTeX table saved to: {output_file}")

# CSV形式でも保存（参考用）
csv_output = Path(config_path) / "results_table.csv"
df.to_csv(csv_output, index=False)
print(f"✓ CSV saved to: {csv_output}")