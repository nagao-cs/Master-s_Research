import csv
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from src.config import RESULT_DIR

if __name__ == "__main__":
    csv_path = RESULT_DIR / "agree_error_relation/summary.csv"
    
    # 対象とするデータセットのリスト
    target_datasets = {'0020', '0019', '0001'}
    
    model_data = defaultdict(lambda: {'agree': [], 'disagree': []})

    with open(file=csv_path, mode="r") as csv_file:
        reader = csv.reader(csv_file)
        
        # ヘッダーを取得し、各カラムのインデックスを特定
        header = next(reader)
        idx_models = header.index('models')
        idx_dataset = header.index('dataset')
        idx_agree = header.index('agree_error_rate')
        idx_disagree = header.index('disagree_error_rate')
        
        # データの読み込みとフィルタリング
        for row in reader:
            if not row:
                continue  # 空行をスキップ
            print(row)
            dataset = row[idx_dataset]
            
            model = row[idx_models]
            agree_rate = float(row[idx_agree])
            disagree_rate = float(row[idx_disagree])
            model_data[model]['agree'].append(agree_rate)
            model_data[model]['disagree'].append(disagree_rate)

    # 各モデルの平均値を計算するためのリスト
    models_list = []
    avg_agree_rates = []
    avg_disagree_rates = []

    for model, rates in model_data.items():
        if rates['agree']:  # データが存在する場合のみ計算
            models_list.append(model)
            
            # 平均の算出
            agree_mean = sum(rates['agree']) / len(rates['agree'])
            disagree_mean = sum(rates['disagree']) / len(rates['disagree'])
            
            avg_agree_rates.append(agree_mean)
            avg_disagree_rates.append(disagree_mean)
    avg_disagree_rates, avg_agree_rates = map(list, zip(*sorted(zip(avg_disagree_rates, avg_agree_rates))))

    # ==========================================
    # ここからプロット処理 (matplotlib を使用)
    # ==========================================
    
    # X軸の位置を生成
    x = np.arange(len(models_list))
    width = 0.35  # 棒の幅
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 同一モデルでagreeとdisagreeを並べて表示するグループ化棒グラフ
    rects1 = ax.bar(x - width/2, avg_agree_rates, width, label='Agree Error Rate', color='skyblue')
    rects2 = ax.bar(x + width/2, avg_disagree_rates, width, label='Disagree Error Rate', color='salmon')
    
    # グラフの装飾
    ax.set_xlabel('Models')
    ax.set_ylabel('Error Rate')
    ax.set_xticks(x)
    ax.set_xticklabels(models_list, rotation=45, ha='right')  # モデル名が長い場合を考慮して斜め表示
    ax.legend()
    
    fig.tight_layout()
    plt.savefig(RESULT_DIR / "agree_error_relation/average_error_rates.png") # 保存する場合
    
    plt.figure(figsize=(8, 6))
    x = np.array(avg_agree_rates)
    y = np.array(avg_disagree_rates)
    plt.scatter(x, y)
    plt.xlabel("agree error rate")
    plt.ylabel("disagree error rate")
    plt.savefig(RESULT_DIR / "agree_error_relation/scatter_agree_error.png")