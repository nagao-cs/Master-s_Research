import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class DetectionStats:
    def __init__(self):
        self.dataframe = pd.DataFrame(columns=[
            'frame_idx',
            'num_gt_instances',
            'num_detection_instances',
            'num_true_positives',
            'num_false_positives',
            'num_false_negatives',
        ])
        self.tp_stats = dict()
        self.fp_stats = dict()
        self.fn_stats = dict()
        self.class_stats = dict()

    def update(self, frame_idx: int, gt: dict, analyzed_detctions: dict):
        for class_id, boxes in analyzed_detctions['total_instances']['TP'].items():
            if class_id not in self.tp_stats:
                self.tp_stats[class_id] = list()
            self.tp_stats[class_id].extend(boxes)
        for class_id, boxes in analyzed_detctions['total_instances']['FP'].items():
            if class_id not in self.fp_stats:
                self.fp_stats[class_id] = list()
            self.fp_stats[class_id].extend(boxes)
        for class_id, boxes in analyzed_detctions['total_instances']['FN'].items():
            if class_id not in self.fn_stats:
                self.fn_stats[class_id] = list()
            self.fn_stats[class_id].extend(boxes)
        for class_id, boxes in analyzed_detctions['total_instances']['TP'].items():
            if class_id not in self.class_stats:
                self.class_stats[class_id] = {'TP': 0, 'FP': 0, 'FN': 0}
            self.class_stats[class_id]['TP'] += len(boxes)
        for class_id, boxes in analyzed_detctions['total_instances']['FP'].items():
            if class_id not in self.class_stats:
                self.class_stats[class_id] = {'TP': 0, 'FP': 0, 'FN': 0}
            self.class_stats[class_id]['FP'] += len(boxes)
        for class_id, boxes in analyzed_detctions['total_instances']['FN'].items():
            if class_id not in self.class_stats:
                self.class_stats[class_id] = {'TP': 0, 'FP': 0, 'FN': 0}
            self.class_stats[class_id]['FN'] += len(boxes)

        num_gt_instances = sum(len(boxes) for boxes in gt.values())
        num_detection_instances = sum(
            len(boxes) for boxes in analyzed_detctions['total_instances']['TP'].values()) + sum(
            len(boxes) for boxes in analyzed_detctions['total_instances']['FP'].values())
        num_true_positives = sum(
            len(boxes) for boxes in analyzed_detctions['total_instances']['TP'].values())
        num_false_positives = sum(
            len(boxes) for boxes in analyzed_detctions['total_instances']['FP'].values())
        num_false_negatives = sum(
            len(boxes) for boxes in analyzed_detctions['total_instances']['FN'].values())

        new_row = {
            'frame_idx': frame_idx,
            'num_gt_instances': num_gt_instances,
            'num_detection_instances': num_detection_instances,
            'num_true_positives': num_true_positives,
            'num_false_positives': num_false_positives,
            'num_false_negatives': num_false_negatives,
        }
        self.dataframe = pd.concat(
            [self.dataframe, pd.DataFrame([new_row])], ignore_index=True)

    def analyze_model_specific_errors(self, comparison_results: dict):
        model_specific_fp = comparison_results['model_specific_FP']
        model_specific_fn = comparison_results['model_specific_FN']

        total_model_specific_fp = sum(
            len(boxes) for boxes in model_specific_fp.values())
        total_model_specific_fn = sum(
            len(boxes) for boxes in model_specific_fn.values())

        print(f"Model-Specific False Positives: {total_model_specific_fp}")
        print(f"Model-Specific False Negatives: {total_model_specific_fn}")

    def plot_transition(self):
        plt.figure(figsize=(12, 8))
        plt.plot(self.dataframe['frame_idx'], self.dataframe['num_gt_instances'],
                 label='GT Instances', color='blue')
        plt.plot(self.dataframe['frame_idx'], self.dataframe['num_detection_instances'],
                 label='Detection Instances', color='orange')

        plt.xlabel('Frame Index')
        plt.ylabel('Number of Instances')
        plt.title('Detection Statistics Over Time')
        plt.legend()
        plt.grid()
        plt.show()

    def plot_detection_analysis(self):
        class_id_name_map = {
            0: 'Pedestrian',
            2: 'Vehicle',
            9: 'Traffic Light',
            11: 'Traffic Sign'
        }

        # TPの数とconfidence分布を2行のサブプロットで表示
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

        # True Positives の数
        tp_counts = []
        class_names = []
        confidence_distributions = dict()
        valid_indices = list()

        for i, (class_id, boxes) in enumerate(self.tp_stats.items()):
            class_name = class_id_name_map.get(class_id, f'Class {class_id}')
            class_names.append(class_name)
            tp_counts.append(len(boxes))

            # confidenceスコアの抽出 (boxesの最後の要素がconfidenceと仮定)
            confidences = [box[-1] for box in boxes]
            if confidences:
                confidence_distributions[class_id] = confidences
                valid_indices.append(i)

        # TP数の棒グラフ
        ax1.bar(class_names, tp_counts)
        ax1.set_xlabel('Class')
        ax1.set_ylabel('Number of True Positives')
        ax1.set_title('True Positives by Class')
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

        # Confidence分布の箱ひげ図
        if confidence_distributions:  # 有効なconfidenceデータがある場合のみプロット
            # リストとしてconfidence値を保持
            confidence_values = []
            labels = []

            # 各クラスのconfidence値をリストに追加
            for class_id, confidences in confidence_distributions.items():
                class_name = class_id_name_map.get(
                    class_id, f'Class {class_id}')
                confidence_values.append(confidences)
                labels.append(class_name)

            # 箱ひげ図の描画
            bp = ax2.boxplot(confidence_values, labels=labels)

            # 箱ひげ図のスタイル設定
            plt.setp(bp['boxes'], color='blue', alpha=0.7)
            plt.setp(bp['whiskers'], color='black', alpha=0.7)
            plt.setp(bp['medians'], color='red')

            ax2.set_xlabel('Class')
            ax2.set_ylabel('Confidence Score')
            ax2.set_title('Confidence Distribution of True Positives')
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        plt.tight_layout()
        plt.show()

        # FPの数とconfidence分布を2行のサブプロットで表示
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

        # False Positives の数
        fp_counts = []
        class_names = []
        confidence_distributions = dict()
        valid_indices = list()

        for i, (class_id, boxes) in enumerate(self.fp_stats.items()):
            class_name = class_id_name_map.get(class_id, f'Class {class_id}')
            class_names.append(class_name)
            fp_counts.append(len(boxes))

            # confidenceスコアの抽出 (boxesの最後の要素がconfidenceと仮定)
            confidences = [box[-1] for box in boxes]
            if confidences:
                confidence_distributions[class_id] = confidences
                valid_indices.append(i)

        # FP数の棒グラフ
        ax1.bar(class_names, fp_counts)
        ax1.set_xlabel('Class')
        ax1.set_ylabel('Number of False Positives')
        ax1.set_title('False Positives by Class')
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

        # Confidence分布の箱ひげ図
        if confidence_distributions:  # 有効なconfidenceデータがある場合のみプロット
            # リストとしてconfidence値を保持
            confidence_values = []
            labels = []

            # 各クラスのconfidence値をリストに追加
            for class_id, confidences in confidence_distributions.items():
                class_name = class_id_name_map.get(
                    class_id, f'Class {class_id}')
                confidence_values.append(confidences)
                labels.append(class_name)

            # 箱ひげ図の描画
            bp = ax2.boxplot(confidence_values, labels=labels)

            # 箱ひげ図のスタイル設定
            plt.setp(bp['boxes'], color='blue', alpha=0.7)
            plt.setp(bp['whiskers'], color='black', alpha=0.7)
            plt.setp(bp['medians'], color='red')

            ax2.set_xlabel('Class')
            ax2.set_ylabel('Confidence Score')
            ax2.set_title('Confidence Distribution of False Positives')
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

            # 統計情報の表示
            for i, confs in enumerate(confidence_values, 1):
                if confs:
                    median = np.median(confs)
                    mean = np.mean(confs)
                    ax2.text(i, 1.05,
                             f'μ={mean:.3f}\nm={median:.3f}',
                             horizontalalignment='center',
                             verticalalignment='bottom')

        plt.tight_layout()
        plt.show()

        # False Negative
        plt.figure(figsize=(12, 6))
        for class_id, boxes in self.fn_stats.items():
            class_name = class_id_name_map.get(class_id, f'Class {class_id}')
            plt.bar(class_name, len(boxes), label=f'Class {class_name}')
        plt.xlabel('Class ID')
        plt.ylabel('Number of False Negatives')
        plt.title('False Negatives by Class')
        plt.legend()
        plt.show()

    def analyze_class_statistics(self):
        """
        各クラスの出現回数とエラー率を分析して表示する
        """
        class_names = {
            0: 'pedestrian',
            2: 'vehicle',
            9: 'traffic_light',
            11: 'traffic_sign'
        }

        print("\n=== クラスごとの検出統計 ===")
        print(
            f"{'クラス':^15} {'総数':^8} {'TP':^8} {'FP':^8} {'FN':^8} {'精度':^8} {'再現率':^8}")
        print("-" * 70)

        for class_id, stats in self.class_stats.items():
            total = stats['TP'] + stats['FN']  # 正解の総数
            total_det = stats['TP'] + stats['FP']  # 検出の総数

            # 精度と再現率の計算
            precision = stats['TP'] / total_det if total_det > 0 else 0
            recall = stats['TP'] / total if total > 0 else 0

            class_name = class_names.get(class_id, f'クラス{class_id}')
            print(f"{class_name:^15} {total:^8d} {stats['TP']:^8d} {stats['FP']:^8d} "
                  f"{stats['FN']:^8d} {precision:^8.3f} {recall:^8.3f}")

        # クラスごとのエラー率の可視化
        plt.figure(figsize=(10, 6))
        class_ids = list(self.class_stats.keys())
        class_names_list = [class_names.get(
            cid, f'{cid}') for cid in class_ids]

        error_rates = []
        for class_id in class_ids:
            stats = self.class_stats[class_id]
            total = stats['TP'] + stats['FN']
            if total > 0:
                error_rate = (stats['FP'] + stats['FN']) / total
            else:
                error_rate = 0
            error_rates.append(error_rate)

        bars = plt.bar(class_names_list, error_rates)

        # バーの上に値を表示
        for bar, rate in zip(bars, error_rates):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                     f'{rate:.3f}', ha='center', va='bottom')

        plt.title('error rate by class')
        plt.xlabel('class')
        plt.ylabel('error rate')
        plt.ylim(0, 1.2)  # y軸の範囲を0-1.2に設定
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        # クラスごとのエラータイプの内訳
        plt.figure(figsize=(10, 6))
        x = np.arange(len(class_ids))
        width = 0.35

        fp_rates = []
        fn_rates = []
        for class_id in class_ids:
            stats = self.class_stats[class_id]
            total = stats['TP'] + stats['FN']
            if total > 0:
                fp_rate = stats['FP'] / total
                fn_rate = stats['FN'] / total
            else:
                fp_rate = fn_rate = 0
            fp_rates.append(fp_rate)
            fn_rates.append(fn_rate)

        plt.bar(x - width/2, fp_rates, width, label='FP rate')
        plt.bar(x + width/2, fn_rates, width, label='FN rate')

        plt.title('Error Type Breakdown by Class')
        plt.xlabel('class')
        plt.ylabel('error rate')
        plt.xticks(x, class_names_list, rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.show()
