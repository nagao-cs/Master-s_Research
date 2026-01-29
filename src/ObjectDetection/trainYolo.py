from ultralytics import YOLO, RTDETR
import os
import argparse


def main():
    argparser = argparse.ArgumentParser(
        description="one version Object Detection"
    )
    argparser.add_argument(
        "--model",
        type=str,
        required=True,
    )
    args = argparser.parse_args()
    print(args)

    modelName: str = args.model
    # 1. モデルのロード
    # 初回は自動で yolov8n.pt (ナノモデル) をダウンロードします
    # 精度重視なら 'yolov8s.pt' や 'yolov8m.pt' に変更してください
    if modelName == "yolov8n":
        model = YOLO('yolov8n.pt')
    elif modelName == "yolov11n":
        model = YOLO("yolov11n.pt")
    elif modelName == "yolov5n":
        model = YOLO("yolov5nu.pt")
    elif modelName == "rtdetr":
        model = RTDETR("rtdetr-l.pt")
    else:
        print("model is not supported")
        exit(1)

    # 2. yamlファイルの絶対パスを取得
    yaml_path = os.path.abspath("carlaTrain.yaml")

    # 3. 学習の実行
    print("学習を開始します...")
    results = model.train(
        data=yaml_path,
        epochs=10,          # エポック数
        imgsz=640,          # 画像サイズ
        batch=16,           # バッチサイズ (GPUメモリに合わせて調整)
        device=0,           # GPUを使用する場合は 0, CPUなら 'cpu'
        project='trainingModel',  # 保存先プロジェクト名
        name=f'training{modelName}',        # 実験名
        patience=10,        # 10エポック精度向上がなければ早期終了
        save=True           # モデルの保存
    )

    print("学習完了！")

    # 4. 検証（オプション）
    metrics = model.val()
    print(f"mAP50-95: {metrics.box.map}")

    # 5. モデルのエクスポート（推論用にonnxなどが欲しければ）
    # model.export(format='onnx')


if __name__ == "__main__":
    # Windowsでのmultiprocessingエラー回避のためのおまじない
    import multiprocessing
    multiprocessing.freeze_support()
    main()
