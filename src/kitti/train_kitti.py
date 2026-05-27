# src/kitti/train_kitti.py

import argparse
import torch
from pathlib import Path
from ultralytics import YOLO, RTDETR


def main():
    parser = argparse.ArgumentParser(
        description="Train ultralytics model on KITTI dataset"
    )
    
    # 学習パラメータ
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs. Default: 100"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training. Adjust based on GPU memory. Default: 16"
    )
    
    # モデル保存先
    parser.add_argument(
        "--save_dir",
        type=str,
        default="src/kitti/runs",
        help="Directory to save trained models. Default: src/kitti/runs"
    )
    
    # 初期モデル
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Pre-trained model to use. Default: yolov8n.pt"
    )
    
    # 画像サイズ
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size for training. Default: 640"
    )
    
    args = parser.parse_args()
    
    device = "cuda"
    
    # ─────────────────────────────────────
    # パス設定
    # ─────────────────────────────────────
    base_dir = Path(__file__).parent.parent.parent  # WindowsNoEditor
    config_path = Path(__file__).parent / "kitti.yaml"
    save_dir = Path(args.save_dir)
    
    print(f"Config file: {config_path}")
    print(f"Save directory: {save_dir.absolute()}")
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # ─────────────────────────────────────
    # モデル読み込み
    # ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("Loading Pre-trained Model")
    print("=" * 60)
    
    try:
        model = YOLO(args.model)
        print(f"✓ Model loaded: {args.model}")
    except Exception as e:
        raise RuntimeError(f"Failed to load model '{args.model}': {e}")
    
    # ─────────────────────────────────────
    # 学習実行
    # ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("Training Configuration")
    print("=" * 60)
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Image size: {args.imgsz}")
    print(f"Device: {device}")
    print("=" * 60 + "\n")
    
    try:
        results = model.train(
            data=str(config_path),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch_size,
            device=device,
            project=str(save_dir),
            name="kitti_model",
            save=True,           # 定期的にチェックポイントを保存
            patience=20,          # 早期停止（20エポック改善なし）
            verbose=True,
            plots=True,           # 学習曲線をプロット
        )
        
        print("\n" + "=" * 60)
        print("✓ Training completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        raise
    
    # ─────────────────────────────────────
    # モデル保存
    # ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("Saving Model")
    print("=" * 60)
    
    # 最終モデルのパス
    final_model_path = save_dir / "kitti_model" / "weights" / "last.pt"
    best_model_path = save_dir / "kitti_model" / "weights" / "best.pt"
    
    if best_model_path.exists():
        print(f"✓ Best model saved at: {best_model_path.absolute()}")
    
    if final_model_path.exists():
        print(f"✓ Final model saved at: {final_model_path.absolute()}")
    
    # カスタム名で保存
    custom_save_path = save_dir / f"kitti_yolov8n_epochs{args.epochs}.pt"
    model.save(str(custom_save_path))
    print(f"✓ Custom saved model at: {custom_save_path.absolute()}")
    
    print("=" * 60)
    print(f"\n📊 Training results:")
    print(f"  - best.pt (best validation): {best_model_path}")
    print(f"  - last.pt (final epoch): {final_model_path}")
    print(f"  - Custom name: {custom_save_path}")
    

if __name__ == "__main__":
    main()