"""
オブジェクト検出モデルを读み込んでアーキテクチャを直接確認
"""
import torch
import sys
from pathlib import Path

# パスの追加
cwd = Path(__file__).parent
sys.path.insert(0, str(cwd.parent.parent))

from src.ObjectDetection.models.Yolov8n import Yolov8nDetector
from src.ObjectDetection.models.Yolov11n import Yolov11nDetector
from src.ObjectDetection.models.Yolov5n import Yolov5nDetector
from src.ObjectDetection.models.rtDETR import RTDETRDetector
from src.ObjectDetection.models.SSD_torch import SSDDetector
from src.ObjectDetection.models.retinanet import RetinanetDetector
from src.ObjectDetection.models.FastRCNN import FasterRCNNDetector
from src.ObjectDetection.models.FCOS import FcosDetector
from src.ObjectDetection.models.trainedYolov8n import Yolov8nTrainedDetector


def count_parameters(model):
    """モデルのパラメータ数を計算"""
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model):
    """学習可能なパラメータ数を計算"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_info(name, detector):
    """モデルの情報を表示"""
    print("\n" + "="*80)
    print(f"【{name}】")
    print("="*80)
    
    try:
        if hasattr(detector, 'model'):
            model = detector.model
        else:
            model = detector
        
        # 基本情報
        print(f"\nModel Type: {type(model).__name__}")
        print(f"Device: {detector.device if hasattr(detector, 'device') else 'Unknown'}")
        
        # パラメータ情報
        total_params = count_parameters(model)
        trainable_params = count_trainable_parameters(model)
        print(f"\nParameters:")
        print(f"  Total:      {total_params:,}")
        print(f"  Trainable:  {trainable_params:,}")
        print(f"  Frozen:     {total_params - trainable_params:,}")
        
        # FLOPs計算（入力サイズに応じて）
        try:
            from fvcore.nn import FlopCounterMode
            
            # ダミー入力の作成
            if "YOLO" in name or "RT-DETR" in name:
                dummy_input = torch.randn(1, 3, 640, 640)
            elif "SSD" in name:
                dummy_input = torch.randn(1, 3, 300, 300)
            elif "Faster" in name or "FCOS" in name:
                dummy_input = torch.randn(1, 3, 800, 800)
            elif "RetinaNet" in name:
                dummy_input = torch.randn(1, 3, 800, 800)
            else:
                dummy_input = torch.randn(1, 3, 640, 640)
            
            if hasattr(detector, 'device'):
                dummy_input = dummy_input.to(detector.device)
            
            with FlopCounterMode(model, display=False) as fcm:
                with torch.no_grad():
                    _ = model(dummy_input)
            
            flops = fcm.total_flops() if hasattr(fcm, 'total_flops') else fcm.flop_dict.get('', 0)
            print(f"\nFLOPs: {flops / 1e9:.2f}G (estimated)")
        except Exception as e:
            print(f"\nFLOPs: Could not compute ({str(e)[:50]}...)")
        
        # モデル構造（簡略版）
        print(f"\nModel Architecture:")
        print("-"*80)
        
        if "YOLO" in name or "RT-DETR" in name:
            # YOLOモデルの場合
            print(model)
        elif "SSD" in name or "Faster" in name or "RetinaNet" in name:
            # PyTorchビルトインモデルの場合
            print(f"  Backbone: {model.backbone}")
            if hasattr(model, 'head'):
                print(f"  Head: {type(model.head).__name__}")
            if hasattr(model, 'box_predictor'):
                print(f"  Box Predictor: {type(model.box_predictor).__name__}")
            if hasattr(model, "neck"):
                print(f"  Neck: {type(model.neck).__name__}")
        else:
            # その他
            print(str(model)[:500] + "..." if len(str(model)) > 500 else str(model))
        
        # モデルの層の詳細
        # print(f"\n\nLayer Information:")
        # print("-"*80)
        
        # layer_count = 0
        # param_count_by_type = {}
        
        # for name_param, param in model.named_parameters():
        #     layer_count += 1
        #     param_type = type(param).__name__
        #     param_size = param.numel()
            
        #     if param_type not in param_count_by_type:
        #         param_count_by_type[param_type] = {'count': 0, 'params': 0}
            
        #     param_count_by_type[param_type]['count'] += 1
        #     param_count_by_type[param_type]['params'] += param_size
        
        # print(f"Total Layers: {layer_count}")
        # print("\nParameters by Type:")
        # for param_type, info in sorted(param_count_by_type.items(), key=lambda x: x[1]['params'], reverse=True):
        #     print(f"  {param_type}: {info['count']:>4} layers, {info['params']:>12,} params")
        
    except Exception as e:
        print(f"Error: {e}")

    

def main():
    print("\n" + "="*100)
    print("Object Detection Models - Direct Architecture Analysis")
    print("="*100)
    
    detectors_to_load = [
        ("YOLOv8n", Yolov8nDetector),
        ("YOLOv11n", Yolov11nDetector),
        ("YOLOv5n", Yolov5nDetector),
        ("RT-DETR", RTDETRDetector),
        # ("SSD300", SSDDetector),
        # ("RetinaNet", RetinanetDetector),
        # ("Faster R-CNN", FasterRCNNDetector),
        # ("FCOS", FcosDetector),
        # ("Trained YOLOv8n", Yolov8nTrainedDetector),
    ]
    
    detectors_info = {}
    
    print("\nLoading models...")
    for model_name, detector_class in detectors_to_load:
        try:
            print(f"  Loading {model_name}...", end=" ")
            detector = detector_class()
            
            model = detector.model if hasattr(detector, 'model') else detector
            total_params = count_parameters(model)
            device = detector.device if hasattr(detector, 'device') else 'Unknown'
            
            detectors_info[model_name] = (detector, total_params, device)
            print(f"✓ ({total_params:,} params)")
            
        except Exception as e:
            print(f"✗ Error: {str(e)[:60]}")

    
    # 各モデルの詳細情報表示
    for model_name, (detector, _, _) in detectors_info.items():
        try:
            print_model_info(model_name, detector)
        except Exception as e:
            print(f"\nError displaying {model_name}: {e}")
    
    # 統計情報
    print("\n" + "="*100)
    print("Summary Statistics")
    print("="*100)
    
    total_all_params = sum(info[1] for info in detectors_info.values())
    avg_params = total_all_params / len(detectors_info) if detectors_info else 0
    
    print(f"\nTotal Models Loaded: {len(detectors_info)}")
    print(f"Total Parameters (all models): {total_all_params:,}")
    print(f"Average Parameters per Model: {avg_params:,.0f}")
    
    # パラメータ数でソート
    print("\nModels Ranked by Parameters (ascending):")
    for i, (name, (_, params, _)) in enumerate(sorted(detectors_info.items(), key=lambda x: x[1][1]), 1):
        print(f"  {i}. {name:<25} {params:>12,} params")


if __name__ == "__main__":
    # GPU使用可能か確認
    print(f"\nGPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    main()