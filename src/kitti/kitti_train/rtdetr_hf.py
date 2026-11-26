import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_convert
from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection


class RTDETRv2Wrapper(nn.Module):
    """HF RT-DETRv2をtorchvision検出モデルと同じ入出力にするラッパー。

    train: forward(images, targets) -> {"loss": ...}
    eval : forward(images) -> [{"boxes", "scores", "labels"}, ...]
    ラベルはtorchvision流儀（0=背景、1始まり）で受け取り、内部で0始まりに変換する。
    """

    def __init__(self, num_classes, ckpt="PekingU/rtdetr_v2_r18vd", size=640, score_thresh=0.01):
        super().__init__()
        self.model = RTDetrV2ForObjectDetection.from_pretrained(
            ckpt, num_labels=num_classes - 1, ignore_mismatched_sizes=True
        )
        self.processor = RTDetrImageProcessor.from_pretrained(ckpt)
        self.size = size
        self.score_thresh = score_thresh

    def _to_batch(self, images):
        # 入力は[0,1]スケール済み。RT-DETRは正規化なしで固定サイズにリサイズするだけ
        return torch.stack([
            F.interpolate(img[None], size=(self.size, self.size), mode="bilinear", align_corners=False)[0]
            for img in images
        ])

    def forward(self, images, targets=None):
        pixel_values = self._to_batch(images)

        if self.training:
            labels = []
            for img, t in zip(images, targets):
                h, w = img.shape[-2:]
                scale = torch.tensor([w, h, w, h], dtype=torch.float32, device=img.device)
                boxes = box_convert(t["boxes"].as_subclass(torch.Tensor), "xyxy", "cxcywh") / scale
                labels.append({"class_labels": t["labels"] - 1, "boxes": boxes})
            out = self.model(pixel_values=pixel_values, labels=labels)
            return {"loss": out.loss}

        out = self.model(pixel_values=pixel_values)
        sizes = torch.tensor([img.shape[-2:] for img in images], device=pixel_values.device)
        results = self.processor.post_process_object_detection(
            out, threshold=self.score_thresh, target_sizes=sizes
        )
        for r in results:
            r["labels"] = r["labels"] + 1
        return results