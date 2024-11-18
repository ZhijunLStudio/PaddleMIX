from tqdm import tqdm
from typing import Optional, Dict, Tuple
from PIL import Image
import os
import paddle
import paddle.nn.functional as F
from dataclasses import dataclass

from ....models.groundingdino.modeling import GroundingDinoModel
from ....processors.groundingdino_processing import GroundingDinoProcessor
from ...core import T, MMDataset, register

@dataclass
class GroundingDinoConfig:
    """Configuration for Grounding DINO filtering."""
    model_name: str = "GroundingDino/groundingdino-swint-ogc"
    box_threshold: float = 0.3
    min_objects: int = 1
    max_objects: int = 4
    min_aspect_ratio: float = 0.05  # 5%
    max_aspect_ratio: float = 0.95  # 95%
    top_percentage: float = 0.3  # Percentage for top detections


def dino_process_single_image(
    image_path: str,
    prompt: str,
    model: GroundingDinoModel,
    processor: GroundingDinoProcessor,
    config: GroundingDinoConfig
) -> Optional[Dict]:
    """Process a single image with Grounding DINO model."""
    try:
        # 加载图像
        if not os.path.isfile(image_path):
            return None
        image = Image.open(image_path).convert("RGB")
        
        # 处理图像
        image_tensor, mask, tokenized_out = processor(images=image, text=prompt)
        if image_tensor is None or image_tensor.shape[0] == 0:
            return None

        # 获取模型预测
        with paddle.no_grad():
            outputs = model(
                image_tensor,
                mask,
                input_ids=tokenized_out["input_ids"],
                attention_mask=tokenized_out["attention_mask"],
                text_self_attention_masks=tokenized_out["text_self_attention_masks"],
                position_ids=tokenized_out["position_ids"],
            )

        # 处理输出
        logits = F.sigmoid(outputs["pred_logits"])[0]  # [nq, 256]
        boxes = outputs["pred_boxes"][0]  # [nq, 4]

        # 计算置信度分数
        scores = logits.max(axis=1)
        
        # 输出所有分数，帮助调试
        print(f"Scores before filtering: {scores}")

        # 应用 box_threshold 过滤低置信度框
        high_confidence_mask = scores >= config.box_threshold
        high_confidence_indices = paddle.nonzero(high_confidence_mask).flatten()
        
        if len(high_confidence_indices) == 0:
            print("No boxes passed the confidence threshold")
            return None
            
        scores = scores[high_confidence_indices]
        boxes = boxes[high_confidence_indices]

        # 输出高置信度框的数量，帮助调试
        print(f"Number of boxes after applying box_threshold: {len(scores)}")

        # 取前配置的百分比
        if len(scores) > 0:
            num_to_keep = max(1, int(config.top_percentage * len(scores)))
            sorted_indices = paddle.argsort(scores, descending=True)[:num_to_keep]
            scores = scores[sorted_indices]
            boxes = boxes[sorted_indices]

        # 计算框的属性
        widths = boxes[:, 2] - boxes[:, 0]  # 修正宽度计算
        heights = boxes[:, 3] - boxes[:, 1]  # 修正高度计算
        aspect_ratios = widths / heights


        return {
            "num_objects": len(boxes),
            "scores": scores,
            "aspect_ratios": aspect_ratios
        }
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None


@register()
def filter_by_dino(
    dataset: MMDataset,
    prompt: str,
    config: Optional[GroundingDinoConfig] = None,
) -> MMDataset:
    """Filter dataset using Grounding DINO object detection."""
    if config is None:
        config = GroundingDinoConfig()

    # 初始化模型和处理器
    processor = GroundingDinoProcessor.from_pretrained(config.model_name)
    model = GroundingDinoModel.from_pretrained(config.model_name)
    model.eval()

    filtered_items = []

    # 使用 tqdm 显示进度条
    for item in tqdm(dataset, desc="Filtering images"):
        image_path = item.get('image')
        if not image_path or not os.path.exists(image_path):
            continue

        # 处理图像并获取检测结果
        detection_results = dino_process_single_image(
            image_path=image_path,
            prompt=prompt,
            model=model,
            processor=processor,
            config=config
        )
        
        if detection_results is None:
            continue

        # 应用过滤条件
        if (
            config.min_objects <= detection_results["num_objects"] <= config.max_objects and
            all(config.min_aspect_ratio <= ar <= config.max_aspect_ratio 
                for ar in detection_results["aspect_ratios"])
        ):
            filtered_items.append(item)

    return MMDataset(filtered_items)
