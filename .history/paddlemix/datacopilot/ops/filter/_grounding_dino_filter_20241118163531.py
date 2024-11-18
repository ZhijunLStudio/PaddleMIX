# from typing import Optional, List, Dict, Tuple
# from dataclasses import dataclass
# import paddle
# import paddle.nn.functional as F
# from PIL import Image
# import requests
# import os
# import logging

# from ....models.groundingdino.modeling import GroundingDinoModel
# from ....processors.groundingdino_processing import GroundingDinoProcessor
# from ...core import T, MMDataset, register

# # 配置日志
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# @dataclass
# class GroundingDinoConfig:
#     """Configuration for Grounding DINO filtering."""
#     model_name: str = "GroundingDino/groundingdino-swint-ogc"
#     box_threshold: float = 0.3
#     text_threshold: float = 0.25
#     min_objects: int = 1
#     max_objects: int = 10
#     min_aspect_ratio: float = 0.2
#     max_aspect_ratio: float = 5.0
#     min_box_area: float = 0.01  # 最小框面积（相对于图像面积的比例）
#     max_box_area: float = 0.9   # 最大框面积

# def validate_boxes(boxes: paddle.Tensor, image_size: Tuple[int, int]) -> bool:
#     """验证检测框的有效性
    
#     Args:
#         boxes: 检测框张量 [N, 4]
#         image_size: 图像尺寸 (height, width)
        
#     Returns:
#         bool: 检测框是否有效
#     """
#     if boxes is None or boxes.shape[0] == 0:
#         logger.warning("No valid boxes detected")
#         return False
        
#     try:
#         # 检查数值是否在合理范围内
#         if paddle.any(boxes < 0) or paddle.any(boxes > 1):
#             logger.warning("Boxes coordinates out of range [0, 1]")
#             return False
            
#         # 检查框的大小是否合理
#         widths = boxes[:, 2]
#         heights = boxes[:, 3]
#         if paddle.any(widths <= 0) or paddle.any(heights <= 0):
#             logger.warning("Invalid box dimensions detected")
#             return False
            
#         return True
#     except Exception as e:
#         logger.error(f"Error validating boxes: {e}")
#         return False

# def process_single_image(
#     image_path: str,
#     prompt: str,
#     model: GroundingDinoModel,
#     processor: GroundingDinoProcessor,
#     config: GroundingDinoConfig
# ) -> Optional[Dict]:
#     """Process a single image with Grounding DINO model."""
#     try:
#         # 加载图像
#         if os.path.isfile(image_path):
#             image = Image.open(image_path).convert("RGB")
#         else:
#             logger.warning(f"Image file not found: {image_path}")
#             return None
            
#         # 记录图像尺寸
#         image_size = image.size
#         logger.info(f"Processing image {image_path} with size {image_size}")
            
#         # 处理图像
#         image_tensor, mask, tokenized_out = processor(images=image, text=prompt)
        
#         if image_tensor is None or image_tensor.shape[0] == 0:
#             logger.warning("Failed to process image: empty tensor")
#             return None

#         # 获取模型预测
#         with paddle.no_grad():
#             try:
#                 outputs = model(
#                     image_tensor,
#                     mask,
#                     input_ids=tokenized_out["input_ids"],
#                     attention_mask=tokenized_out["attention_mask"],
#                     text_self_attention_masks=tokenized_out["text_self_attention_masks"],
#                     position_ids=tokenized_out["position_ids"],
#                 )
#             except Exception as e:
#                 logger.error(f"Model inference failed: {e}")
#                 return None

#         # 处理输出
#         logits = F.sigmoid(outputs["pred_logits"])[0]  # [nq, 256]
#         boxes = outputs["pred_boxes"][0]  # [nq, 4]
        
#         # 验证检测框
#         if not validate_boxes(boxes, image_size):
#             return None

#         # 应用置信度过滤
#         logits_filt = logits.clone()
#         boxes_filt = boxes.clone()
#         filt_mask = logits_filt.max(axis=1) > config.box_threshold
        
#         if not paddle.any(filt_mask):
#             logger.warning("No boxes passed confidence threshold")
#             return None
            
#         logits_filt = logits_filt[filt_mask]
#         boxes_filt = boxes_filt[filt_mask]

#         # 计算指标
#         scores = logits_filt.max(axis=1)
#         widths = boxes_filt[:, 2]
#         heights = boxes_filt[:, 3]
        
#         # 计算框面积
#         areas = widths * heights
#         aspect_ratios = widths / heights

#         # 记录检测结果
#         logger.info(f"Detected {len(boxes_filt)} objects with scores: {scores.numpy().tolist()}")

#         return {
#             "boxes": boxes_filt,
#             "scores": scores,
#             "aspect_ratios": aspect_ratios,
#             "areas": areas,
#             "num_objects": len(boxes_filt)
#         }
        
#     except Exception as e:
#         logger.error(f"Error processing image {image_path}: {str(e)}")
#         return None

# @register()
# def filter_by_dino(
#     dataset: MMDataset,
#     prompt: str,
#     config: Optional[GroundingDinoConfig] = None,
# ) -> MMDataset:
#     """Filter dataset using Grounding DINO object detection."""
#     if config is None:
#         config = GroundingDinoConfig()

#     logger.info(f"Initializing Grounding DINO with config: {config}")
    
#     # 初始化模型和处理器
#     processor = GroundingDinoProcessor.from_pretrained(config.model_name)
#     model = GroundingDinoModel.from_pretrained(config.model_name)
#     model.eval()

#     filtered_items = []
#     total_processed = 0
#     total_failed = 0

#     for item in dataset:
#         total_processed += 1
#         image_path = item.get('image')
        
#         if not image_path or not os.path.exists(image_path):
#             total_failed += 1
#             continue

#         # 处理图像并获取检测结果
#         detection_results = process_single_image(
#             image_path=image_path,
#             prompt=prompt,
#             model=model,
#             processor=processor,
#             config=config
#         )
        
#         if detection_results is None:
#             total_failed += 1
#             continue

#         # 应用过滤条件
#         try:
#             valid_item = (
#                 config.min_objects <= detection_results["num_objects"] <= config.max_objects and
#                 all(config.min_aspect_ratio <= ar <= config.max_aspect_ratio 
#                     for ar in detection_results["aspect_ratios"]) and
#                 all(config.min_box_area <= area <= config.max_box_area 
#                     for area in detection_results["areas"]) and
#                 all(score >= config.box_threshold 
#                     for score in detection_results["scores"])
#             )
            
#             if valid_item:
#                 filtered_items.append(item)
                
#         except Exception as e:
#             logger.error(f"Error applying filters: {e}")
#             total_failed += 1
#             continue

#     # 记录处理统计信息
#     logger.info(f"Processing complete: {total_processed} images processed")
#     logger.info(f"Failed: {total_failed} images")
#     logger.info(f"Filtered dataset size: {len(filtered_items)}")

#     return MMDataset(filtered_items)



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


def diprocess_single_image(
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
        sorted_indices = paddle.argsort(scores, descending=True)

        # 应用 box_threshold 过滤低置信度框
        high_confidence_indices = paddle.nonzero(scores >= config.box_threshold).flatten()
        scores = scores[high_confidence_indices]
        boxes = boxes[high_confidence_indices]

        # 取前配置的百分比
        top_percentage = int(config.top_percentage * len(scores))
        top_indices = sorted_indices[:top_percentage]
        scores = scores[top_indices]
        boxes = boxes[top_indices]

        # 计算框的属性
        widths = boxes[:, 2]
        heights = boxes[:, 3]
        aspect_ratios = widths / heights

        # 输出调试信息
        print("len(boxes):", len(boxes))
        print("scores:", scores)
        print("aspect_ratios:", aspect_ratios)

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
        detection_results = process_single_image(
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
