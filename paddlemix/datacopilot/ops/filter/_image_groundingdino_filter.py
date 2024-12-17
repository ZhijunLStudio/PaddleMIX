# from tqdm import tqdm
# from typing import Optional, Dict, Tuple
# from PIL import Image
# import os
# import paddle
# import paddle.nn.functional as F
# from dataclasses import dataclass

# from ....models.groundingdino.modeling import GroundingDinoModel
# from ....processors.groundingdino_processing import GroundingDinoProcessor
# from ...core import T, MMDataset, register

# @dataclass
# class GroundingDinoConfig:
#     """Configuration for Grounding DINO filtering."""
#     model_name: str = "GroundingDino/groundingdino-swint-ogc"
#     box_threshold: float = 0.3
#     min_objects: int = 1
#     max_objects: int = 4
#     min_aspect_ratio: float = 0.05  # 5%
#     max_aspect_ratio: float = 0.95  # 95%
#     top_percentage: float = 0.3  # Percentage for top detections


# def dino_process_single_image(
#     image_path: str,
#     prompt: str,
#     model: GroundingDinoModel,
#     processor: GroundingDinoProcessor,
#     config: GroundingDinoConfig
# ) -> Optional[Dict]:
#     """Process a single image with Grounding DINO model."""
#     try:
#         # 加载图像
#         if not os.path.isfile(image_path):
#             return None
#         image = Image.open(image_path).convert("RGB")
        
#         # 处理图像
#         image_tensor, mask, tokenized_out = processor(images=image, text=prompt)
#         if image_tensor is None or image_tensor.shape[0] == 0:
#             return None
#         print(f"Image tensors shape: {image_tensor.shape}")
#         print(f"Masks shape: {mask.shape}")

#         # 获取模型预测
#         with paddle.no_grad():
#             outputs = model(
#                 image_tensor,
#                 mask,
#                 input_ids=tokenized_out["input_ids"],
#                 attention_mask=tokenized_out["attention_mask"],
#                 text_self_attention_masks=tokenized_out["text_self_attention_masks"],
#                 position_ids=tokenized_out["position_ids"],
#             )

#         # 处理输出
#         logits = F.sigmoid(outputs["pred_logits"])[0]  # [nq, 256]
#         boxes = outputs["pred_boxes"][0]  # [nq, 4]

#         # 计算置信度分数
#         scores = logits.max(axis=1)
        
#         # 应用 box_threshold 过滤低置信度框
#         high_confidence_mask = scores >= config.box_threshold
#         high_confidence_indices = paddle.nonzero(high_confidence_mask).flatten()
        
#         if len(high_confidence_indices) == 0:
#             print("No boxes passed the confidence threshold")
#             return None
            
#         scores = scores[high_confidence_indices]
#         boxes = boxes[high_confidence_indices]

#         # 取前配置的百分比
#         if len(scores) > 0:
#             num_to_keep = max(1, int(config.top_percentage * len(scores)))
#             sorted_indices = paddle.argsort(scores, descending=True)[:num_to_keep]
#             scores = scores[sorted_indices]
#             boxes = boxes[sorted_indices]

#         # 计算框的属性
#         widths = boxes[:, 2] - boxes[:, 0]  # 修正宽度计算
#         heights = boxes[:, 3] - boxes[:, 1]  # 修正高度计算
#         aspect_ratios = widths / heights


#         return {
#             "num_objects": len(boxes),
#             "scores": scores,
#             "aspect_ratios": aspect_ratios
#         }
#     except Exception as e:
#         print(f"Error processing image {image_path}: {e}")
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

#     # 初始化模型和处理器
#     processor = GroundingDinoProcessor.from_pretrained(config.model_name)
#     model = GroundingDinoModel.from_pretrained(config.model_name)
#     model.eval()

#     filtered_items = []

#     # 使用 tqdm 显示进度条
#     for item in tqdm(dataset, desc="Filtering images"):
#         image_path = item.get('image')
#         if not image_path or not os.path.exists(image_path):
#             continue

#         # 处理图像并获取检测结果
#         detection_results = dino_process_single_image(
#             image_path=image_path,
#             prompt=prompt,
#             model=model,
#             processor=processor,
#             config=config
#         )
        
#         if detection_results is None:
#             continue

#         # 应用过滤条件
#         if (
#             config.min_objects <= detection_results["num_objects"] <= config.max_objects and
#             all(config.min_aspect_ratio <= ar <= config.max_aspect_ratio 
#                 for ar in detection_results["aspect_ratios"])
#         ):
#             filtered_items.append(item)

#     return MMDataset(filtered_items)



# import os
# import paddle
# from tqdm import tqdm
# from typing import Optional, List, Dict
# from dataclasses import dataclass
# from PIL import Image
# from ....models.groundingdino.modeling import GroundingDinoModel
# from ....processors.groundingdino_processing import GroundingDinoProcessor
# from ...core import MMDataset, register
# import paddle.nn.functional as F

# @dataclass
# class GroundingDinoConfig:
#     model_name: str = "GroundingDino/groundingdino-swint-ogc"
#     box_threshold: float = 0.3
#     min_objects: int = 1
#     max_objects: int = 4
#     min_aspect_ratio: float = 0.05
#     max_aspect_ratio: float = 0.95
#     top_percentage: float = 0.3
#     batch_size: int = 8  # Batch size for processing multiple images




# def dino_process_batch(
#     image_paths: List[str],
#     prompt: str,
#     model: GroundingDinoModel,
#     processor: GroundingDinoProcessor,
#     config: GroundingDinoConfig
# ) -> List[Optional[Dict]]:
#     """Process a batch of images with Grounding DINO model."""
#     # try:
#         # Open and resize all images to a consistent size (e.g., 1024x1024)
#     target_size = (224, 224)  # Ensure this matches model expectations
#     images = [Image.open(path).convert("RGB").resize(target_size) for path in image_paths if os.path.isfile(path)]
#     for img in images:
#         print(f"Image size: {img.size}")  # Should print (1024, 1024) for example

    
#     # Preprocess images and text
#     image_tensor, mask, tokenized_out = processor(images=images, text=prompt)
#     print(f"Image tensor shape: {image_tensor.shape}")
#     print(f"Mask shape: {mask.shape}")
#     print(f"Tokenized output shape: {tokenized_out['input_ids'].shape}")

    
#     if image_tensor is None or image_tensor.shape[0] == 0:
#         return [None] * len(image_paths)

#     with paddle.no_grad():
#         outputs = model(
#             image_tensor,
#             mask,
#             input_ids=tokenized_out["input_ids"],
#             attention_mask=tokenized_out["attention_mask"],
#             text_self_attention_masks=tokenized_out["text_self_attention_masks"],
#             position_ids=tokenized_out["position_ids"],
#         )

#     logits = F.sigmoid(outputs["pred_logits"])  # [batch_size, nq, 256]
#     boxes = outputs["pred_boxes"]  # [batch_size, nq, 4]

#     results = []
#     for i in range(len(image_paths)):
#         scores = logits[i].max(axis=1)
#         high_confidence_mask = scores >= config.box_threshold
#         high_confidence_indices = paddle.nonzero(high_confidence_mask).flatten()

#         if len(high_confidence_indices) == 0:
#             results.append(None)
#             continue

#         scores = scores[high_confidence_indices]
#         boxes = boxes[i][high_confidence_indices]

#         num_to_keep = max(1, int(config.top_percentage * len(scores)))
#         sorted_indices = paddle.argsort(scores, descending=True)[:num_to_keep]
#         scores = scores[sorted_indices]
#         boxes = boxes[sorted_indices]

#         widths = boxes[:, 2] - boxes[:, 0]
#         heights = boxes[:, 3] - boxes[:, 1]
#         aspect_ratios = widths / heights

#         results.append({
#             "num_objects": len(boxes),
#             "scores": scores,
#             "aspect_ratios": aspect_ratios
#         })

#     return results
#     # except Exception as e:
#     #     print(f"Error processing batch: {e}")
#     #     return [None] * len(image_paths)





# @register()
# def filter_by_dino(
#     dataset: MMDataset,
#     prompt: str,
#     config: Optional[GroundingDinoConfig] = None
# ) -> MMDataset:
#     """Filter dataset using Grounding DINO object detection (batch version)."""
#     if config is None:
#         config = GroundingDinoConfig()

#     processor = GroundingDinoProcessor.from_pretrained(config.model_name)
#     model = GroundingDinoModel.from_pretrained(config.model_name)
#     model.eval()

#     filtered_items = []
#     batch_size = config.batch_size

#     all_items = []
#     for item in dataset:
#         image_path = item.get('image')
#         if image_path and os.path.exists(image_path):
#             all_items.append(item)

#     # Process in batches
#     for i in tqdm(range(0, len(all_items), batch_size), desc="Filtering images"):
#         batch = all_items[i:i + batch_size]
#         image_paths = [item.get('image') for item in batch]
#         # batch_prompts = [prompt] * len(batch)
#         # print(batch_prompts)

#         detection_results = dino_process_batch(
#             image_paths=image_paths,
#             prompt=prompt,
#             model=model,
#             processor=processor,
#             config=config
#         )

#         for item, result in zip(batch, detection_results):
#             if result and config.min_objects <= result["num_objects"] <= config.max_objects and \
#                 all(config.min_aspect_ratio <= ar <= config.max_aspect_ratio for ar in result["aspect_ratios"]):
#                 filtered_items.append(item)

#     return MMDataset(filtered_items)







from tqdm import tqdm
from typing import Optional, Dict, Tuple
from PIL import Image, ImageDraw
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
    save_results: bool = True  # Flag to control saving results
    output_dir: Optional[str] = "dino_output"  # Directory to save images with boxes


def draw_boxes_on_image(image: Image, boxes: paddle.Tensor, output_dir: str, image_name: str) -> None:
    """Draw bounding boxes on the image and save it to the output directory."""
    draw = ImageDraw.Draw(image)
    
    for box in boxes:
        # Convert box from tensor to center and size (center_x, center_y, width, height)
        center_x, center_y, width, height = box.numpy()
        
        # Convert from center format to corner format
        x_min = center_x - width / 2
        y_min = center_y - height / 2
        x_max = center_x + width / 2
        y_max = center_y + height / 2
        
        # Ensure coordinates are valid (x_min <= x_max and y_min <= y_max)
        if x_min > x_max:
            x_min, x_max = x_max, x_min
        if y_min > y_max:
            y_min, y_max = y_max, y_min
        
        # Convert to pixel coordinates
        W, H = image.size  # Image width and height
        x_min, y_min, x_max, y_max = int(x_min * W), int(y_min * H), int(x_max * W), int(y_max * H)
        
        # Draw the rectangle (bounding box)
        draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)
    
    # Save the image with bounding boxes
    save_path = os.path.join(output_dir, f"{image_name}_with_boxes.jpg")
    image.save(save_path)
    print(f"Saved image with boxes to: {save_path}")


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
        print(f"Image tensors shape: {image_tensor.shape}")
        print(f"Masks shape: {mask.shape}")

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
        
        # 应用 box_threshold 过滤低置信度框
        high_confidence_mask = scores >= config.box_threshold
        high_confidence_indices = paddle.nonzero(high_confidence_mask).flatten()
        
        if len(high_confidence_indices) == 0:
            print("No boxes passed the confidence threshold")
            return None
            
        scores = scores[high_confidence_indices]
        boxes = boxes[high_confidence_indices]
        print(len(scores))

        # 取前配置的百分比
        # if len(scores) > 0:
        #     num_to_keep = max(1, int(config.top_percentage * len(scores)))
        #     sorted_indices = paddle.argsort(scores, descending=True)[:num_to_keep]
        #     scores = scores[sorted_indices]
        #     boxes = boxes[sorted_indices]

        # 计算框的属性
        widths = boxes[:, 2]  # 宽度
        heights = boxes[:, 3]  # 高度
        aspect_ratios = widths / heights

        # 保存图像带框
        if config.save_results and config.output_dir:
            image_name = os.path.basename(image_path).split('.')[0]
            draw_boxes_on_image(image, boxes, config.output_dir, image_name)

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

    # 如果存储文件夹不存在
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

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