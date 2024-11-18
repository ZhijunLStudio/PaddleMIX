from typing import Optional, List, Dict
from dataclasses import dataclass
import paddle
import paddle.nn.functional as F
from PIL import Image
import requests
import os

from ....models.groundingdino.modeling import GroundingDinoModel
from ....processors.groundingdino_processing import GroundingDinoProcessor
from ...core import T, MMDataset, register

@dataclass
class GroundingDinoConfig:
    """Configuration for Grounding DINO filtering."""
    model_name: str = "GroundingDino/groundingdino-swint-ogc"
    box_threshold: float = 0.3
    text_threshold: float = 0.25
    min_objects: int = 1
    max_objects: int = 10
    min_aspect_ratio: float = 0.2
    max_aspect_ratio: float = 5.0

def process_single_image(
    image_path: str,
    prompt: str,
    model: GroundingDinoModel,
    processor: GroundingDinoProcessor,
    config: GroundingDinoConfig
) -> Optional[Dict]:
    """Process a single image with Grounding DINO model.
    
    Args:
        image_path: Path to the image file
        prompt: Text prompt for object detection
        model: Initialized GroundingDinoModel
        processor: Initialized GroundingDinoProcessor
        config: Configuration for filtering
        
    Returns:
        Optional[Dict]: Detection results or None if processing failed
    """
    try:
        # Load image
        if os.path.isfile(image_path):
            image = Image.open(image_path).convert("RGB")
        else:
            image = Image.open(requests.get(image_path, stream=True).raw).convert("RGB")
            
        # Process image
        image_tensor, mask, tokenized_out = processor(images=image, text=prompt)

        # Get model predictions
        with paddle.no_grad():
            outputs = model(
                image_tensor,
                mask,
                input_ids=tokenized_out["input_ids"],
                attention_mask=tokenized_out["attention_mask"],
                text_self_attention_masks=tokenized_out["text_self_attention_masks"],
                position_ids=tokenized_out["position_ids"],
            )

        # Process outputs
        logits = F.sigmoid(outputs["pred_logits"])[0]
        boxes = outputs["pred_boxes"][0]

        # Filter by confidence threshold
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(axis=1) > config.box_threshold
        logits_filt = logits_filt[filt_mask]
        boxes_filt = boxes_filt[filt_mask]

        # Calculate metrics
        scores = logits_filt.max(axis=1)
        widths = boxes_filt[:, 2]
        heights = boxes_filt[:, 3]
        aspect_ratios = widths / heights

        return {
            "boxes": boxes_filt,
            "scores": scores,
            "aspect_ratios": aspect_ratios,
            "num_objects": len(boxes_filt)
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
    """Filter dataset using Grounding DINO object detection.
    
    Args:
        dataset: Input dataset
        prompt: Text prompt for object detection
        config: Configuration for filtering
        
    Returns:
        MMDataset: Filtered dataset
    """
    if config is None:
        config = GroundingDinoConfig()

    # Initialize model and processor
    processor = GroundingDinoProcessor.from_pretrained(config.model_name)
    model = GroundingDinoModel.from_pretrained(config.model_name)
    model.eval()

    filtered_items = []

    for item in dataset:
        image_path = item.get('image')
        if not image_path or not os.path.exists(image_path):
            continue

        # Process image and get detection results
        detection_results = process_single_image(
            image_path=image_path,
            prompt=prompt,
            model=model,
            processor=processor,
            config=config
        )
        
        if detection_results is None:
            continue

        # Apply filtering criteria
        if (config.min_objects <= detection_results["num_objects"] <= config.max_objects and
            all(config.min_aspect_ratio <= ar <= config.max_aspect_ratio 
                for ar in detection_results["aspect_ratios"]) and
            all(score >= config.box_threshold for score in detection_results["scores"])):
            filtered_items.append(item)

    return MMDataset(filtered_items)