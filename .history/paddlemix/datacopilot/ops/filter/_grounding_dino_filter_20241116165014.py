import os
from typing import Optional, List, Dict, Tuple
import paddle
import paddle.nn.functional as F
from PIL import Image
import requests
from dataclasses import dataclass
from ....models.groundingdino.modeling import GroundingDinoModel
from ....processors.groundingdino_processing import GroundingDinoProcessor
from ...core import T, MMDataset, register

@dataclass
class GroundingDinoConfig:
    model_name: str = "GroundingDino/groundingdino-swint-ogc"
    box_threshold: float = 0.3
    text_threshold: float = 0.25
    min_objects: int = 1
    max_objects: int = 10
    min_aspect_ratio: float = 0.2
    max_aspect_ratio: float = 5.0

class GroundingDinoDetector:
    def __init__(self, config: GroundingDinoConfig):
        self.config = config
        self.processor = GroundingDinoProcessor.from_pretrained(config.model_name)
        self.model = GroundingDinoModel.from_pretrained(config.model_name)
        self.model.eval()

    def load_image(self, image_path: str) -> Optional[Image.Image]:
        try:
            if os.path.isfile(image_path):
                return Image.open(image_path).convert("RGB")
            else:
                return Image.open(requests.get(image_path, stream=True).raw).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None

    def process_image(self, image: Image.Image, prompt: str) -> Tuple[paddle.Tensor, paddle.Tensor, Dict]:
        return self.processor(images=image, text=prompt)

    def detect_objects(self, image_path: str, prompt: str) -> Optional[Dict]:
        """Detect objects in the image using Grounding DINO model."""
        image = self.load_image(image_path)
        if image is None:
            return None

        image_tensor, mask, tokenized_out = self.process_image(image, prompt)

        with paddle.no_grad():
            outputs = self.model(
                image_tensor,
                mask,
                input_ids=tokenized_out["input_ids"],
                attention_mask=tokenized_out["attention_mask"],
                text_self_attention_masks=tokenized_out["text_self_attention_masks"],
                position_ids=tokenized_out["position_ids"],
            )

        logits = F.sigmoid(outputs["pred_logits"])[0]
        boxes = outputs["pred_boxes"][0]

        # Filter by confidence threshold
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(axis=1) > self.config.box_threshold
        logits_filt = logits_filt[filt_mask]
        boxes_filt = boxes_filt[filt_mask]

        # Calculate scores and aspect ratios
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

@register()
def filter_by_dino(
    dataset: MMDataset,
    prompt: str,
    config: Optional[GroundingDinoConfig] = None,
    device: str = "cpu"
) -> MMDataset:
    """Filter dataset using Grounding DINO object detection.
    
    Args:
        dataset (MMDataset): Input dataset
        prompt (str): Text prompt for object detection
        config (GroundingDinoConfig, optional): Configuration for filtering
        device (str): Device to run the model on
        
    Returns:
        MMDataset: Filtered dataset
    """
    if config is None:
        config = GroundingDinoConfig()

    detector = GroundingDinoDetector(config)
    filtered_items = []

    for item in dataset:
        image_path = item.get('image')
        if not image_path or not os.path.exists(image_path):
            continue

        detection_results = detector.detect_objects(image_path, prompt)
        if detection_results is None:
            continue

        # Apply filtering criteria
        if (config.min_objects <= detection_results["num_objects"] <= config.max_objects and
            all(config.min_aspect_ratio <= ar <= config.max_aspect_ratio 
                for ar in detection_results["aspect_ratios"]) and
            all(score >= config.box_threshold for score in detection_results["scores"])):
            filtered_items.append(item)

    return MMDataset(filtered_items)

# Usage example
if __name__ == "__main__":
    # Load dataset
    dataset = MMDataset.from_json('./my.json')
    print(f"Original dataset size: {len(dataset)}")

    # Configure Grounding DINO parameters
    config = GroundingDinoConfig(
        box_threshold=0.35,
        text_threshold=0.25,
        min_objects=1,
        max_objects=5,
        min_aspect_ratio=0.5,
        max_aspect_ratio=2.0
    )

    # Chain processing operations
    processed_dataset = (
        dataset
        .filter_by_dino(
            prompt="person, car, building", 
            config=config
        )
        .nonempty()
    )

    print(f"Processed dataset size: {len(processed_dataset)}")
    
    # Export processed dataset
    processed_dataset.export_json('filtered_output.json')