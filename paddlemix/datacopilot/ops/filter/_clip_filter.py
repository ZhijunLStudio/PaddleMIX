import os
from tqdm import tqdm
from typing import Optional
from dataclasses import dataclass

import paddle
from paddlemix.processors.clip_processing import CLIPImageProcessor, CLIPTextProcessor, CLIPProcessor
from paddlemix.models.clip.clip_model import CLIP
from paddlemix.processors.tokenizer import SimpleTokenizer
from ...core import T, MMDataset, register

@dataclass
class CLIPFilterConfig:
    """用于CLIP相似度过滤的配置。"""
    model_name: str = "paddlemix/CLIP/CLIP-ViT-L-14-laion2B-s32B-b82K"
    threshold: float = 0.25  # 相似度阈值


def clip_process_single_image(
    image_path: str,
    text_prompt: str,
    model: CLIP,
    processor: CLIPProcessor,
    config: CLIPFilterConfig
) -> Optional[float]:
    """使用CLIP模型处理单张图片并计算相似度。"""
    try:
        if not os.path.isfile(image_path):
            return None

        # 处理图片和文本
        processed_inputs = processor(
            images=[image_path],
            text=[text_prompt],
            max_length=77,
            return_tensors="pd",
            return_attention_mask=False,
            mode="eval",
            do_resize=True,
            do_crop=True,
            padding_zero=True,
        )

        # 提取图片和文本的输入
        image_tensor = processed_inputs["image"]
        input_ids = processed_inputs["input_ids"]

        # 计算相似度
        with paddle.no_grad():
            similarity = model.clip_score(
                image=image_tensor, 
                input_ids=input_ids
            )
        
        return float(similarity.item())  # 转换为Python浮点数
    except Exception as e:
        print(f"处理图片 {image_path} 时出错：{e}")
        return None


@register()
def filter_by_clip(
    dataset: MMDataset,
    text_prompt: str,
    config: Optional[CLIPFilterConfig] = None,
) -> MMDataset:
    """使用CLIP相似度分数过滤数据集。"""
    if config is None:
        config = CLIPFilterConfig()

    # 初始化模型和处理器
    model = CLIP.from_pretrained(config.model_name, ignore_mismatched_sizes=False)
    model.eval()
    image_processor = CLIPImageProcessor.from_pretrained(os.path.join(config.model_name, "processor", "eval"))
    text_processor = CLIPTextProcessor.from_pretrained(os.path.join(config.model_name, "processor", "eval"))
    tokenizer = SimpleTokenizer()
    processor = CLIPProcessor(image_processor, text_processor, tokenizer)

    filtered_items = []

    # 使用tqdm显示进度条
    for item in tqdm(dataset, desc="使用CLIP过滤图片"):
        image_path = item.get('image')
        if not image_path or not os.path.exists(image_path):
            continue

        # 计算相似度分数
        similarity = clip_process_single_image(
            image_path=image_path,
            text_prompt=text_prompt,
            model=model,
            processor=processor,
            config=config
        )

        # 如果相似度达到阈值，则保留该项
        if similarity is not None and similarity >= config.threshold:
            filtered_items.append(item)

    return MMDataset(filtered_items)

