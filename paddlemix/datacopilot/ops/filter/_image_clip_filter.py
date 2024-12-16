import os
import re
from tqdm import tqdm
from typing import Optional, List
from dataclasses import dataclass

import paddle
from PIL import Image, ImageDraw, ImageFont
from paddlemix.processors.clip_processing import CLIPImageProcessor, CLIPTextProcessor, CLIPProcessor
from paddlemix.models.clip.clip_model import CLIP
from paddlemix.processors.tokenizer import SimpleTokenizer
from paddlemix.datacopilot.core import MMDataset, register
from ...misc import parallel_map, ParallelMode

import time

@dataclass
class CLIPFilterConfig:
    """CLIP过滤的配置"""
    model_name: str = "paddlemix/CLIP/CLIP-ViT-L-14-laion2B-s32B-b82K"
    threshold: float = 0.25
    batch_size: int = 8  # 批量大小
    save_images: bool = True  # 是否保存低置信度图像
    save_dir: str = "./low_confidence_images" # 保存低置信度图像的目录


def clip_process_batch(
    image_paths: List[str],
    text_prompts: List[str],
    model: CLIP,
    processor: CLIPProcessor,
) -> List[Optional[float]]:
    """处理一批图片和文本，返回相似度"""
    try:
        processed_inputs = processor(
            images=image_paths,
            text=text_prompts,
            max_length=77,
            return_tensors="pd",
            return_attention_mask=False,
            mode="eval",
            do_resize=True,
            do_crop=True,
            padding_zero=True,
        )

        image_tensor = processed_inputs["image"]
        input_ids = processed_inputs["input_ids"]

        print(f"Image tensor shape: {image_tensor.shape}")
        print(f"Text prompt shape: {len(text_prompts)}")
        print(f"Tokenized output shape: {input_ids.shape}")

        with paddle.no_grad():
            similarities = model.clip_score(image=image_tensor, input_ids=input_ids)

        return [float(similarity.item()) for similarity in similarities]
    except Exception as e:
        print(f"批量处理出错：{e}")
        return [None] * len(image_paths)


def clean_question(question: str) -> str:
    """清理问题文本，去掉 <image> 占位符和多余换行"""
    return question.replace("<image>", "").replace("\n<image>", " ").replace("<image>\n", " ").strip()


def format_text(question: str, answer: str) -> str:
    """格式化文本为 'question: ... \nanswer: ...'"""
    return f"question: {question} \nanswer: {answer}"


def contains_coordinates(text: str) -> bool:
    """检查文本中是否包含 4 个坐标值的形式 [x1, y1, x2, y2]"""
    pattern = r"\[\s*[\d.]+\s*,\s*[\d.]+\s*,\s*[\d.]+\s*,\s*[\d.]+\s*\]"
    return bool(re.search(pattern, text))


def save_combined_image(image_path, text, similarity, save_dir, sample_index):
    """拼接原图和问答对 + 相似度，并保存图片（上下拼接）"""
    try:
        # 打开原图
        image = Image.open(image_path).convert("RGB")
        image_width, image_height = image.size

        # 设置字体和文本区域宽高
        font = ImageFont.load_default()
        text_height = 100  # 文本区域高度
        combined_width = image_width
        combined_height = image_height + text_height

        # 创建拼接图
        combined_image = Image.new("RGB", (combined_width, combined_height), (255, 255, 255))
        combined_image.paste(image, (0, 0))

        # 绘制文本
        draw = ImageDraw.Draw(combined_image)
        text_area_y = image_height + 10  # 文本起始位置
        draw.text((10, text_area_y), text, fill="black", font=font)
        draw.text((10, text_area_y + 40), f"Similarity: {similarity:.2f}", fill="black", font=font)


        # 生成保存路径
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"low_confidence_{sample_index}.jpg")

        # 保存图片
        combined_image.save(save_path)
        print(f"Saved low confidence sample: {save_path}")
    except Exception as e:
        print(f"保存拼接图片出错：{e}")



@register()
def image_clip_filter(
    dataset: MMDataset,
    config: Optional[CLIPFilterConfig] = None
) -> MMDataset:
    """使用CLIP过滤低置信度问答对并保存图片"""
    if config is None:
        config = CLIPFilterConfig()
    save_dir = config.save_dir  # 从配置中获取保存目录

    model = CLIP.from_pretrained(config.model_name, ignore_mismatched_sizes=False)
    model.eval()
    image_processor = CLIPImageProcessor.from_pretrained(os.path.join(config.model_name, "processor", "eval"))
    text_processor = CLIPTextProcessor.from_pretrained(os.path.join(config.model_name, "processor", "eval"))
    tokenizer = SimpleTokenizer()
    processor = CLIPProcessor(image_processor, text_processor, tokenizer)

    filtered_items = []
    batch_size = config.batch_size

    t0 = time.time()
    # 展开问答对
    all_samples = []
    for item in dataset:
        image_path = item.get('image')
        if not image_path or not os.path.exists(image_path):
            continue
        conversations = []
        for conversation in item.get("conversations", []):
            question, answer = conversation
            if contains_coordinates(question) or contains_coordinates(answer):
                continue  # 跳过包含坐标的问答对
            cleaned_question = clean_question(question)
            formatted_text = format_text(cleaned_question, answer)
            conversations.append((image_path, formatted_text, conversation))
        all_samples.extend(conversations)
    t1 = time.time()
    print(f"展开问答对花费时间{(t1-t0)*1000}ms")
    sample_index = 0
    low_confidence_samples = []
    for i in tqdm(range(0, len(all_samples), batch_size), desc="过滤低置信度问答对"):
        t1_1 = time.time()
        batch = all_samples[i:i + batch_size]
        image_paths = [sample[0] for sample in batch]
        text_prompts = [sample[1] for sample in batch]
        t1_2 = time.time()

        similarities = clip_process_batch(
            image_paths=image_paths,
            text_prompts=text_prompts,
            model=model,
            processor=processor,
        )
        t1_3 = time.time()
        for (image_path, formatted_text, conversation), similarity in zip(batch, similarities):
            
            if similarity is not None and similarity < config.threshold:
                if config.save_images:
                    save_combined_image(
                        image_path=image_path,
                        text=formatted_text,
                        similarity=similarity,
                        save_dir=save_dir,
                        sample_index=sample_index,
                    )
                sample_index += 1
                low_confidence_samples.append((image_path, conversation))
        t1_4 = time.time()
        print(f"数据准备：{t1_2 - t1_1}s，多batch：{t1_3-t1_2}s，后处理：{t1_4-t1_3}s")

    t2 = time.time()
    print(f"过滤低置信度花费时间{(t2-t1)*1000}ms")

    def filter_high_confidence(item):
        image_path = item.get('image')
        if not image_path or not os.path.exists(image_path):
            return None  # 确保正确跳过无效项
        new_conversations = [
            conversation for conversation in item.get("conversations", [])
            if (image_path, conversation) not in low_confidence_samples
        ]
        # 返回项仅当新对话列表非空
        if new_conversations:
            return {"image": image_path, "conversations": new_conversations}
        return None

    filtered_items = parallel_map(
        filter_high_confidence,
        dataset.items,
        max_workers=8,  # 可调整线程数
        mode=ParallelMode.THREAD,
        progress=True,
        order=False,
    )
    t3 = time.time()
    print(f"保留高置信度花费时间{(t3-t2)*1000}ms")


    return MMDataset(filtered_items)

