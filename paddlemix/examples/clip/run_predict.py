# coding:utf-8

import os
# 获取项目的根目录（`paddlemix` 的父目录）
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, project_root)

from paddlemix.processors.tokenizer import SimpleTokenizer
import paddle
from paddlemix.processors.clip_processing import CLIPImageProcessor, CLIPTextProcessor, CLIPProcessor
from paddlemix.models.clip.clip_model import CLIP
from paddlenlp.trainer import PdArgumentParser
import os
from dataclasses import dataclass
import pprint




# Define the ModelArguments class for specifying model configuration
@dataclass
class ModelArguments:
    model: str = "paddlemix/CLIP/CLIP-ViT-L-14-laion2B-s32B-b82K"


# Main function to handle evaluation
def main_worker(model_args):

    model = CLIP.from_pretrained(model_args.model, ignore_mismatched_sizes=False)
    model.eval()


    # Prepare the image-text pair manually
    image_path = "/home/lizhijun/PaddleMIX-develop/paddlemix/demo_images/examples_image1.jpg"  # Replace with the path to your image
    text_input = "person"  # Replace with your text input

    # Use the processors to process the image and text
    image_processor = CLIPImageProcessor.from_pretrained(os.path.join(model_args.model, "processor", "eval"))
    text_processor = CLIPTextProcessor.from_pretrained(os.path.join(model_args.model, "processor", "eval"))
    tokenizer = SimpleTokenizer()
    processor = CLIPProcessor(image_processor, text_processor, tokenizer)

    # Wrap the image and text into a batch
    process_input = processor(
        images=[image_path],
        text=[text_input],
        max_length=77,
        return_tensors="pd",
        return_attention_mask=False,
        mode="eval",
        do_resize=True,
        do_crop=True,
        padding_zero=True,
    )

    # 解构输入
    image = process_input['image']
    input_ids = process_input['input_ids']
    print("text_input:", text_input)

    # 禁用梯度计算进行推理
    with paddle.no_grad():
        # 调用 clip_score 方法
        similarity = model.clip_score(
            image=image, 
            input_ids=input_ids, 
        )



# Ensure proper environment setup
if __name__ == "__main__":
    parser = PdArgumentParser(ModelArguments)
    model_args, = parser.parse_args_into_dataclasses()
    pprint.pprint(model_args)
    main_worker(model_args)
