import os
import re
from typing import List, Dict
from paddlenlp.transformers import Qwen2Tokenizer
from paddlemix.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration
from paddlemix.processors.qwen2_vl_processing import Qwen2VLImageProcessor, Qwen2VLProcessor, process_vision_info
from ...core import T, MMDataset, register
import paddle
from tqdm import tqdm


# 初始化模型和处理器，避免每次调用时加载
class QNAProcessor:
    def __init__(self, model_name: str = "Qwen/Qwen2-VL-7B-Instruct"):
        self.model_name = model_name
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(model_name, dtype="bfloat16")
        self.image_processor = Qwen2VLImageProcessor()
        self.tokenizer = Qwen2Tokenizer.from_pretrained(model_name)
        self.processor = Qwen2VLProcessor(self.image_processor, self.tokenizer)

    def generate_qna_for_image(self, image_path: str) -> Dict:
        """
        对单张图片生成问答对。
        """
        # 准备模型的输入
        image_inputs, video_inputs = process_vision_info([
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ])

        # 指令内容
        instruction = """
        You are an AI visual assistant, and you are seeing a single image. Answer all questions as you are seeing the image.

        Please generate exactly 3 complete question-and-answer pairs about this image, following this structure:

        Q1: [First question about visual content]
        A1: [Detailed answer]

        Q2: [Second question about object relationships or counting]
        A2: [Detailed answer]

        Q3: [Third question about background knowledge or events]
        A3: [Detailed answer]

        For each question:
        1. Only ask about content that can be definitively answered from the image
        2. Provide detailed, multi-sentence answers
        3. Include specific visual details from the image
        4. Maintain an AI assistant perspective when answering

        Make sure each question type is different:
        - First question about basic visual content
        - Second question about object relationships or quantities
        - Third question about deeper context or implications
        """

        # 合并指令和问题
        image_pad_token = "<|vision_start|><|image_pad|><|vision_end|>"
        text = f"<|im_start|>system\n{instruction}<|im_end|>\n<|im_start|>user\n{image_pad_token}<|im_end|>\n<|im_start|>assistant\n"

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pd",
        )

        with paddle.no_grad():
            # 推理生成输出
            generated_ids = self.model.generate(**inputs, max_new_tokens=1280)
            output_text = self.processor.batch_decode(generated_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            
        # 使用正则表达式解析输出，提取问答对
        qna_pairs = []
        output_text = output_text[0]

        # 匹配问答对的正则表达式
        qna_pattern = r"(Q\d:.*?)(A\d:.*?)(?=Q\d:|$)"
        matches = re.findall(qna_pattern, output_text, re.DOTALL)

        # 处理每个匹配的问答对
        for i, match in enumerate(matches):
            question = match[0].strip().replace("Q1:", "").replace("Q2:", "").replace("Q3:", "").strip()
            answer = match[1].strip().replace("A1:", "").replace("A2:", "").replace("A3:", "").strip()

            # 如果问题或答案为空，则跳过该对
            if not question or not answer:
                continue

            # 第一个问答对的问题前加上 <image>\n
            if i == 0:
                question = f"<image>\n{question}"

            qna_pairs.append([question, answer])

        # 返回图像路径及对应的问答对
        return {
            "image": image_path,
            "conversations": qna_pairs
        }


# 定义生成问答对的过滤算子
@register()
def generate_qna_for_images(image_folder_path: str, model_name: str = "Qwen/Qwen2-VL-7B-Instruct") -> MMDataset:
    """
    根据给定的图片文件夹路径为每张图片生成问答对，并返回包含图片路径和问答对的数据集。
    
    参数:
        image_folder_path (str): 图片文件夹路径，包含待处理的图像。
        model_name (str): 使用的模型名称。
        
    返回:
        MMDataset: 生成的数据集，包含每个图像路径和其生成的问答对。
    """
    print("正在生成问答对...")

    # 初始化 QNAProcessor
    qna_processor = QNAProcessor(model_name=model_name)

    # 获取文件夹中的所有图像文件
    image_paths = [os.path.join(image_folder_path, filename) for filename in os.listdir(image_folder_path) if filename.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # 为每个图像生成问答对
    qna_data = []
    for image_path in tqdm(image_paths):
        qna_pair = qna_processor.generate_qna_for_image(image_path)
        qna_data.append(qna_pair)
        print(qna_pair)

    # 将图像路径和问答对拼接成数据集条目
    dataset = MMDataset(qna_data)

    return dataset
