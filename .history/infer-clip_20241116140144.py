import os
import paddle
import numpy as np
from tqdm import tqdm
from paddlemix.processors import CLIPProcessor
from paddlemix.utils import set_seed





def load_model_and_processors(model_path):
    """
    加载模型和处理器。
    """
    # 加载模型
    model = CLIP.from_pretrained(model_path)
    model.eval()

    # 加载处理器
    image_processor = CLIPImageProcessor.from_pretrained(os.path.join(model_path, "processor", "eval"))
    text_processor = CLIPTextProcessor.from_pretrained(os.path.join(model_path, "processor", "eval"))
    tokenizer = SimpleTokenizer()
    processor = CLIPProcessor(image_processor, text_processor, tokenizer)

    return model, processor


class BatchSimilarity:
    """
    用于批量化计算图片和文本相似度。
    """

    def __init__(self, model, processor, batch_size=8):
        self.model = model
        self.processor = processor
        self.batch_size = batch_size

    def compute_similarity(self, image_path, texts):
        """
        计算给定图片和文本之间的相似度。

        Args:
            image_path (str): 图片路径。
            texts (list of str): 文本列表。

        Returns:
            list of float: 图片与每个文本的相似度。
        """
        # 处理图片
        image = self.processor.image_processor([image_path], return_tensors="pd")["pixel_values"]

        # 分批处理文本
        similarities = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            text_features = self.processor.text_processor(batch_texts, return_tensors="pd")["input_ids"]

            # 推理图片和文本的特征
            with paddle.no_grad():
                image_features = self.model.get_image_features(image)
                text_features = self.model.get_text_features(text_features)

                # 归一化
                image_features = image_features / paddle.norm(image_features, axis=-1, keepdim=True)
                text_features = text_features / paddle.norm(text_features, axis=-1, keepdim=True)

                # 计算相似度
                similarity = paddle.matmul(image_features, text_features.T)
                similarities.extend(similarity.numpy().flatten().tolist())

        return similarities


def infer(model, processor, image_path, text):
    """
    单次推理，用于计算图片和单个文本的相似度。

    Args:
        model (paddle.nn.Layer): CLIP模型。
        processor (CLIPProcessor): CLIP处理器。
        image_path (str): 图片路径。
        text (str): 文本描述。

    Returns:
        float: 相似度得分。
    """
    batch_similarity = BatchSimilarity(model, processor, batch_size=1)
    return batch_similarity.compute_similarity(image_path, [text])[0]


if __name__ == "__main__":
    # 设置 GPU 设备
    paddle.set_device("gpu:0")
    
    # 设置随机数种子
    set_seed(42)
    
    # 模型路径
    model_path = "paddlemix/EVA/EVA02-CLIP-L-14"  # 替换为你的模型路径
    
    # 加载模型和处理器
    model, processor = load_model_and_processors(model_path)
    
    # 推理输入
    image_path = "/home/lizhijun/PaddleMIX-develop/newton_man/newton_0.jpg"
    
    # 准备多个文本描述进行测试
    texts = [
        "A portrait of Isaac Newton",
        "A scientific diagram",
        "A landscape painting",
        "A modern photograph",
        "A cat"
    ]
    
    # 对每个文本描述进行推理
    similarity_calculator = BatchSimilarity(model, processor)
    match_probs = similarity_calculator.compute_similarity(image_path, texts)
    
    for text, prob in zip(texts, match_probs):
        print(f"文本 '{text}' 的匹配度概率：{prob}")
