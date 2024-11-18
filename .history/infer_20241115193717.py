import os
os.e
import paddle
from paddlemix.models.clip.clip_model import CLIP
from paddlemix.processors.clip_processing import CLIPImageProcessor, CLIPTextProcessor, CLIPProcessor
from paddlemix.processors.tokenizer import SimpleTokenizer


def load_model_and_processors(model_path):
    # 加载模型
    model = CLIP.from_pretrained(model_path, ignore_mismatched_sizes=False)
    model.eval()

    # 加载处理器
    image_processor = CLIPImageProcessor.from_pretrained(os.path.join(model_path, "processor", "eval"))
    text_processor = CLIPTextProcessor.from_pretrained(os.path.join(model_path, "processor", "eval"))
    tokenizer = SimpleTokenizer()
    processor = CLIPProcessor(image_processor, text_processor, tokenizer)

    return model, processor


# def infer(model, processor, image_path, text):
#     # 处理图像
#     image = processor.image_processor(image_path)
#     image = paddle.to_tensor(image).unsqueeze(0)  # 添加batch维度

#     # 处理文本
#     text_inputs = processor.text_processor([text], return_tensors="pd", padding=True)
#     text_tokens = text_inputs["input_ids"]

#     # 计算匹配度
#     with paddle.no_grad():
#         logits_per_image, logits_per_text = model(image, text_tokens)
#         probs = paddle.nn.functional.softmax(logits_per_image, axis=-1)
    
#     return probs.numpy()



def infer(model, processor, image_path, text):
    
    # 处理图像
    image_inputs = processor.image_processor(image_path, return_tensors="pd")
    image_tensor = image_inputs["pixel_values"]  # 提取实际的图像张量

    # 处理文本
    text_inputs = processor.text_processor([text], return_tensors="pd", padding=True)
    text_tokens = text_inputs["input_ids"]

    # 计算匹配度
    with paddle.no_grad():
        logits_per_image, logits_per_text = model(image_tensor, text_tokens)
        probs = paddle.nn.functional.softmax(logits_per_image, axis=-1)
    
    return probs.numpy()



if __name__ == "__main__":
    # 模型路径
    model_path = "paddlemix/EVA/EVA02-CLIP-L-14"  # 替换为你的模型路径

    # 加载模型和处理器
    model, processor = load_model_and_processors(model_path)

    # 推理输入
    image_path = "/home/lizhijun/PaddleMIX-develop/newton_man/newton_0.jpg"  # 替换为你的图像路径
    text = "your input text"  # 替换为你的文本

    # 获取匹配度
    match_probs = infer(model, processor, image_path, text)
    print("匹配度概率：", match_probs)
