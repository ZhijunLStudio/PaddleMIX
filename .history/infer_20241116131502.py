# import os
# import paddle
# from paddlemix.models.clip.clip_model import CLIP
# from paddlemix.processors.clip_processing import CLIPImageProcessor, CLIPTextProcessor, CLIPProcessor
# from paddlemix.processors.tokenizer import SimpleTokenizer

# def load_model_and_processors(model_path):
#     # 加载模型
#     model = CLIP.from_pretrained(model_path, ignore_mismatched_sizes=False)
#     model.eval()
    
#     # 加载处理器
#     image_processor = CLIPImageProcessor.from_pretrained(os.path.join(model_path, "processor", "eval"))
#     text_processor = CLIPTextProcessor.from_pretrained(os.path.join(model_path, "processor", "eval"))
#     tokenizer = SimpleTokenizer()
#     processor = CLIPProcessor(image_processor, text_processor, tokenizer)
    
#     return model, processor

# def infer(model, processor, image_path, text):
#     # 处理图像
#     image_inputs = processor.image_processor(image_path, return_tensors="pd")
#     # 使用 'image' 键而不是 'pixel_values'
#     image_tensor = image_inputs["image"]
#     print("image_tensor", image_tensor)
    
#     # 处理文本
#     # 处理文本
#     # 直接使用tokenizer进行分词
#     text_tokens = processor.tokenizer.encode(text)
#     print("text_tokens", text_tokens)
#     # 转换为paddle张量
#     text_tokens = paddle.to_tensor([text_tokens], dtype='int64')
#     print("text_tokens", text_tokens)
    
#     # 计算匹配度
#     with paddle.no_grad():
#         logits_per_image, logits_per_text = model(image_tensor, text_tokens)
#         probs = paddle.nn.functional.softmax(logits_per_image, axis=-1)
    
#     return probs.numpy()

# if __name__ == "__main__":
#     # 设置 GPU 设备
#     os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    
#     # 模型路径
#     model_path = "paddlemix/EVA/EVA02-CLIP-L-14"  # 替换为你的模型路径
    
#     # 加载模型和处理器
#     model, processor = load_model_and_processors(model_path)
    
#     # 推理输入
#     image_path = "/home/lizhijun/PaddleMIX-develop/newton_man/newton_0.jpg"  # 替换为你的图像路径
#     text = "your input text"  # 替换为你的文本
    
#     # 获取匹配度
#     match_probs = infer(model, processor, image_path, text)
#     print("匹配度概率：", match_probs)



import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import paddle
from paddlemix.models.clip.clip_model import CLIP
from paddlemix.processors.clip_processing import CLIPImageProcessor, CLIPTextProcessor, CLIPProcessor
from paddlemix.processors.tokenizer import SimpleTokenizer
import paddle.distributed.fleet.layers.mpu as mpu

def set_seed(seed=42):
    """设置随机数种子"""
    paddle.seed(seed)
    # 初始化分布式训练的随机数种子
    if not hasattr(mpu.get_rng_state_tracker(), "states"):
        mpu.get_rng_state_tracker().add("global_seed", seed)

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
#     image_inputs = processor.image_processor(image_path, return_tensors="pd")
#     image_tensor = image_inputs["image"]
    
#     # 处理文本 - 需要确保文本长度符合模型要求
#     text_tokens = processor.tokenizer.encode(text)
#     # 如果需要，可以进行padding到固定长度（比如77）
#     if len(text_tokens) < 77:
#         text_tokens = text_tokens + [0] * (77 - len(text_tokens))
#     text_tokens = text_tokens[:77]  # 截断到最大长度
    
#     # 转换为paddle张量
#     text_tokens = paddle.to_tensor([text_tokens], dtype='int64')
    
#     # 计算匹配度
#     with paddle.no_grad():

#         output = model(image_tensor, text_tokens)
#         print(output)
#         logits_per_image, logits_per_text = model(image_tensor, text_tokens)
#         probs = paddle.nn.functional.softmax(logits_per_image, axis=-1)

    
#     return probs.numpy()



def infer(model, processor, image_path, text):
    # 处理图像
    image_inputs = processor.image_processor(image_path, return_tensors="pd")
    image_tensor = image_inputs["image"]
    
    # 处理文本 - 确保文本长度符合模型要求
    text_tokens = processor.tokenizer.encode(text)
    if len(text_tokens) < 77:
        text_tokens = text_tokens + [0] * (77 - len(text_tokens))
    text_tokens = text_tokens[:77]  # 截断到最大长度
    text_tokens = paddle.to_tensor([text_tokens], dtype='int64')
    
    # 计算匹配度
    with paddle.no_grad():
        # 模型推理
        # output = model(image_tensor, text_tokens)
        # 输出为4轴，取最后一维te'zheng'xian
        output = model(image_tensor, text_tokens, skiploss=True)
        
        # 图像-文本特征相似度
        # logits_per_image = paddle.matmul(image_features, text_features, transpose_y=True) * logit_scale_exp
        # probs = paddle.nn.functional.softmax(logits_per_image, axis=-1)
    
    return probs.numpy()




if __name__ == "__main__":
    # 设置 GPU 设备

    
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
        "A modern photograph"
    ]
    
    # 对每个文本描述进行推理
    for text in texts:
        match_probs = infer(model, processor, image_path, text)
        print(f"文本 '{text}' 的匹配度概率：", match_probs)