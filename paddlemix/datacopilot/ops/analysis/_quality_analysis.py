from paddlenlp.transformers import AutoTokenizer, AutoModelForCausalLM
from paddlemix.datacopilot.core import MMDataset, register
import os
from collections import Counter
from tqdm import tqdm
from typing import Dict, List

# 预置的四种评估指标及其提示词
CRITERIA_PROMPTS = {
    "image_text_matching": """Please evaluate if the provided text caption accurately represents the main features and objects of the image. The caption doesn't need to detail every aspect of the image, but it should capture its primary theme. Rate the overall quality of the text caption's match to the image on a scale of 1-100, considering the criteria mentioned.""",
    "object_detail_fulfillment": """Please evaluate the text caption to determine if it provides detailed descriptions of objects that align with the image. Specifically, assess if the caption sufficiently describes the color, size, position, shape, material, etc., of the objects. Afterward, rate the caption's overall accuracy in capturing object details from the image on a scale of 1-100, based on the criteria provided.""",
    "caption_text_quality": """Please evaluate the text caption based on the following criteria: Grammatical Correctness, Diversity of Vocabulary (e.g., the range and uniqueness of words used), Fluency (e.g., smoothness and natural flow of sentences), Readability, Length, and Structure. Assign an overall quality score on a scale of 1-100.""",
    "semantic_understanding": """Evaluate the given text caption in relation to its corresponding image. Your goal is to determine if the text caption provides additional semantic information that isn't readily apparent just from the image itself. Rate the text caption's semantic depth on a scale from 1 to 100.""",
}

DEFAULT_PROMPT_TEMPLATE = """Text Caption: {caption}

{criteria}
A higher score indicates a higher level of {aspect}. Ensure that your scoring is nuanced and uses the entire range from 0 to 100, reflecting the subtle differences. The score should be given as an integer, with each number between 0 and 100 considered as a potential score, avoiding the tendency to round to multiples of 10. Please first output a single line containing the value indicating the scores. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias."""

# 加载模型的函数，支持传入模型名称
def load_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype="float16")
    return tokenizer, model

def evaluate_image_caption(
    dataset: MMDataset, 
    model_name: str = "Qwen/Qwen2.5-0.5B", 
    selected_metrics: List[str] = None
) -> Dict:
    """
    根据指定的指标评估图文质量。
    :param dataset: MMDataset 数据集
    :param model_name: 模型名称
    :param selected_metrics: 指定要使用的指标，默认为 all
    :return: 每个数据项的评估结果
    """
    # 如果未指定指标，默认使用所有指标
    if selected_metrics is None:
        selected_metrics = list(CRITERIA_PROMPTS.keys())
    
    # 加载模型
    tokenizer, model = load_model(model_name)
    
    # 存储最终结果
    results = {}

    for item in tqdm(dataset):
        item_id = item["id"]
        conversations = item["conversations"]
        
        # 遍历每个问答对
        for idx, conversation in enumerate(conversations):
            question = conversation["from"]  # 获取提问者
            answer = conversation["value"]  # 获取回答
            
            if question != "gpt":
                continue
            
            # 对每个选定的指标生成评估
            for metric in selected_metrics:
                criteria = CRITERIA_PROMPTS[metric]
                aspect = metric.replace("_", " ")
                caption = answer
                
                # 生成完整的 prompt
                full_prompt = DEFAULT_PROMPT_TEMPLATE.format(
                    caption=caption, 
                    criteria=criteria, 
                    aspect=aspect
                )
                
                # 使用 tokenizer 编码输入
                input_features = tokenizer(full_prompt, return_tensors="pd")
                
                # 模型生成输出
                outputs = model.generate(**input_features, max_length=256)
                
                # 解码生成结果
                decoded_output = tokenizer.batch_decode(outputs[0], skip_special_tokens=True)[0]

                
                # 存储结果
                if item_id not in results:
                    results[item_id] = {}
                if idx not in results[item_id]:
                    results[item_id][idx] = {}
                results[item_id][idx][metric] = decoded_output

    return results

@register()
def analyze_image_caption_with_metrics(dataset: MMDataset, model_name: str, selected_metrics: List[str] = None):
    """
    分析多轮对话的图文描述质量。
    """
    results = evaluate_image_caption(dataset, model_name, selected_metrics)
    
    # 打印或存储最终结果
    for item_id, conversations in results.items():
        print(f"Item ID: {item_id}")
        for idx, metrics in conversations.items():
            print(f"  Round {idx}:")
            for metric, output in metrics.items():
                print(f"    {metric}: {output}")
    return results
