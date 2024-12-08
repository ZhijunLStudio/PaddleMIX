from paddlenlp.transformers import AutoTokenizer, AutoModelForCausalLM
from paddlemix.datacopilot.core import MMDataset, register
import os
from collections import Counter
from tqdm import tqdm
from typing import Dict


# 加载模型的函数，支持传入模型名称
def load_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype="float16")
    return tokenizer, model

# 定义用于提取图像特征的 prompt
prompt = """You will be given a description of an image. Extract the following details if mentioned:

Color: List colors mentioned.
Shape: List shapes mentioned.
Position: Describe the object's position (relative to background/other objects).
Size: Describe the object's size (e.g., large, small).
Direction: Describe the object's orientation (e.g., tilt, front/back).
Relationship: Describe relationships between objects (e.g., on, near, etc.).
Action/State: Describe any actions or states (e.g., moving, still).
Category: List object types (e.g., cars, flowers).
Return the information in the following format:

Color: [list colors]
Shape: [list shapes]
Position: [position]
Size: [size description]
Direction: [direction]
Relationship: [relationship]
Action/State: [action/state]
Category: [category]

Text: "[text_input]"
"""



def clean_and_count(all_info):
    """清理并统计每个类别的出现频率"""
    cleaned_info = {}

    for category, items in all_info.items():
        # 清理无效项，例如 'list colors' 和 'None'
        valid_items = [item.strip() for item in items if item not in ['list colors', 'list shapes', 'position', 'None', 'size description', 'direction', 'action/state', 'relationship', 'category']]
        
        # 统计每个类别项的频率
        item_counts = Counter(valid_items)
        
        # 保存清理后的频率统计
        cleaned_info[category] = item_counts

    return cleaned_info


@register()
def analyze_gpt_responses(dataset: MMDataset, model_name: str = "Qwen/Qwen2.5-0.5B") -> Dict:
    """分析数据集中的所有 'gpt' 对话内容，并提取每个类别的信息"""
    results = {}
    all_info = {
        "Color": [],
        "Shape": [],
        "Position": [],
        "Size": [],
        "Direction": [],
        "Relationship": [],
        "Action/State": [],
        "Category": []
    }

    # 加载指定的模型
    tokenizer, model = load_model(model_name)

    for item in tqdm(dataset):
        gpt_responses = []

        # 获取所有 'gpt' 的对话内容
        for conversation in item["conversations"]:
            if conversation["from"] == "gpt":
                gpt_responses.append(conversation["value"])

        # 将所有 'gpt' 对话拼接为一个文本块
        gpt_text = "\n".join(gpt_responses)

        # 替换 prompt 中的占位符
        splice_prompt = prompt.replace("text_input", gpt_text)

        # 使用 tokenizer 对输入文本进行编码
        input_features = tokenizer(splice_prompt, return_tensors="pd")

        # 生成模型的输出
        outputs = model.generate(**input_features, max_length=128)

        # 解码并获取分析结果
        analysis_result = tokenizer.batch_decode(outputs[0], skip_special_tokens=True)[0]

        # 提取具体信息并分类存储
        for category in all_info.keys():
            # 查找并提取该类别的信息
            start_idx = analysis_result.find(f"{category}: [")
            if start_idx != -1:
                start_idx += len(f"{category}: [")
                end_idx = analysis_result.find("]", start_idx)
                if end_idx != -1:
                    info = analysis_result[start_idx:end_idx]
                    all_info[category].extend(info.split(","))
        
        # 存储结果
        results[item['id']] = analysis_result

    # 使用clean_and_count函数清理all_info并统计频率
    cleaned_info = clean_and_count(all_info)

    # 输出每个类别及其项的频率
    for category, counts in cleaned_info.items():
        print(f"{category}:")
        for item, count in counts.items():
            print(f"  {item}: {count}")
        print("-" * 50)
    # 返回合并后的结果，包括每个类别的实际信息
    return cleaned_info
