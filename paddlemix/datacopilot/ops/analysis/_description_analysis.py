# from paddlenlp.transformers import AutoTokenizer, AutoModelForCausalLM
# from paddlemix.datacopilot.core import MMDataset, register
# import os
# from collections import Counter
# from tqdm import tqdm
# from typing import Dict


# # 加载模型的函数，支持传入模型名称
# def load_model(model_name: str):
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForCausalLM.from_pretrained(model_name, dtype="float16")
#     return tokenizer, model

# # 定义用于提取图像特征的 prompt
# prompt = """You will be given a description of an image. Extract the following details if mentioned:

# Color: List colors mentioned.
# Shape: List shapes mentioned.
# Position: Describe the object's position (relative to background/other objects).
# Size: Describe the object's size (e.g., large, small).
# Direction: Describe the object's orientation (e.g., tilt, front/back).
# Relationship: Describe relationships between objects (e.g., on, near, etc.).
# Action/State: Describe any actions or states (e.g., moving, still).
# Category: List object types (e.g., cars, flowers).
# Return the information in the following format:

# Color: [list colors]
# Shape: [list shapes]
# Position: [position]
# Size: [size description]
# Direction: [direction]
# Relationship: [relationship]
# Action/State: [action/state]
# Category: [category]

# Text: "[text_input]"
# """



# def clean_and_count(all_info):
#     """清理并统计每个类别的出现频率"""
#     cleaned_info = {}

#     for category, items in all_info.items():
#         # 清理无效项，例如 'list colors' 和 'None'
#         valid_items = [item.strip() for item in items if item not in ['list colors', 'list shapes', 'position', 'None', 'size description', 'direction', 'action/state', 'relationship', 'category']]
        
#         # 统计每个类别项的频率
#         item_counts = Counter(valid_items)
        
#         # 保存清理后的频率统计
#         cleaned_info[category] = item_counts

#     return cleaned_info


# @register()
# def analyze_gpt_responses(dataset: MMDataset, model_name: str = "Qwen/Qwen2.5-0.5B") -> Dict:
#     """分析数据集中的所有 'gpt' 对话内容，并提取每个类别的信息"""
#     results = {}
#     all_info = {
#         "Color": [],
#         "Shape": [],
#         "Position": [],
#         "Size": [],
#         "Direction": [],
#         "Relationship": [],
#         "Action/State": [],
#         "Category": []
#     }

#     # 加载指定的模型
#     tokenizer, model = load_model(model_name)

#     for item in tqdm(dataset):
#         gpt_responses = []

#         # 获取所有 'gpt' 的对话内容
#         for conversation in item["conversations"]:
#             if conversation["from"] == "gpt":
#                 gpt_responses.append(conversation["value"])

#         # 将所有 'gpt' 对话拼接为一个文本块
#         gpt_text = "\n".join(gpt_responses)

#         # 替换 prompt 中的占位符
#         splice_prompt = prompt.replace("text_input", gpt_text)

#         # 使用 tokenizer 对输入文本进行编码
#         input_features = tokenizer(splice_prompt, return_tensors="pd")

#         # 生成模型的输出
#         outputs = model.generate(**input_features, max_length=128)

#         # 解码并获取分析结果
#         analysis_result = tokenizer.batch_decode(outputs[0], skip_special_tokens=True)[0]

#         # 提取具体信息并分类存储
#         for category in all_info.keys():
#             # 查找并提取该类别的信息
#             start_idx = analysis_result.find(f"{category}: [")
#             if start_idx != -1:
#                 start_idx += len(f"{category}: [")
#                 end_idx = analysis_result.find("]", start_idx)
#                 if end_idx != -1:
#                     info = analysis_result[start_idx:end_idx]
#                     all_info[category].extend(info.split(","))
        
#         # 存储结果
#         results[item['id']] = analysis_result

#     # 使用clean_and_count函数清理all_info并统计频率
#     cleaned_info = clean_and_count(all_info)

#     # 输出每个类别及其项的频率
#     for category, counts in cleaned_info.items():
#         print(f"{category}:")
#         for item, count in counts.items():
#             print(f"  {item}: {count}")
#         print("-" * 50)
#     # 返回合并后的结果，包括每个类别的实际信息
#     return cleaned_info



from paddlenlp.transformers import AutoTokenizer, AutoModelForCausalLM
from paddlemix.datacopilot.core import MMDataset, register
from tqdm import tqdm
from collections import Counter
from typing import Dict
import paddle


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

# 清理并统计每个类别的出现频率
def clean_and_count(all_info):
    cleaned_info = {}
    for category, items in all_info.items():
        valid_items = [item.strip() for item in items if item not in ['list colors', 'list shapes', 'position', 'None', 'size description', 'direction', 'action/state', 'relationship', 'category']]
        item_counts = Counter(valid_items)
        cleaned_info[category] = item_counts
    return cleaned_info



@register()
def description_analysis(dataset: MMDataset, model_name: str = "Qwen/Qwen2.5-0.5B", batch_size: int = 8) -> Dict:
    """分析数据集中的所有 'gpt' 对话内容，并提取每个类别的信息"""
    tokenizer, model = load_model(model_name)
    model.eval()
    
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

    all_data = []  # 存储所有待处理的数据
    total_samples = 0
    filtered_data = {}

    print("收集数据中...")
    for item in dataset:
        image_path = item.get("image", "")
        conversations = item.get("conversations", [])
        
        # 将每个对话转换为一个问题-答案对
        for conversation in conversations:
            if len(conversation) == 2:  # 确保每个conversation是一个问题-答案对
                question, answer = conversation
                cleaned_question = question.strip()
                
                all_data.append({
                    'image_path': image_path,
                    'question': cleaned_question,
                    'answer': answer,
                    'prompt': prompt.replace("text_input", cleaned_question + "\n" + answer)  # 格式化 prompt
                })
                total_samples += 1

    num_batches = (total_samples + batch_size - 1) // batch_size
    print(f"总共收集到 {total_samples} 条数据，将分成 {num_batches} 个批次处理")

    for batch_idx in tqdm(range(num_batches), desc="处理批次"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_samples)
        
        batch_data = all_data[start_idx:end_idx]
        batch_prompts = [item['prompt'] for item in batch_data]

        try:
            input_features = tokenizer(batch_prompts, return_tensors="pd", padding=True)
            
            with paddle.no_grad():
                outputs = model.generate(**input_features, max_length=128)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                if not isinstance(outputs, paddle.Tensor):
                    outputs = paddle.to_tensor(outputs)
                
                outputs_list = outputs.numpy().tolist()

            decoded_outputs = tokenizer.batch_decode(outputs_list, skip_special_tokens=True)

            # 处理当前批次的结果
            for idx, analysis_result in enumerate(decoded_outputs):
                print("analysis_result:", analysis_result)
                # 提取具体信息并分类存储
                for category in all_info.keys():
                    start_idx = analysis_result.find(f"{category}: [")
                    if start_idx != -1:
                        start_idx += len(f"{category}: [")
                        end_idx = analysis_result.find("]", start_idx)
                        if end_idx != -1:
                            info = analysis_result[start_idx:end_idx]
                            all_info[category].extend(info.split(","))
                
                # 如果得分合适，将数据加入过滤后的结果
                current_item = batch_data[idx]
                image_path = current_item['image_path']
                
                # 如果图片路径没有记录，创建新的条目
                if image_path not in filtered_data:
                    filtered_data[image_path] = {
                        "image": image_path,
                        "conversations": []
                    }
                
                # 将问答对加入到对应的图片数据中
                filtered_data[image_path]["conversations"].append([
                    current_item['question'],
                    current_item['answer']
                ])

        except Exception as e:
            print(f"处理批次 {batch_idx + 1}/{num_batches} 时出错:")
            print(f"错误信息: {e}")
            print("-" * 50)
            continue

    # 使用 clean_and_count 函数清理 all_info 并统计频率
    cleaned_info = clean_and_count(all_info)

    # 输出每个类别及其项的频率
    for category, counts in cleaned_info.items():
        print(f"{category}:")
        for item, count in counts.items():
            print(f"  {item}: {count}")
        print("-" * 50)

    # 将过滤后的数据转化为 MMDataset 格式并返回
    final_dataset = list(filtered_data.values())

    print(f"处理完成:")
    print(f"总问答对数量: {total_samples}")
    print(f"涉及的图片数量: {len(filtered_data)}")

    return MMDataset(final_dataset)
