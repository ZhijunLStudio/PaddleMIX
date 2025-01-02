# from paddlenlp.transformers import AutoTokenizer, AutoModelForCausalLM
# from paddlemix.datacopilot.core import MMDataset, register
# from tqdm import tqdm
# from collections import Counter
# from typing import Dict, List, Optional
# import paddle
# import json

# def load_model(model_name: str):
#     """加载模型和分词器"""
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForCausalLM.from_pretrained(model_name, dtype="float16")
#     return tokenizer, model

# def parse_model_output(output_text: str) -> Optional[dict]:
#     """解析模型输出的JSON格式文本"""
#     try:
#         # 找到第一个 '{' 和最后一个 '}' 之间的内容
#         start_idx = output_text.find('{')
#         end_idx = output_text.rfind('}') + 1
#         if start_idx == -1 or end_idx == 0:
#             return None
        
#         json_str = output_text[start_idx:end_idx]
#         parsed_data = json.loads(json_str)
        
#         # 确保所有必需的键都存在
#         required_keys = ['colors', 'shapes', 'position', 'size', 'direction', 
#                         'relationships', 'actions', 'categories']
        
#         # 为缺失的键添加默认值
#         for key in required_keys:
#             if key not in parsed_data:
#                 parsed_data[key] = ["N/A"] if key in ['colors', 'shapes', 'relationships', 
#                                                      'actions', 'categories'] else "N/A"
                
#         return parsed_data
#     except json.JSONDecodeError:
#         print(f"Failed to parse output: {output_text}")
#         return None

# def clean_and_count(all_info: List[dict]) -> Dict[str, Counter]:
#     """清理并统计每个类别的出现频率"""
#     cleaned_info = {}
    
#     # 定义需要处理的类别及其对应的key
#     categories = {
#         'Colors': 'colors',
#         'Shapes': 'shapes',
#         'Position': 'position',
#         'Size': 'size',
#         'Direction': 'direction',
#         'Relationships': 'relationships',
#         'Actions': 'actions',
#         'Categories': 'categories'
#     }
    
#     # 初始化所有类别的Counter
#     for category in categories.keys():
#         cleaned_info[category] = Counter()
    
#     # 处理每条数据
#     for item in all_info:
#         if not item:  # 跳过无效数据
#             continue
            
#         for category, key in categories.items():
#             value = item.get(key, "N/A")
#             if isinstance(value, list):
#                 # 处理列表类型的值
#                 for v in value:
#                     if v != "N/A":
#                         cleaned_info[category][v.strip().lower()] += 1
#             else:
#                 # 处理字符串类型的值
#                 if value != "N/A":
#                     cleaned_info[category][value.strip().lower()] += 1
    
#     return cleaned_info

# # 定义提示模板
# ANALYSIS_PROMPT = '''
# Your task is to analyze the given text description and extract specific attributes step by step. Follow these steps:

# 1. **Understand the description**: Read the description carefully and identify all key elements, such as objects, actions, and their relationships.
# 2. **Attribute extraction**:
#    - Identify colors mentioned in the description (e.g., "red", "blue").
#    - Identify shapes if mentioned (e.g., "circle", "square"). If none are present, use "N/A".
#    - Describe the position of the objects (e.g., "on the street", "next to the building").
#    - Describe the size of the objects (e.g., "large", "small").
#    - Identify the direction or orientation of objects (e.g., "facing forward").
#    - Identify relationships between objects (e.g., "next to", "behind").
#    - Identify actions mentioned in the description (e.g., "parked", "standing").
#    - Identify categories of objects (e.g., "car", "building").
# 3. **Output the result**: Organize the extracted information into the following JSON format:

# ```json
# {
#     "colors": ["color1", "color2", ...],
#     "shapes": ["shape1", "shape2", ...],
#     "position": "description of position",
#     "size": "size description",
#     "direction": "direction or orientation",
#     "relationships": ["relationship1", "relationship2", ...],
#     "actions": ["action1", "action2", ...],
#     "categories": ["category1", "category2", ...]
# }
# '''

# @register()
# def description_analysis(dataset: MMDataset, 
#                         model_name: str = "Qwen/Qwen2.5-0.5B", 
#                         batch_size: int = 8) -> Dict:
#     """
#     分析数据集中的所有对话内容，提取和统计各类属性信息。
    
#     Args:
#         dataset: MMDataset对象，包含图片路径和对话内容
#         model_name: 使用的模型名称
#         batch_size: 批处理大小
    
#     Returns:
#         Dict: 包含处理后的数据集
#     """
#     # 加载模型和分词器
#     tokenizer, model = load_model(model_name)
#     model.eval()
    
#     # 存储所有解析后的结果
#     all_parsed_results = []
#     filtered_data = {}
    
#     # 收集待处理数据
#     all_data = []
#     print("收集数据中...")
#     for item in dataset:
#         image_path = item.get("image", "")
#         conversations = item.get("conversations", [])
        
#         # 将每个对话转换为问答对
#         for conversation in conversations:
#             if len(conversation) == 2:
#                 question, answer = conversation
#                 cleaned_question = question.strip()
                
#                 all_data.append({
#                     'image_path': image_path,
#                     'question': cleaned_question,
#                     'answer': answer,
#                     'prompt': ANALYSIS_PROMPT.replace("[text_input]", 
#                                                     f"{cleaned_question}\n{answer}")
#                 })
    
#     total_samples = len(all_data)
#     num_batches = (total_samples + batch_size - 1) // batch_size
#     print(f"总共收集到 {total_samples} 条数据，将分成 {num_batches} 个批次处理")
    
#     # 批量处理数据
#     for batch_idx in tqdm(range(num_batches), desc="处理批次"):
#         start_idx = batch_idx * batch_size
#         end_idx = min(start_idx + batch_size, total_samples)
#         batch_data = all_data[start_idx:end_idx]
#         batch_prompts = [item['prompt'] for item in batch_data]

#         try:
#             # 模型推理
#             input_features = tokenizer(batch_prompts, return_tensors="pd", padding=True)
#             with paddle.no_grad():
#                 outputs = model.generate(**input_features, max_length=512)

#             # 解码输出
#             decoded_outputs = tokenizer.batch_decode(outputs[0], skip_special_tokens=True)
#             print(f"Decoded Outputs for Batch {batch_idx}:\n{decoded_outputs}")

#             # 处理当前批次结果
#             for idx, analysis_result in enumerate(decoded_outputs):
#                 parsed_result = parse_model_output(analysis_result)
#                 print(f"Parsed Result for Input {idx} in Batch {batch_idx}: {parsed_result}")
#                 if parsed_result:
#                     all_parsed_results.append(parsed_result)

#                     # 更新过滤后的数据集
#                     current_item = batch_data[idx]
#                     image_path = current_item['image_path']
#                     if image_path not in filtered_data:
#                         filtered_data[image_path] = {
#                             "image": image_path,
#                             "conversations": []
#                         }
#                     filtered_data[image_path]["conversations"].append([
#                         current_item['question'],
#                         current_item['answer']
#                     ])
#                 else:
#                     print(f"Invalid parsed result for input: {analysis_result}")

#         except Exception as e:
#             print(f"处理批次 {batch_idx + 1}/{num_batches} 时出错:")
#             print(f"错误信息: {str(e)}")
#             print("-" * 50)
#             continue
    
#     # 清理并统计结果
#     cleaned_info = clean_and_count(all_parsed_results)
    
#     # 输出统计信息
#     print("\n=== 属性统计结果 ===")
#     for category, counts in cleaned_info.items():
#         print(f"\n{category}:")
#         for item, count in counts.most_common(10):  # 显示每个类别的前10个最常见项
#             print(f"  {item}: {count}")
    
#     # 输出处理总结
#     print("\n=== 处理总结 ===")
#     print(f"总问答对数量: {total_samples}")
#     print(f"成功解析数量: {len(all_parsed_results)}")
#     print(f"涉及的图片数量: {len(filtered_data)}")
    
#     # 返回过滤后的数据集
#     return MMDataset(list(filtered_data.values()))




from paddlenlp.transformers import AutoTokenizer, AutoModelForCausalLM
from paddlemix.datacopilot.core import MMDataset, register
from tqdm import tqdm
from collections import Counter
from typing import Dict, List, Optional
import paddle
import json

def load_model(model_name: str):
    """加载模型和分词器"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype="float16")
    return tokenizer, model

def parse_model_output(output_text: str) -> Optional[dict]:
    """解析模型输出的JSON格式文本"""
    try:
        # 找到第一个 '{' 和最后一个 '}' 之间的内容
        start_idx = output_text.find('{')
        end_idx = output_text.rfind('}') + 1
        if start_idx == -1 or end_idx == 0:
            return None
        
        json_str = output_text[start_idx:end_idx]
        parsed_data = json.loads(json_str)
        
        # 确保所有必需的键都存在
        required_keys = ['colors', 'shapes', 'position', 'size', 'direction', 
                        'relationships', 'actions', 'categories']
        
        # 为缺失的键添加默认值
        for key in required_keys:
            if key not in parsed_data:
                parsed_data[key] = ["N/A"] if key in ['colors', 'shapes', 'relationships', 
                                                     'actions', 'categories'] else "N/A"
                
        return parsed_data
    except json.JSONDecodeError:
        print(f"Failed to parse output: {output_text}")
        return None

def clean_and_count(all_info: List[dict]) -> Dict[str, Counter]:
    """清理并统计每个类别的出现频率"""
    cleaned_info = {}
    
    # 定义需要处理的类别及其对应的key
    categories = {
        'Colors': 'colors',
        'Shapes': 'shapes',
        'Position': 'position',
        'Size': 'size',
        'Direction': 'direction',
        'Relationships': 'relationships',
        'Actions': 'actions',
        'Categories': 'categories'
    }
    
    # 初始化所有类别的Counter
    for category in categories.keys():
        cleaned_info[category] = Counter()
    
    # 处理每条数据
    for item in all_info:
        if not item:  # 跳过无效数据
            continue
            
        for category, key in categories.items():
            value = item.get(key, "N/A")
            if isinstance(value, list):
                # 处理列表类型的值
                for v in value:
                    if v != "N/A":
                        cleaned_info[category][v.strip().lower()] += 1
            else:
                # 处理字符串类型的值
                if value != "N/A":
                    cleaned_info[category][value.strip().lower()] += 1
    
    return cleaned_info

# 定义提示模板
ANALYSIS_PROMPT = '''
Extract attributes from the following conversation about an image. Output only a JSON object with these attributes:
- colors: [array of colors mentioned]
- shapes: [array of shapes mentioned]
- position: string describing spatial position
- size: string describing size
- direction: string describing direction
- relationships: [array of relationships between objects]
- actions: [array of actions mentioned]
- categories: [array of objects/items mentioned]

Use "N/A" for any attribute not mentioned in the conversation.

Example Input:
Q: What is in the image?
A: A man is standing next to a car.
Q: What is the color of the car?
A: The car is red.

Example Output:
{
  "colors": ["red"],
  "shapes": ["N/A"],
  "position": "next to the man",
  "size": "N/A",
  "direction": "N/A",
  "relationships": ["man next to car"],
  "actions": ["standing"],
  "categories": ["man", "car"]
}

Conversation to analyze:
{text_input}
'''



@register()
def description_analysis(dataset: MMDataset,
    model_name: str = "Qwen/Qwen2.5-0.5B",
    batch_size: int = 1) -> Dict:
    """
    分析数据集中的所有对话内容，提取和统计各类属性信息。

    TEXT
    Args:
        dataset: MMDataset对象，包含图片路径和对话内容
        model_name: 使用的模型名称
        batch_size: 批处理大小

    Returns:
        Dict: 包含处理后的数据集
    """
    # 加载模型和分词器
    tokenizer, model = load_model(model_name)
    model.eval()

    # 存储所有解析后的结果
    all_parsed_results = []
    filtered_data = {}

    # 收集待处理数据
    all_data = []
    print("收集数据中...")
    for item in dataset:
        image_path = item.get("image", "")
        conversations = item.get("conversations", [])
        
        # 拼接所有问答对为完整的对话内容
        full_caption = ""
        for conversation in conversations:
            question, answer = conversation
            # 清理问题中的 <image> 标签（如果有的话）
            question = question.replace('<image>\n', '').replace('\n<image>', '').replace('<image>', '')
            full_caption += f"Question: {question.strip()}\nAnswer: {answer.strip()}\n"

        
        # 构造完整的 prompt
        full_prompt = ANALYSIS_PROMPT.replace("{text_input}", full_caption)
        
        all_data.append({
            'image_path': image_path,
            'prompt': full_prompt
        })

    total_samples = len(all_data)
    num_batches = (total_samples + batch_size - 1) // batch_size
    print(f"总共收集到 {total_samples} 条数据，将分成 {num_batches} 个批次处理")

    # 批量处理数据
    for batch_idx in tqdm(range(num_batches), desc="处理批次"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_samples)
        batch_data = all_data[start_idx:end_idx]
        batch_prompts = [item['prompt'] for item in batch_data]
        batch_image_paths = [item['image_path'] for item in batch_data]

        try:
            # 模型推理
            input_features = tokenizer(batch_prompts, return_tensors="pd", padding=True)
            with paddle.no_grad():
                outputs = model.generate(**input_features, max_length=512)

            # 解码输出
            decoded_outputs = tokenizer.batch_decode(outputs[0], skip_special_tokens=True)
            print(f"image_path:{batch_image_paths}")
            print(f"prompt:{batch_prompts}")
            print(f"Decoded Outputs for Batch {batch_idx}:\n{decoded_outputs}")

            # 处理当前批次结果
            for idx, analysis_result in enumerate(decoded_outputs):
                parsed_result = parse_model_output(analysis_result)
                print(f"Parsed Result for Input {idx} in Batch {batch_idx}: {parsed_result}")
                if parsed_result:
                    all_parsed_results.append(parsed_result)

                    # 更新过滤后的数据集
                    current_item = batch_data[idx]
                    image_path = current_item['image_path']
                    if image_path not in filtered_data:
                        filtered_data[image_path] = {
                            "image": image_path,
                            "conversations": conversations
                        }
            print("*"*50)

        except Exception as e:
            print(f"处理批次 {batch_idx + 1}/{num_batches} 时出错:")
            print(f"错误信息: {str(e)}")
            print("-" * 50)
            continue

    # 清理并统计结果
    cleaned_info = clean_and_count(all_parsed_results)

    # 输出统计信息
    print("\n=== 属性统计结果 ===")
    for category, counts in cleaned_info.items():
        print(f"\n{category}:")
        for item, count in counts.most_common(10):  # 显示每个类别的前10个最常见项
            print(f"  {item}: {count}")

    # 输出处理总结
    print("\n=== 处理总结 ===")
    print(f"总问答对数量: {total_samples}")
    print(f"成功解析数量: {len(all_parsed_results)}")
    print(f"涉及的图片数量: {len(filtered_data)}")

    return MMDataset(list(filtered_data.values()))