from paddlenlp.transformers import AutoTokenizer, AutoModelForCausalLM
from paddlemix.datacopilot.core import MMDataset, register
from tqdm import tqdm
from typing import Dict, List
import json
import paddle
import re

def load_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype="float16")
    return tokenizer, model

def extract_rating(text: str) -> int:
    """
    从文本中提取Total rating后的评分
    返回-1表示没有找到有效评分
    """
    match = re.search(r'Total rating:\s*([1-4])(?:\D|$)', text)
    if match:
        return int(match.group(1))
    return -1

prompt_template = """
You will be given a user_question and system_answer couple. Your task is to provide a 'total rating' scoring how well the system_answer answers the user concerns expressed in the user_question. Give your answer on a scale of 1 to 4, where 1 means that the system_answer is not helpful at all, and 4 means that the system_answer completely and helpfully addresses the user_question. Here is the scale you should use to build your answer: 
1: The system_answer is terrible: completely irrelevant to the question asked, or very partial 
2: The system_answer is mostly not helpful: misses some key aspects of the question 
3: The system_answer is mostly helpful: provides support, but still could be improved 
4: The system_answer is excellent: relevant, direct, detailed, and addresses all the concerns raised in the question 
Provide your feedback as follows: 
Feedback::: 
Evaluation: (your rationale for the rating, as a text) 
Total rating: (your rating, as a number between 1 and 4) 
You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer. 
Now here are the question and answer. 
Question: {question} 
Answer: {answer} 
Provide your feedback.
"""

@register()
def gpt_responses_judge(dataset: MMDataset, model_name: str = "Qwen/Qwen2.5-0.5B", batch_size: int = 1) -> Dict:
    tokenizer, model = load_model(model_name)
    model.eval()
    
    # 存储所有待处理的数据
    all_data = []  # 用于存储每个问答对及其相关信息
    total_pairs = 0
    valid_pairs = 0
    filtered_data = {}  # 用于按图片路径组织过滤后的数据

    print("收集数据中...")
    for item in dataset:
        image_path = item.get("image", "")
        conversations = item.get("conversations", [])
        
        # 处理每个conversation
        for conv in conversations:
            if isinstance(conv, list) and len(conv) == 2:
                total_pairs += 1
                question, answer = conv
                cleaned_question = question.strip()
                
                all_data.append({
                    'image_path': image_path,
                    'question': cleaned_question,
                    'answer': answer,
                    'prompt': prompt_template.format(question=cleaned_question, answer=answer)
                })

    total_samples = len(all_data)
    num_batches = (total_samples + batch_size - 1) // batch_size
    
    print(f"总共收集到 {total_samples} 条问答对，将分成 {num_batches} 个批次处理")

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
            for idx, eval_result in enumerate(decoded_outputs):
                
                rating = extract_rating(eval_result)
                current_item = batch_data[idx]
                print(f"当前问答对为:{current_item}, 得分为:{rating}")
                # 只保留评分大于等于3的问答对
                if rating >= 3:
                    valid_pairs += 1
                    image_path = current_item['image_path']
                    
                    # 如果这个图片路径还没有对应的数据，创建一个新的
                    if image_path not in filtered_data:
                        filtered_data[image_path] = {
                            "image": image_path,
                            "conversations": []
                        }
                    
                    # 添加当前的问答对到对应图片的conversations中
                    filtered_data[image_path]["conversations"].append([
                        current_item['question'],
                        current_item['answer']
                    ])

        except Exception as e:
            print(f"处理批次 {batch_idx + 1}/{num_batches} 时出错:")
            print(f"错误信息: {e}")
            print("-" * 50)
            continue

    # 转换filtered_data为列表格式
    final_dataset = list(filtered_data.values())

    print(f"处理完成:")
    print(f"总问答对数量: {total_pairs}")
    print(f"有效问答对数量 (评分>=3): {valid_pairs}")
    print(f"有效率: {(valid_pairs/total_pairs*100):.2f}%")
    print(f"涉及的图片数量: {len(filtered_data)}")

    return MMDataset(final_dataset)