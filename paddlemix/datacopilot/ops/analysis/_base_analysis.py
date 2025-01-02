from collections import Counter
import matplotlib.pyplot as plt
import fasttext
import os
from paddlenlp.transformers import AutoTokenizer
from paddlemix.datacopilot.core import MMDataset, register, ParallelMode
from functools import partial
from typing import Dict, Any
import json

from ..visualize._analysis_plot import visualize_results

# 初始化 FastText 语言检测模型和 Tokenizer，wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
FASTTEXT_MODEL_PATH = "/home/lizhijun/llm/PaddleMix/lid.176.bin" 
lang_model = fasttext.load_model(FASTTEXT_MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")


def detect_language(text: str) -> str:
    """使用 FastText 进行语言检测"""
    try:
        prediction = lang_model.predict(text.strip(), k=1)
        return prediction[0][0].replace("__label__", "")  # 返回语言代码
    except Exception:
        return "unknown"


def save_to_json(data, filename):
    """将数据保存到JSON文件"""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def count_data_statistics(dataset: MMDataset) -> Dict:
    """统计数据集的基本数量信息，包括异常数据"""
    
    # 直接访问 items 并进行验证
    valid_items = []
    invalid_count = 0

    for item in dataset.items:
        # 这里需要根据你的实际验证方式对 item 进行检查
        if "image" in item and isinstance(item.get("conversations"), list) and item["conversations"]:
            valid_items.append(item)
        else:
            invalid_count += 1

    # 统计有效项的对话数
    conversation_counts = [
        len(item.get("conversations", [])) for item in valid_items
    ]
    
    total_conversations = sum(conversation_counts)  # 所有对话的总数
    max_conversations = max(conversation_counts, default=0)  # 最大对话数
    min_conversations = min(conversation_counts, default=0)  # 最小对话数
    avg_conversations = total_conversations / len(conversation_counts) if conversation_counts else 0  # 平均对话数

    unique_images = len(set(item.get("image", None) for item in valid_items if "image" in item))

    return {
        "total_records": len(dataset),  # 数据集总记录数
        "unique_images": unique_images,  # 唯一图片数量
        "total_conversations": total_conversations,  # 总对话数
        "max_conversations": max_conversations,  # 最大对话数
        "min_conversations": min_conversations,  # 最小对话数
        "avg_conversations": avg_conversations,  # 平均对话数
        "invalid_item_count": invalid_count,  # 无效数据数量
        "valid_items": valid_items,  # 有效数据项
    }



def analyze_field_distribution(dataset: MMDataset) -> Dict:
    """分析字段分布信息"""
    human_msgs, assistant_msgs = [], []
    languages = Counter()
    mismatched_language_pairs = 0

    mismatched_pairs = []  # 存储不匹配语言的问答对

    def process_conversation(item):
        nonlocal mismatched_language_pairs
        human_lang = assistant_lang = None  # 确保每次循环都初始化

        for i, conv in enumerate(item.get("conversations", [])):
            # Assuming the conversation format is now text-related like bounding box, description etc.
            # Modify for specific requirements like handling bounding boxes or descriptions
            conv_text = conv[0]  # Conversation text
            human_msgs.append(conv_text)
            human_lang = detect_language(conv_text)

            # Assuming the assistant response is in the second part of the tuple
            if len(conv) > 1:
                assistant_msgs.append(conv[1])
                assistant_lang = detect_language(conv[1])

                if human_lang != 'unknown' and assistant_lang != 'unknown':
                    if human_lang != assistant_lang:
                        mismatched_language_pairs += 1
                        mismatched_pairs.append({
                            "conversation_id": item.get("id"),
                            "human_message": conv_text,
                            "human_language": human_lang,
                            "assistant_message": conv[1],
                            "assistant_language": assistant_lang
                        })

                languages[human_lang] += 1
                languages[assistant_lang] += 1

    dataset.map(process_conversation, max_workers=8, mode=ParallelMode.THREAD, progress=True)

    return {
        "human_message_count": len(human_msgs),
        "assistant_message_count": len(assistant_msgs),
        "mismatched_language_pairs_count": mismatched_language_pairs,
        "languages_distribution": dict(languages),
    }


def validate_image_paths(dataset: MMDataset) -> Dict:
    """验证图片路径的分布和文件存在性"""
    def get_image_path(item):
        return item.get("image", None)

    all_paths = dataset.map(get_image_path, max_workers=8, mode=ParallelMode.THREAD, progress=True)
    all_paths = [path for path in all_paths if path is not None]
    missing_paths = [path for path in all_paths if not os.path.exists(path)]

    path_distribution = Counter(os.path.dirname(path) for path in all_paths)

    return {
        "total_images": len(all_paths),
        "missing_images": len(missing_paths),
        "path_distribution": dict(path_distribution),
    }


def detect_anomalies(dataset: MMDataset, json_output) -> Dict:
    """检测数据集中的异常项"""
    def check_anomalies(item):
        anomalies = {}
        if not all(key in item for key in ["image", "conversations"]):
            anomalies["missing_fields"] = True
        if "conversations" in item and (
            not item["conversations"] or any(not conv[0].strip() for conv in item["conversations"])
        ):
            anomalies["empty_conversations"] = True
        return anomalies

    anomaly_results = dataset.map(check_anomalies, max_workers=8, mode=ParallelMode.THREAD)
    missing_fields = [item for item, result in zip(dataset, anomaly_results) if result.get("missing_fields")]
    empty_conversations = [item for item, result in zip(dataset, anomaly_results) if result.get("empty_conversations")]

    save_to_json(missing_fields, f"{json_output}/03_missing_fields.json")
    save_to_json(empty_conversations, f"{json_output}/03_empty_conversations.json")

    return {
        "missing_field_count": len(missing_fields),
        "empty_conversation_count": len(empty_conversations),
    }


def decode_token_ids(token_counts: Counter) -> Counter:
    """解码 Token ID 为原始文字"""
    decoded_counts = Counter()
    for token_id, count in token_counts.items():
        decoded_text = tokenizer.decode([token_id]).strip()
        decoded_counts[decoded_text] += count
    return decoded_counts


def run_token_analysis(dataset: MMDataset) -> Dict:
    """统一处理 Token 分析"""
    token_results = dataset.map(analyze_tokens, max_workers=16, progress=True)
    
    human_token_distribution = Counter()
    assistant_token_distribution = Counter()
    total_human_tokens = 0
    total_assistant_tokens = 0
    human_tokens = []
    assistant_tokens = []

    for result in token_results:
        total_human_tokens += result["human"]["total_tokens"]
        total_assistant_tokens += result["assistant"]["total_tokens"]
        
        human_token_distribution.update(result["human"]["token_distribution"])
        assistant_token_distribution.update(result["assistant"]["token_distribution"])
        
        human_tokens.extend(result["human"]["token_distribution"].elements())
        assistant_tokens.extend(result["assistant"]["token_distribution"].elements())

    human_decoded_counts = decode_token_ids(human_token_distribution)
    assistant_decoded_counts = decode_token_ids(assistant_token_distribution)

    num_common_tokens = 20
    human_high_freq_tokens = Counter(human_tokens).most_common(num_common_tokens)
    assistant_high_freq_tokens = Counter(assistant_tokens).most_common(num_common_tokens)
    human_low_freq_tokens = Counter(human_tokens).most_common()[-num_common_tokens:]
    assistant_low_freq_tokens = Counter(assistant_tokens).most_common()[-num_common_tokens:]

    human_high_freq_decoded = decode_token_ids(Counter(dict(human_high_freq_tokens)))
    assistant_high_freq_decoded = decode_token_ids(Counter(dict(assistant_high_freq_tokens)))
    human_low_freq_decoded = decode_token_ids(Counter(dict(human_low_freq_tokens)))
    assistant_low_freq_decoded = decode_token_ids(Counter(dict(assistant_low_freq_tokens)))

    return {
        "human": {
            "total_tokens": total_human_tokens,
            "high_freq_tokens": human_high_freq_decoded,
            "low_freq_tokens": human_low_freq_decoded
        },
        "assistant": {
            "total_tokens": total_assistant_tokens,
            "high_freq_tokens": assistant_high_freq_decoded,
            "low_freq_tokens": assistant_low_freq_decoded
        }
    }


def analyze_tokens(item: Dict[str, Any]) -> Dict[str, Any]:
    """分析每个会话中的 token"""
    human_tokens = []
    assistant_tokens = []
    all_human_token_ids = []
    all_assistant_token_ids = []

    for conv in item["conversations"]:
        tokens = tokenizer(conv[0], truncation=True, return_tensors="pd", use_fast=True)["input_ids"].numpy().flatten()
        if conv[0]:  # Add logic based on your conversation format
            human_tokens.extend(tokens)
            all_human_token_ids.extend(tokens)
        if len(conv) > 1:  # If there’s an assistant response
            assistant_tokens.extend(tokens)
            all_assistant_token_ids.extend(tokens)

    human_token_counts = Counter(human_tokens)
    assistant_token_counts = Counter(assistant_tokens)

    total_human_tokens = len(human_tokens)
    total_assistant_tokens = len(assistant_tokens)

    return {
        "human": {
            "total_tokens": total_human_tokens,
            "token_distribution": human_token_counts,
        },
        "assistant": {
            "total_tokens": total_assistant_tokens,
            "token_distribution": assistant_token_counts,
        }
    }


@register()
def run_base_analysis(dataset: MMDataset, analysis_flags: Dict[str, bool] = None, output_dir="output_directory") -> Dict:
    """统一调用所有分析功能，并根据分析标志控制执行的分析"""
    if analysis_flags is None:
        analysis_flags = {
            "data_statistics": True,
            "field_distribution": True,
            "path_validation": True,
            "anomaly_detection": True,
            "token_analysis": True
        }

    results = {}

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if analysis_flags["data_statistics"]:
        print("Start statistical analysis...")
        results["data_statistics"] = count_data_statistics(dataset)

    if analysis_flags["field_distribution"]:
        print("Start field distribution analysis...")
        results["field_distribution"] = analyze_field_distribution(dataset)

    if analysis_flags["path_validation"]:
        print("Start image path validation...")
        results["path_validation"] = validate_image_paths(dataset)

    if analysis_flags["anomaly_detection"]:
        print("Start anomaly detection...")
        results["anomaly_detection"] = detect_anomalies(dataset, json_output=output_dir)

    if analysis_flags["token_analysis"]:
        print("Start token analysis...")
        results["token_analysis"] = run_token_analysis(dataset)

    print("All analysis is completed and the results are being visualized")
    visualize_results(results, output_dir, analysis_flags)

    return results
