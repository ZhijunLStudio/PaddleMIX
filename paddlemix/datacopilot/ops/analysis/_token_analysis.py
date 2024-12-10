# from collections import Counter
# import json
# import jieba
# import matplotlib.pyplot as plt
# from paddlenlp.transformers import AutoTokenizer
# import paddle
# from matplotlib import rcParams
# from matplotlib import font_manager
# from paddlemix.datacopilot.core import MMDataset, register
# from typing import Dict

# # 设置字体路径
# font_path = '/home/lizhijun/PaddleMIX-develop/PaddleNLP/font/SimHei.ttf'  

# # 手动添加字体到 matplotlib 字体管理器
# font_manager.fontManager.addfont(font_path)

# # 设置 matplotlib 使用 SimHei 字体
# plt.rcParams['font.family'] = 'SimHei'  # 使用 SimHei 字体
# rcParams['axes.unicode_minus'] = False  # 正常显示负号

# # 初始化分词器
# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

# def analyze_tokens(dataset: MMDataset) -> Dict:
#     """分析 Token 统计信息"""
#     human_tokens = []
#     assistant_tokens = []

#     for item in dataset:
#         for conv in item["conversations"]:
#             tokens = tokenizer(conv["value"], truncation=True, return_tensors="pd")["input_ids"].numpy().flatten()
#             if conv["from"] == "human":
#                 human_tokens.extend(tokens)
#             elif conv["from"] == "assistant":
#                 assistant_tokens.extend(tokens)

#     # 计算频率分布
#     human_token_counts = Counter(human_tokens)
#     assistant_token_counts = Counter(assistant_tokens)
    
#     # 计算总 token 数
#     human_total_tokens = len(human_tokens)
#     assistant_total_tokens = len(assistant_tokens)

#     return {
#         "human": {
#             "total_tokens": human_total_tokens,
#             "token_distribution": human_token_counts,
#         },
#         "assistant": {
#             "total_tokens": assistant_total_tokens,
#             "token_distribution": assistant_token_counts,
#         },
#         "overall": {
#             "total_tokens": human_total_tokens + assistant_total_tokens,
#             "human_ratio": human_total_tokens / (human_total_tokens + assistant_total_tokens),
#             "assistant_ratio": assistant_total_tokens / (human_total_tokens + assistant_total_tokens),
#         }
#     }

# def decode_token_ids(token_counts: Counter) -> Counter:
#     """解码 Token ID 为原始文字"""
#     decoded_counts = Counter()
#     for token_id, count in token_counts.items():
#         decoded_text = tokenizer.decode([token_id]).strip()
#         decoded_counts[decoded_text] += count
#     return decoded_counts

# def plot_token_distribution(token_counts: Counter, title: str, output_path: str) -> None:
#     """绘制 Token 分布图"""
#     most_common = token_counts.most_common(20)
#     tokens, frequencies = zip(*most_common)

#     plt.figure(figsize=(12, 6))
#     plt.bar(range(len(tokens)), frequencies, tick_label=tokens)
#     plt.xticks(rotation=45, fontsize=10)
#     plt.xlabel("Decoded Tokens")
#     plt.ylabel("Frequency")
#     plt.title(title)
#     plt.tight_layout()
#     plt.savefig(output_path)
#     plt.close()

# def plot_sentence_length_distribution(dataset: MMDataset, output_path: str) -> None:
#     """绘制句子长度分布图"""
#     lengths = []
#     for item in dataset:
#         for conv in item["conversations"]:
#             tokens = tokenizer(conv["value"], truncation=True, return_tensors="pd")["input_ids"].numpy().flatten()
#             lengths.append(len(tokens))
    
#     plt.figure(figsize=(10, 5))
#     plt.hist(lengths, bins=20, color='blue', alpha=0.7)
#     plt.xlabel("Sentence Length (Tokens)")
#     plt.ylabel("Frequency")
#     plt.title("Sentence Length Distribution")
#     plt.tight_layout()
#     plt.savefig(output_path)
#     plt.close()

# def analyze_field_distribution(dataset: MMDataset) -> Dict:
#     """分析字段分布信息"""
#     human_msgs = []
#     assistant_msgs = []
#     for item in dataset:
#         for conv in item["conversations"]:
#             if conv["from"] == "human":
#                 human_msgs.append(conv["value"])
#             elif conv["from"] == "assistant":
#                 assistant_msgs.append(conv["value"])

#     human_word_count = Counter(jieba.lcut(" ".join(human_msgs)))
#     assistant_word_count = Counter(jieba.lcut(" ".join(assistant_msgs)))

#     return {
#         "human_word_count": human_word_count.most_common(10),
#         "assistant_word_count": assistant_word_count.most_common(10)
#     }

# @register()
# def run_token_analysis(dataset: MMDataset) -> Dict:
#     """统一调用所有分析功能"""
#     results = {}

#     # 1. Token 统计分析
#     token_results = analyze_tokens(dataset)
#     results["token_analysis"] = token_results

#     # 2. 字段分布分析
#     field_dist = analyze_field_distribution(dataset)
#     results["field_distribution"] = field_dist

#     # 3. 绘制分布图
#     human_decoded_counts = decode_token_ids(token_results["human"]["token_distribution"])
#     assistant_decoded_counts = decode_token_ids(token_results["assistant"]["token_distribution"])

#     plot_token_distribution(human_decoded_counts, "Human Token Distribution", "human_token_distribution.png")
#     plot_token_distribution(assistant_decoded_counts, "Assistant Token Distribution", "assistant_token_distribution.png")
#     plot_sentence_length_distribution(dataset, "sentence_length_distribution.png")

#     return results






# from collections import Counter
# import jieba
# import matplotlib.pyplot as plt
# from paddlenlp.transformers import AutoTokenizer
# from paddlemix.datacopilot.core import MMDataset, register
# from typing import Dict, Any
# from functools import partial

# # 初始化 Tokenizer
# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")


# def decode_token_ids(token_counts: Counter) -> Counter:
#     """解码 Token ID 为原始文字"""
#     decoded_counts = Counter()
#     for token_id, count in token_counts.items():
#         decoded_text = tokenizer.decode([token_id]).strip()
#         decoded_counts[decoded_text] += count
#     return decoded_counts


# def plot_token_distribution(token_counts: Counter, title: str, output_path: str) -> None:
#     most_common = token_counts.most_common(20)
    
#     # 如果没有任何 token 被统计，跳过绘图，或者使用默认图像提示
#     if not most_common:
#         print(f"Warning: No tokens to plot for {title}")
#         return

#     tokens, frequencies = zip(*most_common)

#     plt.figure(figsize=(12, 6))
#     plt.bar(range(len(tokens)), frequencies, tick_label=tokens)
#     plt.xticks(rotation=45, fontsize=10)
#     plt.xlabel("Decoded Tokens")
#     plt.ylabel("Frequency")
#     plt.title(title)
#     plt.tight_layout()
#     plt.savefig(output_path)
#     plt.close()

# def analyze_tokens(item: Dict[str, Any]) -> Dict[str, Any]:
#     human_tokens = []
#     assistant_tokens = []

#     # 遍历 conversations，区分 human 和 assistant 的对话
#     for conv in item["conversations"]:
#         tokens = tokenizer(conv["value"], truncation=True, return_tensors="pd", use_fast=True)["input_ids"].numpy().flatten()
#         if conv["from"] == "human":
#             human_tokens.extend(tokens)
#         elif conv["from"] == "gpt":  # 将 "gpt" 修改为 "assistant"
#             assistant_tokens.extend(tokens)


#     human_token_counts = Counter(human_tokens)
#     assistant_token_counts = Counter(assistant_tokens)

#     return {
#         "human": {
#             "total_tokens": len(human_tokens),
#             "token_distribution": human_token_counts,
#         },
#         "assistant": {
#             "total_tokens": len(assistant_tokens),
#             "token_distribution": assistant_token_counts,
#         },
#         "overall": {
#             "total_tokens": len(human_tokens) + len(assistant_tokens),
#             "human_ratio": len(human_tokens) / (len(human_tokens) + len(assistant_tokens)),
#             "assistant_ratio": len(assistant_tokens) / (len(human_tokens) + len(assistant_tokens)),
#         }
#     }

# # 使用 MMDataset 的 map 和其他功能进行分析
# @register()
# def run_token_analysis(dataset: MMDataset) -> Dict:
#     # Step 1: 直接进行 token 分析
#     token_results = dataset.map(analyze_tokens, max_workers=16, progress=True)
#     # Step 2: 收集结果
#     human_token_distribution = Counter()
#     assistant_token_distribution = Counter()

#     # 遍历 token_results 列表中的每一项
#     for result in token_results:  # 这里修改为遍历每个元素
#         # 更新 human 和 assistant 的 token 分布
#         human_token_distribution.update(result["human"]["token_distribution"])
#         assistant_token_distribution.update(result["assistant"]["token_distribution"])

#     # Step 3: 解码并绘制结果
#     human_decoded_counts = decode_token_ids(human_token_distribution)  # 直接传递 human_token_distribution
#     assistant_decoded_counts = decode_token_ids(assistant_token_distribution)  # 直接传递 assistant_token_distribution

#     # 绘制 token 分布
#     plot_token_distribution(human_decoded_counts, "Human Token Distribution", "human_token_distribution.png")
#     plot_token_distribution(assistant_decoded_counts, "Assistant Token Distribution", "assistant_token_distribution.png")

#     return {
#         "token_analysis": {
#             "human": human_decoded_counts,
#             "assistant": assistant_decoded_counts,
#         }
#     }



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
FASTTEXT_MODEL_PATH = "/home/lizhijun/PaddleMIX-develop/lid.176.bin" 
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
    # 对数据集进行清洗并获取有效项
    valid_items = dataset.sanitize(progress=True).items
    invalid_count = len(dataset) - len(valid_items)

    # 计算有效项的对话数
    conversation_counts = [
        len(item.get("conversations", [])) for item in valid_items
    ]
    
    # 计算对话统计信息
    total_conversations = sum(conversation_counts)  # 所有对话的总数
    max_conversations = max(conversation_counts, default=0)  # 最大对话数
    min_conversations = min(conversation_counts, default=0)  # 最小对话数
    avg_conversations = total_conversations / len(conversation_counts) if conversation_counts else 0  # 平均对话数

    # 计算唯一图片数
    unique_images = len(set(item.get("image", None) for item in valid_items if "image" in item))

    # 返回所有统计信息，并包含 valid_items
    return {
        "total_records": len(dataset),
        "unique_images": unique_images,
        "total_conversations": total_conversations,
        "max_conversations": max_conversations,
        "min_conversations": min_conversations,
        "avg_conversations": avg_conversations,
        "invalid_item_count": invalid_count,
        "valid_items": valid_items,  # 添加 valid_items 作为统计信息的一部分
    }




def analyze_field_distribution(dataset: MMDataset) -> Dict:
    """分析字段分布信息"""
    human_msgs, assistant_msgs = [], []
    languages = Counter()
    mismatched_language_pairs = 0

    # 用于记录不匹配的问答对
    mismatched_pairs = []  # 存储不匹配语言的问答对

    def process_conversation(item):
        nonlocal mismatched_language_pairs
        human_lang = assistant_lang = None  # 确保每次循环都初始化

        for i, conv in enumerate(item.get("conversations", [])):
            if conv["from"] == "human":
                human_msgs.append(conv["value"])
                human_lang = detect_language(conv["value"])  # 检测人类消息语言
            elif conv["from"] == "gpt":
                assistant_msgs.append(conv["value"])
                assistant_lang = detect_language(conv["value"])  # 检测助手消息语言

                # 如果human消息和assistant消息的语言都不是unknown，进行比较
                if human_lang != 'unknown' and assistant_lang != 'unknown':
                    # 如果语言不同，统计为不匹配
                    if human_lang != assistant_lang:
                        mismatched_language_pairs += 1
                        # 确保人类消息和助手消息不重复
                        mismatched_pairs.append({
                            "conversation_id": item.get("id"),
                            "human_message": item["conversations"][i-1]["value"],  # 获取上一条人类消息
                            "human_language": human_lang,
                            "assistant_message": conv["value"],  # 获取当前的助手消息
                            "assistant_language": assistant_lang
                        })

                # 更新语言分布
                languages[human_lang] += 1
                languages[assistant_lang] += 1

    # 通过并行处理对数据集进行分析
    dataset.map(process_conversation, max_workers=8, mode=ParallelMode.THREAD, progress=True)

    return {
        "human_message_count": len(human_msgs),
        "assistant_message_count": len(assistant_msgs),
        "mismatched_language_pairs_count": mismatched_language_pairs,
        "languages_distribution": dict(languages),
        # "mismatched_language_pairs": mismatched_pairs  # 返回不匹配的问答对
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
        if not all(key in item for key in ["id", "image", "conversations"]):
            anomalies["missing_fields"] = True
        if "conversations" in item and (
            not item["conversations"] or any(not conv["value"].strip() for conv in item["conversations"])
        ):
            anomalies["empty_conversations"] = True
        return anomalies

    anomaly_results = dataset.map(check_anomalies, max_workers=8, mode=ParallelMode.THREAD)
    missing_fields = [item for item, result in zip(dataset, anomaly_results) if result.get("missing_fields")]
    empty_conversations = [item for item, result in zip(dataset, anomaly_results) if result.get("empty_conversations")]

    # 保存JSON文件
    save_to_json(missing_fields, f"{json_output}/03_missing_fields.json")
    save_to_json(empty_conversations, f"{json_output}/03_empty_conversations.json")

    # 返回统计信息和异常项
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
    
    # 初始化用于收集数据的容器
    human_token_distribution = Counter()
    assistant_token_distribution = Counter()
    total_human_tokens = 0
    total_assistant_tokens = 0
    human_tokens = []
    assistant_tokens = []

    # 遍历 token_results 收集统计信息
    for result in token_results:
        # 只更新 total_tokens 和 token 分布统计，而不保存完整的 token_distribution
        total_human_tokens += result["human"]["total_tokens"]
        total_assistant_tokens += result["assistant"]["total_tokens"]
        
        # 仅更新 token 分布的统计
        human_token_distribution.update(result["human"]["token_distribution"])
        assistant_token_distribution.update(result["assistant"]["token_distribution"])
        
        # 收集所有的 token 用于后续高频和低频词分析
        human_tokens.extend(result["human"]["token_distribution"].elements())
        assistant_tokens.extend(result["assistant"]["token_distribution"].elements())

    # 解码并绘制结果
    human_decoded_counts = decode_token_ids(human_token_distribution)
    assistant_decoded_counts = decode_token_ids(assistant_token_distribution)

    # 计算高频词和低频词的分布
    num_common_tokens = 20
    human_high_freq_tokens = Counter(human_tokens).most_common(num_common_tokens)
    assistant_high_freq_tokens = Counter(assistant_tokens).most_common(num_common_tokens)
    human_low_freq_tokens = Counter(human_tokens).most_common()[-num_common_tokens:]
    assistant_low_freq_tokens = Counter(assistant_tokens).most_common()[-num_common_tokens:]

    # 解码高频词和低频词
    human_high_freq_decoded = decode_token_ids(Counter(dict(human_high_freq_tokens)))
    assistant_high_freq_decoded = decode_token_ids(Counter(dict(assistant_high_freq_tokens)))
    human_low_freq_decoded = decode_token_ids(Counter(dict(human_low_freq_tokens)))
    assistant_low_freq_decoded = decode_token_ids(Counter(dict(assistant_low_freq_tokens)))

    # 返回结果
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

    # 遍历 conversations，区分 human 和 assistant 的对话
    for conv in item["conversations"]:
        tokens = tokenizer(conv["value"], truncation=True, return_tensors="pd", use_fast=True)["input_ids"].numpy().flatten()
        if conv["from"] == "human":
            human_tokens.extend(tokens)
            all_human_token_ids.extend(tokens)
        elif conv["from"] == "gpt":
            assistant_tokens.extend(tokens)
            all_assistant_token_ids.extend(tokens)

    # 计算 human 和 assistant 的 token 数量
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


    # 如果输出目录不存在，创建该目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    # 数据统计分析
    if analysis_flags["data_statistics"]:
        print("Start statistical analysis...")
        results["data_statistics"] = count_data_statistics(dataset)

    # 字段分布分析
    if analysis_flags["field_distribution"]:
        print("Start field distribution analysis...")
        results["field_distribution"] = analyze_field_distribution(dataset)

    # 图片路径验证
    if analysis_flags["path_validation"]:
        print("Start image path validation...")
        results["path_validation"] = validate_image_paths(dataset)

    # 异常项检测
    if analysis_flags["anomaly_detection"]:
        print("Start anomaly detection...")
        results["anomaly_detection"] = detect_anomalies(dataset, json_output=output_dir)

    # Token 分析
    if analysis_flags["token_analysis"]:
        print("Start token analysis...")
        results["token_analysis"] = run_token_analysis(dataset)

    # 可视化所有结果
    print("All analysis is completed and the results are being visualized")
    visualize_results(results, output_dir, analysis_flags)  # 输出目录根据需要设置

    return results
