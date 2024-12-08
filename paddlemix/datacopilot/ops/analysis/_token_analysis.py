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






from collections import Counter
import jieba
import matplotlib.pyplot as plt
from paddlenlp.transformers import AutoTokenizer
from paddlemix.datacopilot.core import MMDataset, register
from typing import Dict, Any
from functools import partial

# 初始化 Tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")


def decode_token_ids(token_counts: Counter) -> Counter:
    """解码 Token ID 为原始文字"""
    decoded_counts = Counter()
    for token_id, count in token_counts.items():
        decoded_text = tokenizer.decode([token_id]).strip()
        decoded_counts[decoded_text] += count
    return decoded_counts


def plot_token_distribution(token_counts: Counter, title: str, output_path: str) -> None:
    most_common = token_counts.most_common(20)
    
    # 如果没有任何 token 被统计，跳过绘图，或者使用默认图像提示
    if not most_common:
        print(f"Warning: No tokens to plot for {title}")
        return

    tokens, frequencies = zip(*most_common)

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(tokens)), frequencies, tick_label=tokens)
    plt.xticks(rotation=45, fontsize=10)
    plt.xlabel("Decoded Tokens")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def analyze_tokens(item: Dict[str, Any]) -> Dict[str, Any]:
    human_tokens = []
    assistant_tokens = []

    # 遍历 conversations，区分 human 和 assistant 的对话
    for conv in item["conversations"]:
        tokens = tokenizer(conv["value"], truncation=True, return_tensors="pd", use_fast=True)["input_ids"].numpy().flatten()
        if conv["from"] == "human":
            human_tokens.extend(tokens)
        elif conv["from"] == "gpt":  # 将 "gpt" 修改为 "assistant"
            assistant_tokens.extend(tokens)


    human_token_counts = Counter(human_tokens)
    assistant_token_counts = Counter(assistant_tokens)

    return {
        "human": {
            "total_tokens": len(human_tokens),
            "token_distribution": human_token_counts,
        },
        "assistant": {
            "total_tokens": len(assistant_tokens),
            "token_distribution": assistant_token_counts,
        },
        "overall": {
            "total_tokens": len(human_tokens) + len(assistant_tokens),
            "human_ratio": len(human_tokens) / (len(human_tokens) + len(assistant_tokens)),
            "assistant_ratio": len(assistant_tokens) / (len(human_tokens) + len(assistant_tokens)),
        }
    }

# 使用 MMDataset 的 map 和其他功能进行分析
@register()
def run_token_analysis(dataset: MMDataset) -> Dict:
    # Step 1: 直接进行 token 分析
    token_results = dataset.map(analyze_tokens, max_workers=16, progress=True)
    # Step 2: 收集结果
    human_token_distribution = Counter()
    assistant_token_distribution = Counter()

    # 遍历 token_results 列表中的每一项
    for result in token_results:  # 这里修改为遍历每个元素
        # 更新 human 和 assistant 的 token 分布
        human_token_distribution.update(result["human"]["token_distribution"])
        assistant_token_distribution.update(result["assistant"]["token_distribution"])

    # Step 3: 解码并绘制结果
    human_decoded_counts = decode_token_ids(human_token_distribution)  # 直接传递 human_token_distribution
    assistant_decoded_counts = decode_token_ids(assistant_token_distribution)  # 直接传递 assistant_token_distribution

    # 绘制 token 分布
    plot_token_distribution(human_decoded_counts, "Human Token Distribution", "human_token_distribution.png")
    plot_token_distribution(assistant_decoded_counts, "Assistant Token Distribution", "assistant_token_distribution.png")

    return {
        "token_analysis": {
            "human": human_decoded_counts,
            "assistant": assistant_decoded_counts,
        }
    }
