import matplotlib.pyplot as plt
import os
import numpy as np
import json

def plot_data_statistics(data_statistics, output_dir):
    """绘制 Data Statistics 图，显示会话数按 12 份划分的柱状图，并在右侧添加统计信息"""
    total_records = data_statistics['total_records']
    unique_images = data_statistics['unique_images']
    total_conversations = data_statistics['total_conversations']
    max_conversations = data_statistics['max_conversations']
    min_conversations = data_statistics['min_conversations']
    avg_conversations = data_statistics['avg_conversations']
    valid_items = data_statistics.get("valid_items", [])

    # 获取所有有效项的会话数
    conversation_counts = [
        len(item.get("conversations", [])) for item in valid_items
    ]

    # 计算第5百分位数和第95百分位数
    lower_percentile = np.percentile(conversation_counts, 5)
    upper_percentile = np.percentile(conversation_counts, 95)
    
    # 根据第5和第95百分位数，计算剩余区间
    lower_range_counts = [count for count in conversation_counts if count < lower_percentile]
    upper_range_counts = [count for count in conversation_counts if count > upper_percentile]
    middle_range_counts = [count for count in conversation_counts if lower_percentile <= count <= upper_percentile]
    
    # 分别计算这三部分的频率
    num_bins_middle = 8
    bins_middle = np.linspace(lower_percentile, upper_percentile, num_bins_middle + 1)
    
    # 创建标签
    bin_labels = [f"< {int(lower_percentile)}"] + [f'{int(bins_middle[i])} to {int(bins_middle[i+1])}' for i in range(num_bins_middle)] + [f"> {int(upper_percentile)}"]
    
    # 计算每个区间的频率
    conversation_freq = [0] * (num_bins_middle + 2)
    for count in lower_range_counts:
        conversation_freq[0] += 1
    for count in upper_range_counts:
        conversation_freq[-1] += 1
    for count in middle_range_counts:
        for i in range(num_bins_middle):
            if bins_middle[i] <= count < bins_middle[i + 1]:
                conversation_freq[i + 1] += 1
                break
    
    # 创建一个宽度更大的图形，分为两部分
    fig = plt.figure(figsize=(15, 6))
    
    # 左侧绘制柱状图
    ax1 = fig.add_subplot(121)
    ax1.bar(bin_labels, conversation_freq, color='skyblue', edgecolor='black', alpha=0.7)
    ax1.set_title("Data Statistics: Conversations by Range")
    ax1.set_xlabel("Conversations Range")
    ax1.set_ylabel("Frequency")
    ax1.tick_params(axis='x', rotation=45)
    
    # 右侧添加统计信息
    ax2 = fig.add_subplot(122, facecolor='white')
    ax2.axis('off')
    
    # 定义统计信息文本
    stats_text = [
        f"Total Records: {total_records}",
        f"Unique Images: {unique_images}",
        f"Total Conversations: {total_conversations}",
        f"Max Conversations: {max_conversations}",
        f"Min Conversations: {min_conversations}",
        f"Avg Conversations: {avg_conversations:.2f}"
    ]
    
    # 在右侧白色画布上添加文本
    for i, text in enumerate(stats_text):
        ax2.text(0.1, 0.9 - i*0.1, text, fontsize=12, ha='left', va='center')
    
    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(f"{output_dir}/00_data_statistics.png")
    plt.close()


import matplotlib.pyplot as plt
import numpy as np

def plot_field_distribution(field_distribution, output_dir):
    """绘制 Field Distribution 图"""
    human_message_count = field_distribution['human_message_count']
    assistant_message_count = field_distribution['assistant_message_count']
    mismatched_language_pairs_count = field_distribution['mismatched_language_pairs_count']

    # 语言分布，取前10
    languages_distribution = field_distribution['languages_distribution']
    sorted_languages = sorted(languages_distribution.items(), key=lambda x: x[1], reverse=True)[:10]
    languages, language_counts = zip(*sorted_languages)

    # 创建一个宽图，分为两部分
    fig = plt.figure(figsize=(15, 6))

    # 左侧绘制语言分布图
    ax1 = fig.add_subplot(121)
    ax1.bar(languages, language_counts, color='lightgreen')
    ax1.set_title("Language Distribution (Top 10)")
    ax1.set_xlabel("Language")
    ax1.set_ylabel("Count")
    ax1.tick_params(axis='x', rotation=45)

    # 右侧添加统计信息
    ax2 = fig.add_subplot(122, facecolor='white')
    ax2.axis('off')

    # 定义统计信息文本
    stats_text = [
        f"Human Message Count: {human_message_count}",
        f"Assistant Message Count: {assistant_message_count}",
        f"Mismatched Language Pairs: {mismatched_language_pairs_count}"
    ]
    
    # 在右侧白色画布上添加文本
    for i, text in enumerate(stats_text):
        ax2.text(0.1, 0.9 - i*0.1, text, fontsize=12, ha='left', va='center')

    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(f"{output_dir}/01_field_distribution.png")
    plt.close()



def plot_image_path_distribution(validation_result, output_dir):
    """绘制图片路径分布图，并显示统计信息"""
    total_images = validation_result['total_images']
    missing_images = validation_result['missing_images']
    path_distribution = validation_result['path_distribution']

    # 获取所有路径和对应的计数
    paths, path_counts = zip(*path_distribution.items())

    # 创建一个宽图，分为两部分
    fig = plt.figure(figsize=(15, 8))

    # 左侧绘制路径分布图
    ax1 = fig.add_subplot(121)
    ax1.bar(paths, path_counts, color='lightblue')
    ax1.set_title("Image Path Distribution")
    ax1.set_xlabel("Image Path")
    ax1.set_ylabel("Image Count")
    ax1.tick_params(axis='x', rotation=90)  # 旋转X轴标签，以便显示更多路径

    # 右侧添加统计信息
    ax2 = fig.add_subplot(122, facecolor='white')
    ax2.axis('off')

    # 定义统计信息文本
    stats_text = [
        f"Total Images: {total_images}",
        f"Missing Images: {missing_images}",
    ]
    
    # 在右侧白色画布上添加文本
    for i, text in enumerate(stats_text):
        ax2.text(0.1, 0.9 - i*0.1, text, fontsize=12, ha='left', va='center')

    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(f"{output_dir}/02_image_path_distribution.png")
    plt.close()



def plot_anomaly_statistics(anomaly_results, output_dir):
    """绘制缺少字段和空对话的统计图，并保存异常项的JSON"""
    missing_field_count = anomaly_results['missing_field_count']
    empty_conversation_count = anomaly_results['empty_conversation_count']

    # 创建一个柱状图，显示异常项数量
    labels = ['Missing Fields', 'Empty Conversations']
    counts = [missing_field_count, empty_conversation_count]

    # 创建图表
    plt.figure(figsize=(8, 5))
    plt.bar(labels, counts, color=['lightcoral', 'lightblue'])
    plt.title("Anomaly Statistics")
    plt.xlabel("Anomaly Type")
    plt.ylabel("Count")

    # 保存图表
    plt.tight_layout()
    plt.savefig(f"{output_dir}/03_anomaly_statistics.png")
    plt.close()




import matplotlib.pyplot as plt

def plot_token_distribution(token_analysis, role, output_dir):
    """绘制高频词和低频词的分布图"""
    high_freq_tokens = token_analysis['high_freq_tokens']
    low_freq_tokens = token_analysis['low_freq_tokens']

    # 提取高频词和低频词
    high_tokens, high_counts = zip(*high_freq_tokens.items())
    low_tokens, low_counts = zip(*low_freq_tokens.items())

    # 创建一个新的图形
    plt.figure(figsize=(12, 6))

    # 高频词
    plt.subplot(121)
    plt.bar(high_tokens, high_counts, color='green')
    plt.title(f"{role} High Frequency Tokens")
    plt.xlabel("Token")
    plt.ylabel("Count")
    plt.xticks(rotation=45)

    # 低频词
    plt.subplot(122)
    plt.bar(low_tokens, low_counts, color='red')
    plt.title(f"{role} Low Frequency Tokens")
    plt.xlabel("Token")
    plt.ylabel("Count")
    plt.xticks(rotation=45)

    # 调整布局
    plt.tight_layout()

    # 保存图像
    plt.savefig(f"{output_dir}/04_{role}_token_analysis.png")
    plt.close()


def visualize_results(results, output_dir, analysis_flags):
    """统一绘制所有分析结果，根据传入的标志控制绘制哪些分析结果"""
    
    # Data Statistics: 如果设置了 data_statistics 标志，则绘制数据统计图
    if analysis_flags.get("data_statistics", False):
        plot_data_statistics(results["data_statistics"], output_dir)
    
    # Field Distribution: 如果设置了 field_distribution 标志，则绘制字段分布图
    if analysis_flags.get("field_distribution", False):
        plot_field_distribution(results["field_distribution"], output_dir)

    # Path Validation: 如果设置了 path_validation 标志，则绘制图片路径验证图
    if analysis_flags.get("path_validation", False):
        plot_image_path_distribution(results["path_validation"], output_dir)
    
    # Path Validation & Anomaly Detection: 如果设置了 anomaly_detection 标志，则绘制异常检测统计图
    if analysis_flags.get("anomaly_detection", False):
        plot_anomaly_statistics(results["anomaly_detection"], output_dir)

    # Token Analysis: Human: 如果设置了 token_analysis 标志且包含 "human" 数据，则绘制人类的 Token 分布图
    if analysis_flags.get("token_analysis", False) and "human" in results.get("token_analysis", {}):
        plot_token_distribution(results["token_analysis"]["human"], "human", output_dir)

    # Token Analysis: Assistant: 如果设置了 token_analysis 标志且包含 "assistant" 数据，则绘制助手的 Token 分布图
    if analysis_flags.get("token_analysis", False) and "assistant" in results.get("token_analysis", {}):
        plot_token_distribution(results["token_analysis"]["assistant"], "assistant", output_dir)
