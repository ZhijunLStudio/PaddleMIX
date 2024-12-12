import numpy as np
from paddlemix.datacopilot.core import MMDataset


def conversation_percentage_filter(dataset: MMDataset, min_percentile: float, max_percentile: float) -> MMDataset:
    """
    根据对话数的百分位数过滤数据集项。
    
    参数:
        dataset (MMDataset): 要过滤的数据集。
        min_percentile (float): 最小百分位数（例如 0 表示第 0 百分位数）。
        max_percentile (float): 最大百分位数（例如 95 表示第 95 百分位数）。
        
    返回:
        MMDataset: 过滤后的数据集。
    """
    print("开始过滤对话数百分位数")
    if not (0 <= min_percentile <= 100 and 0 <= max_percentile <= 100):
        raise ValueError("百分位数范围应在 0 到 100 之间。")

    # 统计对话数
    conversation_counts = np.array([
        len(item.get("conversations", [])) for item in dataset.items
    ])
    
    # 计算百分位数
    min_threshold = np.percentile(conversation_counts, min_percentile)
    max_threshold = np.percentile(conversation_counts, max_percentile)

    print(f"过滤对话数范围: {min_threshold} 到 {max_threshold}")

    # 过滤数据集
    filtered_items = [
        item for item, count in zip(dataset.items, conversation_counts)
        if min_threshold <= count <= max_threshold
    ]

    return filtered_items



