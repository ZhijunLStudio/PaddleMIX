from typing import Optional
from ...core import T, MMDataset, register
from functools import partial


def is_max_line_length_valid(item, min_length: int = 10, max_length: float = float('inf')) -> bool:
    """
    检查会话的最大行长度是否在指定范围内。

    Args:
        item (dict): 包含会话信息的字典。
        min_length (int): 最小最大行长度，默认值为 10。
        max_length (float): 最大最大行长度，默认值为无穷大。

    Returns:
        bool: 如果最大行长度在 [min_length, max_length] 范围内，返回 True；否则返回 False。
    """
    # 清理 conversations 中的问答对，去除 <image>
    cleaned_conversations = [
        [q.replace('<image>\n', '').replace('\n<image>', '').replace('<image>', '').strip(), 
         a.strip()]
        for q, a in item['conversations']
    ]

    # 计算每个问答对中的最大长度
    max_line_length = 0
    for q, a in cleaned_conversations:
        # 比较问题和答案的长度，记录最大值
        max_line_length = max(max_line_length, len(q), len(a))

    # 判断是否在指定范围内
    return min_length <= max_line_length <= max_length



@register()
def maximum_line_length_filter(
    dataset, 
    min_length: Optional[int] = 10, 
    max_length: Optional[float] = float('inf')  # 默认无上限
) -> MMDataset:
    """
    根据会话的最大行长度过滤数据集。

    Args:
        dataset (MMDataset): 待过滤的数据集。
        min_length (int): 最小最大行长度，默认为 10。
        max_length (float): 最大最大行长度，默认为无穷大（无上限）。

    Returns:
        MMDataset: 过滤后的数据集。
    """
    print("正在过滤最大行长度不符合要求的样本...")
    # 创建过滤函数
    filter_func = partial(is_max_line_length_valid, min_length=min_length, max_length=max_length)
    
    # 调用 dataset.filter
    filtered_dataset = dataset.filter(
        func=filter_func, 
        max_workers=8, 
        progress=True
    )
    
    return filtered_dataset