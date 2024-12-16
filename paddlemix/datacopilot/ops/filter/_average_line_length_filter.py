from typing import Optional
from ...core import T, MMDataset, register
from functools import partial


def is_avg_line_length_valid(item, min_length: int = 10, max_length: float = float('inf')) -> bool:
    """
    检查会话的平均行长度是否在指定范围内。

    Args:
        item (dict): 包含会话信息的字典。
        min_length (int): 最小平均行长度，默认值为 10。
        max_length (float): 最大平均行长度，默认值为无穷大。

    Returns:
        bool: 如果平均行长度在 [min_length, max_length] 范围内，返回 True；否则返回 False。
    """
    # 拼接 conversations 内容
    user_conv = '\n\n'.join(
        ''.join(conversation) for conversation in item['conversations']
    ).replace('<image>\n', '').replace('\n<image>', '').replace('<image>', '')

    # 按行分割文本
    lines = user_conv.splitlines()

    # 如果没有有效行，直接返回 False
    if not lines:
        return False

    # 计算平均行长度
    avg_line_length = sum(len(line) for line in lines) / len(lines)

    # 计算平均行长度
    avg_line_length = sum(len(line) for line in lines) / len(lines)

    # 判断是否在指定范围内
    return min_length <= avg_line_length <= max_length


@register()
def average_line_length_filter(
    dataset, 
    min_length: Optional[int] = 10, 
    max_length: Optional[float] = float('inf')  # 默认无上限
) -> MMDataset:
    """
    根据会话的平均行长度过滤数据集。

    Args:
        dataset (MMDataset): 待过滤的数据集。
        min_length (int): 最小平均行长度，默认为 10。
        max_length (float): 最大平均行长度，默认为无穷大（无上限）。

    Returns:
        MMDataset: 过滤后的数据集。
    """
    print("正在过滤平均行长度不符合要求的样本...")
    # 创建过滤函数
    filter_func = partial(is_avg_line_length_valid, min_length=min_length, max_length=max_length)
    
    # 调用 dataset.filter
    filtered_dataset = dataset.filter(
        func=filter_func, 
        max_workers=8, 
        progress=True
    )
    
    return filtered_dataset




