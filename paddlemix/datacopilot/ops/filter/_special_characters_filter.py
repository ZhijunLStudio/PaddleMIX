from typing import Optional
from ...core import T, MMDataset, register
from functools import partial


def is_special_char_ratio_valid(item, min_ratio: float = 0.0, max_ratio: float = 0.25) -> bool:
    """
    检查样本中特殊字符比例是否在指定范围内。

    Args:
        item (dict): 包含文本信息的字典。
        min_ratio (float): 最小特殊字符比例，默认值为 0.0。
        max_ratio (float): 最大特殊字符比例，默认值为 0.25。

    Returns:
        bool: 如果特殊字符比例在 [min_ratio, max_ratio] 范围内，返回 True；否则返回 False。
    """
    # 拼接会话内容
    user_conv = '\n\n'.join(
        ''.join(conversation) for conversation in item['conversations']
    ).replace('<image>\n', '').replace('\n<image>', '').replace('<image>', '')

    # 计算特殊字符的数量
    special_characters = [
        '|', ':', ';', '@', '(', ')', '[', ']', '{', '}', '^', '\'', '\"', '’',
        '`', '?', '$', '%', '#', '!', '&', '*', '+', ',', '.'
    ]
    special_char_count = sum(1 for char in user_conv if char in special_characters)

    # 计算特殊字符比例
    total_chars = len(user_conv)
    special_char_ratio = special_char_count / total_chars if total_chars > 0 else 0.0


    # 判断是否在指定范围内
    return min_ratio <= special_char_ratio <= max_ratio


@register()
def special_characters_filter(
    dataset, 
    min_ratio: Optional[float] = 0.0, 
    max_ratio: Optional[float] = 0.25
) -> MMDataset:
    """
    根据样本的特殊字符比例过滤数据集。

    Args:
        dataset (MMDataset): 待过滤的数据集。
        min_ratio (float): 最小特殊字符比例，默认值为 0.0。
        max_ratio (float): 最大特殊字符比例，默认值为 0.25。

    Returns:
        MMDataset: 过滤后的数据集。
    """
    print("正在过滤特殊字符比例不符合要求的样本...")
    # 创建过滤函数
    filter_func = partial(is_special_char_ratio_valid, min_ratio=min_ratio, max_ratio=max_ratio)
    
    # 调用 dataset.filter
    filtered_dataset = dataset.filter(
        func=filter_func, 
        max_workers=8, 
        progress=True
    )
    
    return filtered_dataset