from typing import Optional
from ...core import T, MMDataset, register
from functools import partial


def is_alnum_ratio_valid(item, min_ratio: float = 0.25, max_ratio: float = float('inf')) -> bool:
    """
    检查样本中字母或数字字符占总字符数的比例是否在指定范围内。

    Args:
        item (dict): 包含文本信息的字典。
        min_ratio (float): 最小比例，默认值为 0.25。
        max_ratio (float): 最大比例，默认值为正无穷。

    Returns:
        bool: 如果比例在 [min_ratio, max_ratio] 范围内，返回 True；否则返回 False。
    """
    # 获取文本内容
    user_conv = '\n\n'.join(
        ''.join(conversation) for conversation in item['conversations']
    ).replace('<image>\n', '').replace('\n<image>', '').replace('<image>', '')


    # 计算字母和数字的总数
    alnum_count = sum(1 for char in user_conv if char.isalnum())
    
    # 计算字母或数字字符的比例
    alnum_ratio = alnum_count / len(user_conv) if len(user_conv) > 0 else 0.0

    # 判断是否在指定比例范围内
    return min_ratio <= alnum_ratio <= max_ratio


@register()
def alphanumeric_ratio_filter(
    dataset, 
    min_ratio: Optional[float] = 0.25, 
    max_ratio: Optional[float] = float('inf')
) -> MMDataset:
    """
    根据样本中字母或数字字符比例过滤数据集。

    Args:
        dataset (MMDataset): 待过滤的数据集。
        min_ratio (float): 最小比例，默认为 0.25。
        max_ratio (float): 最大比例，默认为正无穷。

    Returns:
        MMDataset: 过滤后的数据集。
    """
    print("正在过滤字母或数字字符比例不符合要求的样本...")
    # 创建过滤函数
    filter_func = partial(is_alnum_ratio_valid, min_ratio=min_ratio, max_ratio=max_ratio)
    
    # 调用 dataset.filter
    filtered_dataset = dataset.filter(
        func=filter_func, 
        max_workers=8, 
        progress=True
    )
    
    return filtered_dataset