from typing import Optional
from ...core import T, MMDataset, register
from functools import partial
import numpy as np


def is_char_ngram_valid(item, rep_len: int = 10, min_ratio: float = 0.0, max_ratio: float = 0.5) -> bool:
    """
    检查会话的字符 n-gram 重复比例是否在指定范围内。

    Args:
        item (dict): 包含会话信息的字典。
        rep_len (int): n-gram 的长度，默认为 10。
        min_ratio (float): 最小重复比例，默认为 0.0。
        max_ratio (float): 最大重复比例，默认为 0.5。

    Returns:
        bool: 如果重复比例在 [min_ratio, max_ratio] 范围内，返回 True；否则返回 False。
    """
    # 拼接 conversations 内容
    user_conv = '\n\n'.join(
        ''.join(conversation) for conversation in item['conversations']
    ).replace('<image>\n', '').replace('\n<image>', '').replace('<image>', '')

    # 如果文本长度小于 n-gram 长度，直接返回 False
    if len(user_conv) < rep_len:
        return False

    # 生成 n-grams
    char_ngrams = [
        user_conv[i:i + rep_len]
        for i in range(len(user_conv) - rep_len + 1)
    ]

    # 统计每个 n-gram 的频率
    freq_char_ngrams = {}
    for ngram in char_ngrams:
        freq_char_ngrams[ngram] = freq_char_ngrams.get(ngram, 0) + 1

    # 如果没有有效 n-gram，直接返回 False
    if len(freq_char_ngrams) == 0:
        return False

    # 计算重复 n-gram 的比例
    freq_values = list(freq_char_ngrams.values())
    total_ngrams = sum(freq_values)
    num_no_rep_ngrams = len([freq for freq in freq_values if freq == 1])
    num_rep_ngrams = min(
        int(np.sqrt(len(freq_values))),
        len(freq_values) - num_no_rep_ngrams
    )
    rep_ratio = sum(sorted(freq_values, reverse=True)[:num_rep_ngrams]) / total_ngrams


    # 判断是否在指定范围内
    return min_ratio <= rep_ratio <= max_ratio


@register()
def char_ngram_repetition_filter(
    dataset, 
    rep_len: Optional[int] = 10, 
    min_ratio: Optional[float] = 0.0, 
    max_ratio: Optional[float] = 0.1
) -> MMDataset:
    """
    根据会话的字符 n-gram 重复比例过滤数据集。

    Args:
        dataset (MMDataset): 待过滤的数据集。
        rep_len (int): n-gram 的长度，默认为 10。
        min_ratio (float): 最小重复比例，默认为 0.0。
        max_ratio (float): 最大重复比例，默认为 0.5。

    Returns:
        MMDataset: 过滤后的数据集。
    """
    print("正在过滤字符 n-gram 重复比例不符合要求的样本...")
    # 创建过滤函数
    filter_func = partial(is_char_ngram_valid, rep_len=rep_len, min_ratio=min_ratio, max_ratio=max_ratio)
    
    # 调用 dataset.filter
    filtered_dataset = dataset.filter(
        func=filter_func, 
        max_workers=1, 
        progress=True
    )
    
    return filtered_dataset