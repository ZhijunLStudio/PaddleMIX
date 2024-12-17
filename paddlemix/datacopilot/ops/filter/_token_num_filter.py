import os
from typing import Optional
from functools import partial
from ...core import T, MMDataset, register
from paddlenlp.transformers import AutoTokenizer
import sys

# 默认使用的 tokenizer 模型
DEFAULT_TOKENIZER = "Qwen/Qwen2.5-0.5B"

# 定义 Token 过滤函数
def compute_token_count(user_conv: str, tokenizer: AutoTokenizer) -> int:
    """
    计算样本（会话）的 token 数量。

    Args:
        user_conv (str): 合并后的对话文本。
        tokenizer (AutoTokenizer): 用于 token 化的 tokenizer 实例。

    Returns:
        int: 该样本的 token 数量。
    """
    tokens = tokenizer(user_conv, truncation=True, return_tensors="pd", use_fast=True)["input_ids"].flatten()
    return len(tokens)

@register()
def token_num_filter(
    dataset, 
    tokenizer_model: str = DEFAULT_TOKENIZER, 
    min_tokens: Optional[int] = 10, 
    max_tokens: Optional[int] = sys.maxsize
) -> MMDataset:
    """
    根据样本中的 token 数量过滤数据集。

    Args:
        dataset (MMDataset): 待过滤的数据集。
        tokenizer_model (str): 采用的 tokenizer 模型名称，默认为 `Qwen/Qwen2.5-0.5B`。
        min_tokens (int): 最小 token 数量，默认值为 10。
        max_tokens (int): 最大 token 数量，默认值为 `sys.maxsize`。

    Returns:
        MMDataset: 过滤后的数据集。
    """
    print(f"正在基于 token 数量过滤样本，最小 token 数量: {min_tokens}, 最大 token 数量: {max_tokens}...")
    
    # 初始化 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)

    def filter_func(item):
        # 获取对话文本并清理
        user_conv = '\n\n'.join(
            ''.join(conversation) for conversation in item['conversations']
        ).replace('<image>', '').replace('\n', '')  # 清理 `<image>` 标签和换行符

        # 计算 token 数量
        num_tokens = compute_token_count(user_conv, tokenizer)

        # 判断是否符合 token 数量范围
        return min_tokens <= num_tokens <= max_tokens

    # 调用 dataset.filter
    filtered_dataset = dataset.filter(
        func=filter_func, 
        max_workers=8, 
        progress=True
    )
    
    return filtered_dataset