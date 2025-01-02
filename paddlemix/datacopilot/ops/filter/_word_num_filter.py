import string
import re
from typing import List
import sys
from typing import Optional
from ...core import T, MMDataset, register
from functools import partial

# 定义特殊字符（可以根据需要调整）
SPECIAL_CHARACTERS = string.punctuation + "“”‘’"  # 这里可以添加需要清理的特殊字符

def words_refinement(words: List[str], strip_chars: str = SPECIAL_CHARACTERS) -> List[str]:
    """
    对单词列表进行清理，去除特殊字符和标点符号。
    
    Args:
        words (List[str]): 原始单词列表。
        strip_chars (str): 需要去除的字符，默认为特殊字符。
        
    Returns:
        List[str]: 清理后的单词列表。
    """
    refined_words = []

    # 对每个单词进行处理
    for word in words:
        # 使用正则表达式去除掉特殊字符和标点符号
        refined_word = re.sub(f"[{re.escape(strip_chars)}]", "", word)
        # 只保留字母和数字的单词，去除空字符串
        if refined_word:
            refined_words.append(refined_word.lower())  # 将单词转换为小写

    return refined_words


# 计算词数并进行过滤的主逻辑
def compute_word_count(item, min_num: int = 10, max_num: int = sys.maxsize, context=False) -> bool:
    """
    计算单个样本的词数并判断是否在指定的范围内。

    Args:
        item (dict): 样本字典，包含会话内容。
        min_num (int): 最小词数限制。
        max_num (int): 最大词数限制。
        context (bool): 是否使用上下文信息。

    Returns:
        bool: 如果样本的词数在 [min_num, max_num] 范围内，返回 True，否则返回 False。
    """
    # 拼接 conversations 内容
    user_conv = '\n\n'.join(
        ''.join(conversation) for conversation in item['conversations']
    ).replace('<image>\n', '').replace('\n<image>', '').replace('<image>', '')

    # 通过空格进行分词
    words = user_conv.split()  # 按空格分割

    # 使用 words_refinement 进行词汇处理
    words = words_refinement(words, strip_chars=SPECIAL_CHARACTERS)
    word_count = len(words)

    # 判断是否符合词数范围
    if min_num <= word_count <= max_num:
        return True
    else:
        return False


# 过滤函数的注册
@register()
def word_num_filter(
    dataset, 
    min_num: Optional[int] = 10, 
    max_num: Optional[int] = sys.maxsize
) -> MMDataset:
    """
    根据样本的词数过滤数据集。

    Args:
        dataset (MMDataset): 待过滤的数据集。
        min_num (int): 最小词数限制，默认 10。
        max_num (int): 最大词数限制，默认 sys.maxsize。

    Returns:
        MMDataset: 过滤后的数据集。
    """
    print(f"正在过滤词数不符合要求的样本，最小词数: {min_num}, 最大词数: {max_num}...")

    # 创建过滤函数
    filter_func = partial(compute_word_count, min_num=min_num, max_num=max_num)

    # 调用 dataset.filter
    filtered_dataset = dataset.filter(
        func=filter_func, 
        max_workers=8, 
        progress=True
    )

    return filtered_dataset