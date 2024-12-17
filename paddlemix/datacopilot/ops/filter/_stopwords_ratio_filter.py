from typing import Optional
from ...core import T, MMDataset, register
from functools import partial
import nltk
from nltk.corpus import stopwords

# 下载 NLTK 的停用词资源
nltk.download('stopwords')

# 获取停用词列表
stop_words = set(stopwords.words('english'))


def is_stopwords_ratio_valid(item, min_ratio: float = 0.25) -> bool:
    """
    检查样本中的停用词比例是否大于等于指定的最小值。

    Args:
        item (dict): 包含文本信息的字典。
        min_ratio (float): 最小停用词比例，默认值为 0.25。

    Returns:
        bool: 如果停用词比例大于等于 min_ratio，返回 True；否则返回 False。
    """
    # 获取文本内容
    user_conv = '\n\n'.join(
        ''.join(conversation) for conversation in item['conversations']
    ).replace('<image>\n', '').replace('\n<image>', '').replace('<image>', '')

    # 按空格分割文本为单词
    words = user_conv.split()

    # 计算停用词的数量
    stopword_count = sum(1 for word in words if word.lower() in stop_words)

    # 计算停用词比例
    stopword_ratio = stopword_count / len(words) if len(words) > 0 else 0.0

    # 判断是否符合比例要求
    return stopword_ratio >= min_ratio


@register()
def stopwords_ratio_filter(
    dataset, 
    min_ratio: Optional[float] = 0.25
) -> MMDataset:
    """
    根据样本的停用词比例过滤数据集。

    Args:
        dataset (MMDataset): 待过滤的数据集。
        min_ratio (float): 最小停用词比例，默认值为 0.25。

    Returns:
        MMDataset: 过滤后的数据集。
    """
    print("正在过滤停用词比例不符合要求的样本...")
    # 创建过滤函数
    filter_func = partial(is_stopwords_ratio_valid, min_ratio=min_ratio)
    
    # 调用 dataset.filter
    filtered_dataset = dataset.filter(
        func=filter_func, 
        max_workers=8, 
        progress=True
    )
    
    return filtered_dataset