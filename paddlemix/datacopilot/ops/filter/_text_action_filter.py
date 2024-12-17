from typing import Optional
from functools import partial
from ...core import T, MMDataset, register
import spacy

# python -m spacy download en_core_web_sm

# 加载 spaCy 模型
def load_spacy_model(lang: str):
    """
    加载 spaCy 模型，根据指定的语言加载对应的模型。

    Args:
        lang (str): 语言代码，支持 'en'（英语）和 'zh'（中文）。

    Returns:
        spacy.Language: spaCy 语言模型实例。
    """
    if lang == 'en':
        return spacy.load("en_core_web_sm")  # 英语
    else:
        raise ValueError(f"Unsupported language: {lang}")


def is_action_count_valid(item, nlp, min_action_num: int = 1) -> bool:
    """
    检查样本中的动词数量是否大于等于指定的最小值。

    Args:
        item (dict): 包含文本信息的样本字典。
        nlp (spacy.Language): 已加载的 spaCy 模型。
        min_action_num (int): 最小动词数量，默认值为 1。

    Returns:
        bool: 如果动词数量大于等于 min_action_num，返回 True；否则返回 False。
    """
    # 获取文本内容并清理特殊字符
    user_conv = '\n\n'.join(
        ''.join(conversation) for conversation in item['conversations']
    ).replace('<image>\n', '').replace('\n<image>', '').replace('<image>', '')

    # 使用 spaCy 模型处理文本
    doc = nlp(user_conv)

    # 根据语言选择动词检测规则
    action_poss = ['VERB']
    action_tags = ['VB', 'VBP', 'VBZ', 'VBD', 'VBG', 'VBN']

    # 统计动词数量
    num_actions = sum(1 for token in doc if token.pos_ in action_poss and token.tag_ in action_tags)

    # 判断是否符合动词数量要求
    return num_actions >= min_action_num


@register()
def text_action_filter(
    dataset, 
    lang: str = 'en', 
    min_action_num: Optional[int] = 1
) -> MMDataset:
    """
    根据样本中的动词数量过滤数据集。

    Args:
        dataset (MMDataset): 待过滤的数据集。
        lang (str): 文本语言，支持 'en'（英语）和 'zh'（中文）。
        min_action_num (int): 最小动词数量，默认值为 1。

    Returns:
        MMDataset: 过滤后的数据集。
    """
    print(f"正在基于语言 {lang} 和动词数量 {min_action_num} 过滤样本...")
    
    # 加载 spaCy 模型，只加载一次
    nlp = load_spacy_model(lang)

    # 创建过滤函数
    filter_func = partial(is_action_count_valid, nlp=nlp, min_action_num=min_action_num)
    
    # 调用 dataset.filter
    filtered_dataset = dataset.filter(
        func=filter_func, 
        max_workers=8, 
        progress=True
    )
    
    return filtered_dataset