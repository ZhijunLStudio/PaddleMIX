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
        lang (str): 语言代码，支持 'en'（英语）。

    Returns:
        spacy.Language: spaCy 语言模型实例。
    """
    if lang == 'en':
        return spacy.load("en_core_web_sm")  # 英语
    else:
        raise ValueError(f"Unsupported language: {lang}")


def is_entity_dependency_valid(item, nlp, min_dependency_num: int = 1, any_or_all: str = 'any') -> bool:
    """
    检查样本中的实体依赖关系是否符合指定的条件。

    Args:
        item (dict): 包含文本信息的样本字典。
        nlp (spacy.Language): 已加载的 spaCy 模型。
        min_dependency_num (int): 每个实体的最小依赖边数量，默认值为 1。
        any_or_all (str): 筛选策略，'any' 表示只要有一个实体满足条件即可，
        'all' 表示所有实体都必须满足条件。

    Returns:
        bool: 如果实体的依赖关系符合要求，返回 True；否则返回 False。
    """
    # 获取文本内容并清理特殊字符
    user_conv = '\n\n'.join(
        ''.join(conversation) for conversation in item['conversations']
    ).replace('<image>\n', '').replace('\n<image>', '').replace('<image>', '')
    # print("user_conv:", user_conv)

    # 使用 spaCy 模型处理文本
    doc = nlp(user_conv)

    # 定义实体的 POS 和 Tag 规则
    entity_poss = ['NOUN', 'PROPN', 'PRON']  # 名词、专有名词、代词
    entity_tags = ['NN', 'NR', 'PN', 'NNS', 'NNP', 'NNPS', 'PRP']

    # 识别实体并初始化依赖计数
    entity_to_dependency_nums = {}
    for token in doc:
        if token.pos_ in entity_poss and token.tag_ in entity_tags:
            entity_to_dependency_nums[token] = 0

    # 计算实体的依赖边数量
    for obj in entity_to_dependency_nums:
        if obj.dep_ != 'ROOT':  # 非根节点计数
            entity_to_dependency_nums[obj] += 1

    for token in doc:
        # 跳过标点符号
        if token.pos_ == 'PUNCT':
            continue

        # 如果 token 的头节点是某个实体，则增加依赖计数
        if token.head in entity_to_dependency_nums.keys() and token.dep_ != 'ROOT':
            entity_to_dependency_nums[token.head] += 1

    # 获取所有实体的依赖边数量
    dependency_counts = [n for _, n in entity_to_dependency_nums.items()]

    # 筛选逻辑
    if any_or_all == 'any':
        # 只要有一个实体满足依赖条件即可
        return any(count >= min_dependency_num for count in dependency_counts)
    elif any_or_all == 'all':
        # 所有实体都必须满足依赖条件
        return all(count >= min_dependency_num for count in dependency_counts)
    else:
        raise ValueError(f"Unsupported any_or_all value: {any_or_all}")


@register()
def text_entity_dependency_filter(
    dataset, 
    lang: str = 'en', 
    min_dependency_num: Optional[int] = 10, 
    any_or_all: str = 'any'
) -> MMDataset:
    """
    根据样本中的实体依赖关系过滤数据集。

    Args:
        dataset (MMDataset): 待过滤的数据集。
        lang (str): 文本语言，支持 'en'（英语）。
        min_dependency_num (int): 每个实体的最小依赖边数量，默认值为 1。
        any_or_all (str): 筛选策略，'any' 表示只要有一个实体满足条件即可，
                          'all' 表示所有实体都必须满足条件。

    Returns:
        MMDataset: 过滤后的数据集。
    """
    print(f"正在基于语言 {lang} 和实体依赖条件 {any_or_all} 过滤样本，最小依赖边数量为 {min_dependency_num}...")
    
    # 加载 spaCy 模型，只加载一次
    nlp = load_spacy_model(lang)

    # 创建过滤函数
    filter_func = partial(is_entity_dependency_valid, nlp=nlp, min_dependency_num=min_dependency_num, any_or_all=any_or_all)
    
    # 调用 dataset.filter
    filtered_dataset = dataset.filter(
        func=filter_func, 
        max_workers=8, 
        progress=True
    )
    
    return filtered_dataset