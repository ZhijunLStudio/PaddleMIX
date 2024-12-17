from typing import Optional
from ...core import T, MMDataset, register
from functools import partial
from kenlm import Model


def compute_perplexity(text: str, model: Model) -> float:
    """
    使用 KenLM 模型计算文本的困惑度。

    Args:
        text (str): 待计算困惑度的文本。
        model (Model): KenLM 语言模型实例。

    Returns:
        float: 文本的困惑度。
    """
    # 初始化对数概率和单词数
    log_prob, num_words = 0, 0

    # 按行计算困惑度
    for line in text.splitlines():
        log_prob += model.score(line)  # 累加对数概率
        num_words += len(line.split()) + 1  # 单词数（包含结束标记）

    # 计算困惑度
    return (10.0 ** (-log_prob / num_words)) if num_words > 0 else float('inf')


def is_perplexity_valid(item, model_path: str, max_ppl: float = 1500) -> bool:
    """
    检查样本的困惑度是否小于指定阈值。

    Args:
        item (dict): 包含文本信息的字典。
        model_path (str): KenLM 模型的文件路径。
        max_ppl (float): 最大困惑度阈值，默认值为 1500。

    Returns:
        bool: 如果困惑度小于等于 max_ppl，返回 True；否则返回 False。
    """
    # 加载 KenLM 模型
    model = Model(model_path)

    # 拼接会话内容
    user_conv = '\n\n'.join(
        ''.join(conversation) for conversation in item['conversations']
    ).replace('<image>\n', '').replace('\n<image>', '').replace('<image>', '')
    print("user_conv:", user_conv)

    # 计算困惑度
    ppl = compute_perplexity(user_conv, model)

    # 打印调试信息
    print(f"文本: {user_conv[:50]}... 困惑度: {ppl}")

    # 判断是否符合条件
    return ppl <= max_ppl


@register()
def perplexity_filter(
    dataset, 
    model_path: str, 
    max_ppl: Optional[float] = 1500
) -> MMDataset:
    """
    根据样本的困惑度过滤数据集。

    Args:
        dataset (MMDataset): 待过滤的数据集。
        model_path (str): KenLM 模型的文件路径。
        max_ppl (float): 最大困惑度阈值，默认值为 1500。

    Returns:
        MMDataset: 过滤后的数据集。
    """
    print("正在过滤困惑度超出阈值的样本...")
    # 创建过滤函数
    filter_func = partial(is_perplexity_valid, model_path=model_path, max_ppl=max_ppl)
    
    # 调用 dataset.filter
    filtered_dataset = dataset.filter(
        func=filter_func, 
        max_workers=1, 
        progress=True
    )
    
    return filtered_dataset