import os
import fasttext
import requests
from typing import Optional, List, Union
from functools import partial
from ...core import T, MMDataset, register

FASTTEXT_MODEL_PATH = "/home/lizhijun/llm/PaddleMix/lid.176.bin"
FASTTEXT_MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"

# 检查并加载 FastText 模型
def load_fasttext_model(model_path: str, model_url: str) -> fasttext.FastText._FastText:
    if not os.path.exists(model_path):
        print(f"FastText 模型文件 {model_path} 不存在，正在下载...")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        response = requests.get(model_url, stream=True)
        with open(model_path, 'wb') as f:
            f.write(response.content)
        print(f"FastText 模型已下载到 {model_path}")
    print(f"正在加载 FastText 模型 {model_path}...")
    return fasttext.load_model(model_path)

# 加载 FastText 模型
lang_model = load_fasttext_model(FASTTEXT_MODEL_PATH, FASTTEXT_MODEL_URL)



# 判断样本语言是否符合要求
def is_language_valid(item, lang: Optional[Union[str, List[str]]] = None, min_score: float = 0.8) -> bool:
    """
    检查样本的语言是否符合指定语言，并且置信分数大于等于指定最小值。

    Args:
        item (dict): 包含文本信息的样本。
        lang (Union[str, List[str], None]): 允许的语言代码，可以是单个字符串、多语言列表或 None。
        min_score (float): 最小语言置信分数，默认值为 0.8。

    Returns:
        bool: 如果样本的语言符合要求且置信分数足够高，返回 True；否则返回 False。
    """

    user_conv = '\n\n'.join(
    ''.join(conversation) for conversation in item['conversations']
    ).replace('<image>', '').replace('\n', '')


    try:
        prediction = lang_model.predict(user_conv, k=1)
        lang_id = prediction[0][0].replace("__label__", "")
        lang_score = prediction[1][0]
    except Exception as e:
        print(f"语言检测失败，错误信息: {e}")
        return False

    # 检查语言代码和置信分数
    if lang:
        if isinstance(lang, str):
            lang = [lang]  # 如果是单个字符串，转为列表
        return lang_id in lang and lang_score >= min_score
    else:
        # 如果未指定语言，仅检查置信分数
        return lang_score >= min_score


@register()
def language_id_filter(
    dataset, 
    lang: Optional[Union[str, List[str]]] = None, 
    min_score: float = 0.8
) -> MMDataset:
    """
    根据样本的语言ID和置信分数过滤数据集。

    Args:
        dataset (MMDataset): 待过滤的数据集。
        lang (Union[str, List[str], None]): 允许的语言代码，可以是单个字符串、多语言列表或 None。
        min_score (float): 最小语言置信分数，默认值为 0.8。

    Returns:
        MMDataset: 过滤后的数据集。
    """
    print("正在过滤不符合语言ID要求的样本...")
    # 创建过滤函数
    filter_func = partial(is_language_valid, lang=lang, min_score=min_score)
    
    # 调用 dataset.filter
    filtered_dataset = dataset.filter(
        func=filter_func, 
        max_workers=8, 
        progress=True
    )
    
    return filtered_dataset