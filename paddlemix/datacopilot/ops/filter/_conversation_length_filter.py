from typing import Optional, List
from ...core import T, MMDataset, register
from functools import partial



def is_chat_length_valid(item, max_length: int = 2048) -> bool:
    """
    检查聊天的长度是否小于指定的最大长度。

    Args:
        item (dict): 包含聊天信息的字典。
        max_length (int): 聊天的最大允许长度，默认为 2048。

    Returns:
        bool: 如果聊天长度小于 max_length，返回 True；否则返回 False。
    """
    # 拼接 conversations 内容
    user_conv = '\n\n'.join(
        ''.join(conversation) for conversation in item['conversations']
    ).replace('<image>\n', '').replace('\n<image>', '').replace('<image>', '')

    return len(user_conv) < max_length



@register()
def conversation_length_filter(
    dataset, 
    max_length: Optional[int] = 2048, 
) -> MMDataset:
    print("正在过滤过长的会话...")
    # 创建过滤函数
    filter_func = partial(is_chat_length_valid, max_length=max_length)
    
    # 调用 dataset.filter
    filtered_dataset = dataset.filter(
        func=filter_func, 
        max_workers=8, 
        progress=True
    )
    
    return filtered_dataset


