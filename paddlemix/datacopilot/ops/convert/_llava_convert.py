from typing import List, Dict
from ...core import T, MMDataset, register
from functools import partial

# 定义转换函数
def convert_llava_item(item: Dict, image_path_prefix: str = '') -> Dict:
    """
    转换每个数据项为目标格式。
    
    参数:
        item (dict): 原始数据项，包含 'image' 和 'conversations' 键。
        image_path_prefix (str): 图片路径的前缀，默认为空字符串。如果不传递，将使用默认值 'datasets/llava/valid_images/'。
    
    返回:
        dict: 转换后的数据项，包含 'image' 和 'conversations' 键。
    """

    
    # 检查是否存在 'image' 键，如果没有，则设置为空字符串
    image = item.get('image', '')  # 如果没有 'image' 键，默认为空字符串

    # 如果 'image' 为空字符串，则跳过这个会话
    if not image:
        return None  # 如果没有图片，跳过此项，返回 None
    
    # 拼接图片路径
    image = image_path_prefix + image
    # print(item['conversations'])
    
    conversations = []
    for i in range(0, len(item['conversations']), 2):
        human_message = item['conversations'][i]['value']
        gpt_message = item['conversations'][i+1]['value'] if i+1 < len(item['conversations']) else ''
        conversations.append([human_message, gpt_message])

    # 构造转换后的数据结构
    transformed_item = {
        "image": image,
        "conversations": conversations
    }

    return transformed_item

@register()
def llava_convert(dataset: MMDataset) -> MMDataset:

    print('Converting llava dataset...')
    # 使用 map 算子进行批量转换, 传递 'datasets/llava/valid_images/' 作为默认路径
    filter_func = partial(convert_llava_item, image_path_prefix='datasets/llava/valid_images/')

    # 调用 dataset.filter
    dataset = dataset.map(
        func=filter_func, 
        max_workers=8, 
        progress=True
    )
    
    # 过滤掉 None 的项
    dataset = dataset.nonempty()  # 通过 nonempty 过滤掉 None
    
    return dataset
