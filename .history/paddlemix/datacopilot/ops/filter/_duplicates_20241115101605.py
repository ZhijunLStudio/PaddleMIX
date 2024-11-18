import os
from typing import Optional, List, Dict
import imagehash
from PIL import Image
from ...core import T, MMDataset, register

@register()
def compute_hash(
    item: T,
    hash_method: str = "phash",
) -> Optional[str]:
    """计算图像的感知哈希值。"""
    image_path = item.get('image')
    if not image_path or not os.path.exists(image_path):
        return None

    try:
        with Image.open(image_path) as img:
            if hash_method == "phash":
                hash_value = imagehash.phash(img)
            elif hash_method == "dhash":
                hash_value = imagehash.dhash(img)
            elif hash_method == "average_hash":
                hash_value = imagehash.average_hash(img)
            else:
                raise ValueError(f"不支持的哈希方法: {hash_method}")
            return str(hash_value)
            
    except Exception as e:
        print(f"处理图像时出错 {image_path}: {e}")
        return None

def merge_conversations(conversations_list: List[List[Dict]]) -> List[Dict]:
    """合并多个对话列表，去除重复的对话。"""
    merged = []
    seen_pairs = set()  # 用于追踪已经见过的问答对
    
    for conversations in conversations_list:
        for i in range(0, len(conversations), 2):  # 每次处理一个问答对
            if i + 1 < len(conversations):
                # 创建问答对的唯一标识
                qa_pair = (
                    conversations[i]['value'].strip(),
                    conversations[i+1]['value'].strip()
                )
                
                if qa_pair not in seen_pairs:
                    merged.extend([conversations[i], conversations[i+1]])
                    seen_pairs.add(qa_pair)
    
    return merged

@register()
def remove_duplicates(
    dataset: MMDataset,
    hash_method: str = "phash",
    merge_text: bool = False
) -> MMDataset:
    """使用感知哈希算法移除重复图像，可选择是否合并对话内容。
    
    Args:
        dataset (MMDataset): 输入数据集
        hash_method (str): 使用的哈希算法类型，默认为 "phash"
        merge_text (bool): 是否合并重复图像的对话内容，默认为 False
        
    Returns:
        MMDataset: 处理后的数据集
    """
    # 用于存储已经出现的哈希值及其相关信息
    hash_dict: Dict[str, List] = {}
    filtered_items = []
    
    # 计算所有图像的哈希值
    hash_values = dataset.map(
        lambda x: compute_hash(x, hash_method),
        max_workers=8,
        progress=True
    )

    # 遍历数据集，处理重复图像
    for item, hash_value in zip(dataset, hash_values):
        if not hash_value:
            continue
            
        if hash_value not in hash_dict:
            # 新的哈希值，初始化
            hash_dict[hash_value] = {
                'item': item,
                'conversations_list': [item['conversations']]
            }
        elif merge_text:
            # 已存在的哈希值，且需要合并文本
            hash_dict[hash_value]['conversations_list'].append(item['conversations'])
    
    # 处理结果
    for hash_value, data in hash_dict.items():
        new_item = data['item'].copy()
        if merge_text and len(data['conversations_list']) > 1:
            # 合并对话内容
            new_item['conversations'] = merge_conversations(data['conversations_list'])
        filtered_items.append(new_item)
            
    # 返回新的数据集实例
    return MMDataset(filtered_items)