import os
from PIL import Image
from typing import Optional, List
from ...core import T, MMDataset, register
import imagehash
from functools import partial



def is_valid_image_hash(
    item, 
    seen_hashes: set, 
    hash_method: str = "phash"
) -> bool:
    """
    判断图像是否需要保留（基于哈希值去重）。
    
    Args:
        item (dict): 数据样本，包含图像路径的字典。
        seen_hashes (set): 用于记录已出现的哈希值的集合。
        hash_method (str): 使用的哈希算法类型，支持 "phash" (默认), "dhash", 和 "average_hash"。
        
    Returns:
        bool: 如果图像需要保留，则返回 True；否则返回 False。
    """
    image_path = item.get('image')
    if not image_path or not os.path.exists(image_path):
        return False

    try:
        with Image.open(image_path) as img:
            # 计算哈希值
            if hash_method == "phash":
                img_hash = str(imagehash.phash(img))
            elif hash_method == "dhash":
                img_hash = str(imagehash.dhash(img))
            elif hash_method == "average_hash":
                img_hash = str(imagehash.average_hash(img))
            else:
                raise ValueError(f"不支持的哈希方法: {hash_method}")
            
            # 检查哈希值是否已存在
            if img_hash in seen_hashes:
                return False
            seen_hashes.add(img_hash)
            return True
    except Exception as e:
        print(f"处理图像 {image_path} 时发生错误: {e}")
        return False


def image_hash_filter(
    dataset, 
    hash_method: Optional[str] = "phash"
) -> MMDataset:
    """
    使用图像哈希值过滤数据集。
    
    Args:
        dataset (MMDataset): 输入数据集。
        hash_method (Optional[str]): 使用的哈希算法类型，默认 "phash"。
        
    Returns:
        MMDataset: 过滤后的数据集。
    """
    print("正在过滤重复图像...")

    # 初始化一个集合，用于记录已出现的哈希值
    seen_hashes = set()
    
    # 创建过滤函数，绑定 seen_hashes
    filter_func = partial(is_valid_image_hash, seen_hashes=seen_hashes, hash_method=hash_method)
    
    # 调用 dataset.filter
    filtered_dataset = dataset.filter(
        func=filter_func, 
        max_workers=8, 
        progress=True
    )
    
    return filtered_dataset