import os
from PIL import Image
from typing import Optional, List
from ...core import T, MMDataset, register
import imagehash




# 定义哈希过滤函数
def is_valid_image_hash(
    item, 
    hash_method: str = "phash"
) -> Optional[str]:
    """
    计算图像的感知哈希值。
    
    Args:
        item (dict): 数据样本，包含图像路径的字典。
        hash_method (str): 使用的哈希算法类型，支持 "phash" (默认), "dhash", 和 "average_hash"。
        
    Returns:
        Optional[str]: 图像的哈希值（字符串形式），如果计算失败则返回 None。
    """
    image_path = item.get('image')
    if not image_path or not os.path.exists(image_path):
        return None

    try:
        with Image.open(image_path) as img:
            if hash_method == "phash":
                return str(imagehash.phash(img))
            elif hash_method == "dhash":
                return str(imagehash.dhash(img))
            elif hash_method == "average_hash":
                return str(imagehash.average_hash(img))
            else:
                raise ValueError(f"不支持的哈希方法: {hash_method}")
    except Exception as e:
        return None



