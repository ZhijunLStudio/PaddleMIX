import os
from PIL import Image
from typing import Optional, List
from ...core import T, MMDataset, register
import imagehash


# 定义宽高比过滤函数
def is_valid_image_aspect_ratio(
    item, 
    min_ratio: float = 0.333, 
    max_ratio: float = 3.0
) -> bool:
    """
    检查图像的宽高比是否在给定范围内。
    
    Args:
        item (dict): 包含图像路径等信息的字典。
        min_ratio (float): 最小宽高比，默认为 0.333。
        max_ratio (float): 最大宽高比，默认为 3.0。
        
    Returns:
        bool: 如果宽高比在范围内，返回 True；否则返回 False。
    """
    image_path = item.get('image')
    if not image_path or not os.path.exists(image_path):
        return False
    
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            ratio = width / height
            return min_ratio <= ratio <= max_ratio
    except Exception as e:
        print(f"处理图像时出错 {image_path}: {e}")
        return False