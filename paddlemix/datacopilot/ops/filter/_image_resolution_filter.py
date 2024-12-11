import os
from PIL import Image
from typing import Optional, List
from ...core import T, MMDataset, register
import imagehash


# 定义分辨率过滤函数
def is_valid_image_resolution(
    item, 
    max_width: float = 727.88, 
    max_height: float = 606.24
) -> bool:
    """
    检查图像分辨率是否在给定的最大宽度和高度范围内。
    
    Args:
        item (dict): 包含图像路径等信息的字典。
        max_width (float): 最大宽度，默认为 727.88。
        max_height (float): 最大高度，默认为 606.24。
        
    Returns:
        bool: 如果图像分辨率符合条件，返回 True；否则返回 False。
    """
    image_path = item.get('image')
    if not image_path or not os.path.exists(image_path):
        return False
    
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            return width <= max_width and height <= max_height
    except Exception as e:
        print(f"处理图像时出错 {image_path}: {e}")
        return False