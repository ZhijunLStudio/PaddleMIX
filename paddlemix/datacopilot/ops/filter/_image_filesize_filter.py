import os
from PIL import Image
from typing import Optional, List
from ...core import T, MMDataset, register


# 定义文件大小过滤函数
def is_valid_image_file_size(
    item, 
    max_size_kb: float = 124
    ) -> bool:
    """
    检查图像文件大小是否在给定范围内。
    
    Args:
        item (dict): 包含图像路径等信息的字典。
        max_size_kb (float): 最大文件大小（以 KB 为单位），默认为 124。
        
    Returns:
        bool: 如果文件大小符合条件，返回 True；否则返回 False。
    """
    image_path = item.get('image')
    if not image_path or not os.path.exists(image_path):
        return False
    
    try:
        file_size_kb = os.path.getsize(image_path) / 1024  # 转换为 KB
        return file_size_kb <= max_size_kb
    except Exception as e:
        print(f"处理图像文件大小时出错 {image_path}: {e}")
        return False
