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
