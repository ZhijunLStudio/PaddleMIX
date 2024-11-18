import os
from PIL import Image
from typing import Optional, List
from ...core import T, MMDataset, register

@register()
def filter_by_aspect_ratio(
    dataset: MMDataset,
    min_ratio: float = 0.333,
    max_ratio: float = 3.0
) -> MMDataset:
    """
    根据图像宽高比过滤样本。
    
    Args:
        dataset (MMDataset): 输入数据集。
        min_ratio (float): 最小宽高比，默认为 0.333。
        max_ratio (float): 最大宽高比，默认为 3.0。
        
    Returns:
        MMDataset: 过滤后的数据集。
    """
    filtered_items = []
    
    for item in dataset:
        image_path = item.get('image')
        if not image_path or not os.path.exists(image_path):
            continue
        
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                ratio = width / height
                if min_ratio <= ratio <= max_ratio:
                    filtered_items.append(item)
        except Exception as e:
            print(f"处理图像时出错 {image_path}: {e}")
    
    return MMDataset(filtered_items)


@register()
def filter_by_resolution(
    dataset: MMDataset,
    max_width: float = 727.88,
    max_height: float = 606.24
) -> MMDataset:
    """
    根据图像分辨率过滤样本。
    
    Args:
        dataset (MMDataset): 输入数据集。
        max_width (float): 最大宽度，默认为 727.88。
        max_height (float): 最大高度，默认为 606.24。
        
    Returns:
        MMDataset: 过滤后的数据集。
    """
    filtered_items = []
    
    for item in dataset:
        image_path = item.get('image')
        if not image_path or not os.path.exists(image_path):
            continue
        
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                if width <= max_width and height <= max_height:
                    filtered_items.append(item)
        except Exception as e:
            print(f"处理图像时出错 {image_path}: {e}")
    
    return MMDataset(filtered_items)


@register()
def filter_by_file_size(
    dataset: MMDataset,
    max_size_kb: float = 124
) -> MMDataset:
    """
    根据图像文件大小过滤样本。
    
    Args:
        dataset (MMDataset): 输入数据集。
        max_size_kb (float): 最大文件大小（以 KB 为单位），默认为 124。
        
    Returns:
        MMDataset: 过滤后的数据集。
    """
    filtered_items = []
    
    for item in dataset:
        image_path = item.get('image')
        if not image_path or not os.path.exists(image_path):
            continue
        
        try:
            file_size_kb = os.path.getsize(image_path) / 1024  # 转换为 KB
            if file_size_kb <= max_size_kb:
                filtered_items.append(item)
        except Exception as e:
            print(f"处理图像文件大小时出错 {image_path}: {e}")
    
    return MMDataset(filtered_items)
