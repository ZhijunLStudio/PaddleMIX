import os
from PIL import Image
from typing import Optional, List
from ...core import T, MMDataset, register
from functools import partial


# 定义分辨率过滤函数
def is_valid_image_resolution(
    item, 
    min_width: float = 112, 
    min_height: float = 112, 
    max_width: Optional[float] = None, 
    max_height: Optional[float] = None
) -> bool:
    """
    检查图像分辨率是否在给定的最小和最大宽度、高度范围内。
    
    Args:
        item (dict): 包含图像路径等信息的字典。
        min_width (float): 最小宽度，默认为 112。
        min_height (float): 最小高度，默认为 112。
        max_width (Optional[float]): 最大宽度，默认为 None（不限制）。
        max_height (Optional[float]): 最大高度，默认为 None（不限制）。
        
    Returns:
        bool: 如果图像分辨率符合条件，返回 True；否则返回 False。
    """
    image_path = item.get('image')
    if not image_path or not os.path.exists(image_path):
        return False
    
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            
            # 检查最小宽高
            if width < min_width or height < min_height:
                return False
            
            # 检查最大宽高
            if max_width is not None and width > max_width:
                return False
            if max_height is not None and height > max_height:
                return False
            
            return True
    except Exception as e:
        print(f"处理图像时出错 {image_path}: {e}")
        return False


@register()
def image_resolution_filter(
    dataset, 
    min_width: Optional[float] = 112, 
    min_height: Optional[float] = 112, 
    max_width: Optional[float] = None, 
    max_height: Optional[float] = None
) -> MMDataset:
    print("正在过滤分辨率不合适的图像...")
    # 创建过滤函数
    filter_func = partial(is_valid_image_resolution, min_width=min_width, min_height=min_height)
    
    # 调用 dataset.filter
    filtered_dataset = dataset.filter(
        func=filter_func, 
        max_workers=8, 
        progress=True
    )
    
    return filtered_dataset


