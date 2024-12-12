import os
from typing import Optional
from ...core import T, MMDataset, register
from functools import partial



# 定义文件大小过滤函数
def is_valid_image_file_size(
    item: dict,
    min_size_kb: Optional[float] = 10,
    max_size_kb: Optional[float] = None
) -> bool:
    """
    检查图像文件大小是否在给定范围内。
    
    Args:
        item (dict): 包含图像路径等信息的字典。
        min_size_kb (Optional[float]): 最小文件大小（以 KB 为单位），默认为 None，不限制最小值。
        max_size_kb (Optional[float]): 最大文件大小（以 KB 为单位），默认为 None，不限制最大值。
        
    Returns:
        bool: 如果文件大小符合条件，返回 True；否则返回 False。
    """
    image_path = item.get('image')
    if not image_path or not os.path.exists(image_path):
        return False
    
    try:
        file_size_kb = os.path.getsize(image_path) / 1024  # 转换为 KB
        if (min_size_kb is not None and file_size_kb < min_size_kb) or (max_size_kb is not None and file_size_kb > max_size_kb):
            return False
        return True
    except Exception as e:
        print(f"处理图像文件大小时出错 {image_path}: {e}")
        return False



@register()
def image_filesize_filter(
    dataset: MMDataset, 
    min_size_kb: Optional[float] = 10, 
    max_size_kb: Optional[float] = None
) -> MMDataset:
    print("正在过滤文件大小不符合要求的图像...")
    
    # 使用 partial 绑定参数到过滤函数
    filter_func = partial(
        is_valid_image_file_size, 
        min_size_kb=min_size_kb, 
        max_size_kb=max_size_kb
    )
    
    # 调用 dataset.filter
    dataset = dataset.filter(
        func=filter_func, 
        max_workers=8, 
        progress=True
    )
    return dataset