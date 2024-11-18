# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Optional
import imagehash
from PIL import Image

from ...core import T, MMDataset,
from ...core import 


@register()
def compute_hash(
    item: T,
    hash_method: str = "phash",
) -> Optional[str]:
    """计算图像的感知哈希值。
    
    支持多种哈希算法：
    - phash (感知哈希): 对图像的频域特征进行哈希，对图像缩放、压缩等操作具有鲁棒性
    - dhash (差异哈希): 基于相邻像素的差异计算哈希，计算速度快，对渐变图像效果好
    - average_hash (平均哈希): 最简单的图像哈希算法，计算速度最快但精确度较低
    
    Args:
        item (T): 数据集中的单个数据项，需要符合MM schema格式
        hash_method (str, optional): 使用的哈希算法类型. 默认为 "phash"
    
    Returns:
        Optional[str]: 成功则返回图像的哈希值，失败则返回None
    """
    # 从MM schema中获取图像路径
    image_path = item.get('image')
    if not image_path or not os.path.exists(image_path):
        return None

    try:
        with Image.open(image_path) as img:
            # 根据选择的方法计算哈希值
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


@register()
def remove_duplicates(
    dataset: MMDataset,
    hash_method: str = "phash"
) -> MMDataset:
    """使用感知哈希算法移除数据集中的重复图像。
    
    该函数首先对所有图像计算哈希值，然后通过比较哈希值来识别和移除重复图像。
    支持多种哈希算法，可以根据具体需求选择合适的算法。
    
    Args:
        dataset (MMDataset): 输入数据集
        hash_method (str, optional): 使用的哈希算法类型. 默认为 "phash"
        
    Returns:
        MMDataset: 移除重复图像后的数据集
    """
    # 用于存储已经出现的哈希值
    hash_set = set()
    filtered_items = []
    
    # 使用map函数并行计算所有图像的哈希值
    hash_values = dataset.map(
        lambda x: compute_hash(x, hash_method),
        max_workers=8,
        progress=True
    )
    
    # 遍历数据集，保留哈希值唯一的图像
    for item, hash_value in zip(dataset, hash_values):
        if hash_value and hash_value not in hash_set:
            hash_set.add(hash_value)
            filtered_items.append(item)
            
    # 返回新的数据集实例
    return MMDataset(filtered_items)