from paddlemix.datacopilot.core import MMDataset
from paddlemix.datacopilot.ops.filter._image_base_filter import (
    is_valid_image_aspect_ratio,
    is_valid_image_resolution,
    is_valid_image_file_size,
    is_valid_image_hash,
)

# 加载数据集
dataset = MMDataset.from_json('./llava_v1_5_mix665k.json')
print(f"原始数据集大小: {len(dataset)}")

# 链式调用多个算子进行过滤
filtered_dataset = (
    dataset
    .filter(is_valid_image_aspect_ratio, max_workers=8, progress=True)
    .filter(lambda x: is_valid_image_resolution(x, max_width=1024, max_height=768), max_workers=8, progress=True)
    .filter(lambda x: is_valid_image_file_size(x, max_size_kb=500), max_workers=8, progress=True)
    .filter(is_valid_image_hash, max_workers=8, progress=True)
    .nonempty()
)

print(f"过滤后的数据集大小: {len(filtered_dataset)}")

# 打印过滤后的数据集
for item in filtered_dataset:
    print(item)
