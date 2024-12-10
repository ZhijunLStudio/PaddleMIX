from paddlemix.datacopilot.core import MMDataset
from paddlemix.datacopilot.ops.filter._base_filter import validate_image, validate_conversation

# 数据集路径
anno_path = 'datasets/llava/val_chatml.json'

# 加载数据集
print("Loading dataset...")
dataset = MMDataset.from_json(anno_path)

# 过滤无效图像
print("Filtering invalid images...")
# dataset = dataset.filter(
#     func=validate_image, 
#     max_workers=8, 
#     progress=True
# )

# 过滤无效文本
print("Filtering invalid conversations...")
dataset = dataset.filter(
    func=validate_conversation, 
    max_workers=8, 
    progress=True
)

print("Dataset validation complete.")
dataset.export_json(anno_path.replace('.json', '_filter.json'))