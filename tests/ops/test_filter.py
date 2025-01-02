import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

from paddlemix.datacopilot.core import MMDataset
from paddlemix.datacopilot.ops.convert._llava_convert import llava_convert
from paddlemix.datacopilot.ops.filter._base_filter import valid_data_filter

# 数据集路径
anno_path = 'datasets/llava/01_val_chatml.json'

# 加载数据集
print("Loading dataset...")
dataset = MMDataset.from_json(anno_path)
print("初始数据集数量为:", len(dataset))


# 转换算子
dataset = dataset.llava_convert()

# 0.过滤无效图像和文本
# dataset = dataset.valid_data_filter()




dataset = dataset.nonempty()

print("过滤后数据集数量为:", len(dataset))
print("Dataset validation complete.")
dataset.export_json(anno_path.replace('.json', '_filter-noempty.json'))
# dataset.export_json("test.json")