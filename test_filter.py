import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

from paddlemix.datacopilot.core import MMDataset
from paddlemix.datacopilot.ops.filter._base_filter import valid_data_filter
from paddlemix.datacopilot.ops.filter._image_clip_filter import CLIPFilterConfig
from paddlemix.datacopilot.ops.filter._conversation_percentage_filter import conversation_percentage_filter
from paddlemix.datacopilot.ops.filter._conversation_hash_filter import remove_text_duplicates
from paddlemix.datacopilot.ops.filter._image_filesize_filter import image_filesize_filter
from paddlemix.datacopilot.ops.filter._image_hash_filter import image_hash_filter
from paddlemix.datacopilot.ops.filter._image_ration_filter import image_ration_filter
from paddlemix.datacopilot.ops.filter._image_resolution_filter import image_resolution_filter
from paddlemix.datacopilot.ops.filter._conversation_length_filter import conversation_length_filter

# 数据集路径
anno_path = 'datasets/llava/02_val_chatml_filter.json'

# 加载数据集
print("Loading dataset...")
dataset = MMDataset.from_json(anno_path)
print("初始数据集数量为:", len(dataset))

# 0.过滤无效图像和文本
# dataset = dataset.valid_data_filter()

# 1.配置CLIP过滤器
clip_config = CLIPFilterConfig(
    model_name="paddlemix/CLIP/CLIP-ViT-L-14-laion2B-s32B-b82K",
    threshold=0.15,  # 设置相似度阈值
    batch_size=2560,  # 批量大小
    save_images=False  # 控制是否保存低置信度图像
)

# 使用过滤器处理数据集并保存图片
dataset = dataset.image_clip_filter(config=clip_config)

# 2.根据对话数的百分位数过滤
# dataset = conversation_percentage_filter(dataset, min_percentile=5, max_percentile=95)

# 3.根据simhash/minhash过滤重复文本
# dataset = remove_text_duplicates(dataset, method="simhash", threshold=0.75, num_perm=256, print_duplicates=False, max_workers=24)

# 4.根据图像文件大小过滤
# dataset = image_filesize_filter(dataset)

# 5.图像哈希过滤
# dataset = image_hash_filter(dataset)

# 6.图像宽高比过滤
# dataset = image_ration_filter(dataset)

# 7.图像分辨率大小过滤
# dataset = dataset.image_resolution_filter()

# 8.会话长度过滤
# dataset = dataset.conversation_length_filter()

print("过滤后数据集数量为:", len(dataset))
print("Dataset validation complete.")
dataset.export_json(anno_path.replace('.json', '_filter1.json'))
# dataset.export_json("test.json")