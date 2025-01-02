算子名称：llava_convert
功能介绍：用于将llava数据集转换为paddlemix标准格式，处理图像路径、对话配对，并过滤无效数据。
使用介绍：

dataset = MMDataset.from_json(anno_path)
# 算子使用
dataset = dataset.llava_convert()
dataset = dataset.llava_convert(image_path_prefix='datasets/llava/valid_images/')

参数详情：
输入：
数据集 (MMDataset): 包含以下字段的原始数据集：
image: 图像的相对路径。
conversations: 人类和GPT消息的对话列表。
可选参数：image_path_prefix：图像路径的前缀，用于拼接json中的image路径。
输出：
数据集 (MMDataset): 转换后的数据集，包含以下字段：
image: 图像的绝对路径。
conversations: 人类和GPT消息的对话列表，已过滤无效数据。


