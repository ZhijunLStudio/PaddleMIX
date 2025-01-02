算子名称：llava_convert
功能介绍：用于将llava数据集转换为paddlemix标准格式，处理图像路径、对话配对，并过滤无效数据。
使用介绍：

# 数据加载
dataset = MMDataset.from_json(anno_path)
# 算子使用
dataset = dataset.llava_convert()
dataset = dataset.llava_convert(image_path_prefix='datasets/llava/valid_images/')

参数详情：
输入：
数据集 (MMDataset): 包含以下字段的原始数据集：
image: 图像的相对路径。
conversations: llava格式的对话列表。
可选参数：image_path_prefix：图像路径的前缀，用于拼接json中的image路径。
输出：
数据集 (MMDataset): 转换后的数据集，包含以下字段：
image: 图像的绝对路径。
conversations: paddlemix支持的对话列表，已过滤无效数据。




算子名称：valid_data_filter

功能介绍：

valid_data_filter 用于过滤数据集中无效的图像和对话数据，确保数据符合使用要求。通过验证图像是否可加载和对话结构是否合规，清理数据集中的异常条目。

使用介绍：

PYTHON
# 数据加载
dataset = MMDataset.from_json(anno_path)

# 使用算子过滤无效数据
dataset = dataset.valid_data_filter()
参数详情：

输入：

数据集 (MMDataset): 包含以下字段的原始数据集：
image: 图像文件的路径。
conversations: 对话列表（每条对话应为 [human_message, gpt_message] 的结构）。
输出：

数据集 (MMDataset): 过滤后的数据集，包含以下字段：
image: 可加载的图像路径。
conversations: 符合格式要求的对话列表。
功能详情：

图像有效性检查：

验证 image 字段指定的图片文件是否存在且可加载。
无法加载的图片将被过滤。
对话合规性检查：

确保 conversations 为有效的列表结构。
每条对话必须是包含两部分 [human_message, gpt_message] 的列表或元组。
对话内容不能包含关键字 USER 或 ASSISTANT。
确保对话内容为非空字符串。



算子名称：conversation_length_filter

功能介绍：

conversation_length_filter 用于过滤数据集中会话内容过长的条目。通过指定最大字符长度限制（默认 2048），清除超出长度限制的会话。

使用介绍：

PYTHON
# 数据加载
dataset = MMDataset.from_json(anno_path)

# 使用算子过滤超长会话
dataset = dataset.conversation_length_filter(max_length=2048)
可选参数：

max_length (int): 会话的最大字符长度，默认值为 2048。

会话长度检查：

将 conversations 中的所有内容拼接成一个字符串。
去除 <image> 占位符及其换行符。
检查拼接后的字符串长度是否小于 max_length。