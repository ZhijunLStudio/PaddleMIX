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



算子名称：conversation_percentage_filter

功能介绍：

conversation_percentage_filter 用于根据对话数量的百分位数范围，过滤数据集中对话数量过少或过多的条目。通过设定最小和最大百分位数，可以灵活控制保留的数据范围。


使用介绍：
# 配置参数
min_percentile = 5   # Minimum percentile
max_percentile = 95  # Maximum percentile
dataset = dataset.conversation_percentage_filter(min_percentile=min_percentile, max_percentile=max_percentile)
# 使用默认参数
dataset = dataset.conversation_percentage_filter()

可选参数：
min_percentile (float): 最小百分位数，表示保留数据集中对话数量大于等于该百分位数的条目。例如，min_percentile=5 表示保留对话数量大于等于第 5 百分位数的条目。
max_percentile (float): 最大百分位数，表示保留数据集中对话数量小于等于该百分位数的条目。例如，max_percentile=95 表示保留对话数量小于等于第 95 百分位数的条目。




算子名称：alphanumeric_ratio_filter

功能介绍：

alphanumeric_ratio_filter 算子用于根据文本中字母或数字字符占总字符数的比例过滤数据集中的样本。对于含有过多特殊字符或非字母数字内容的样本，该算子可以将其移除，以保证数据集质量。

使用介绍：
# 应用字母数字比例过滤算子
min_ratio = 0.25  # 设置最小字母数字比例
max_ratio = 0.75  # 设置最大字母数字比例
dataset = dataset.alphanumeric_ratio_filter(min_ratio=min_ratio, max_ratio=max_ratio)
dataset = dataset.alphanumeric_ratio_filter()

可选参数：
min_ratio (float):样本中文本的最小字母或数字字符比例，默认值为 0.25。
max_ratio (float):样本中文本的最大字母或数字字符比例，默认值为 正无穷。


算子名称：average_line_length_filter

功能介绍：

average_line_length_filter 算子用于根据会话的 平均行长度 过滤数据集中的样本。通过设定每行文本的最小和最大长度范围，可以剔除那些行内容过短或过长的样本，以提高数据集的质量。

使用方法：

PYTHON
dataset = dataset.average_line_length_filter(min_length=15, max_length=50)
参数详情：


可选参数：

min_length (int):
每行文本的最小平均长度，默认值为 10。
max_length (float):
每行文本的最大平均长度，默认值为 无穷大。


算子名称：char_ngram_repetition_filter

功能介绍：

char_ngram_repetition_filter 算子用于根据会话中 字符 n-gram 的重复比例 过滤数据集。通过设定 n-gram 的长度以及重复比例的上下限，可以有效剔除重复内容过多或过少的样本，以提高数据集的多样性和质量。

使用方法：

PYTHON
dataset = dataset.char_ngram_repetition_filter(rep_len=10, min_ratio=0.1, max_ratio=0.4)

可选参数：

rep_len (int):
n-gram 的长度，默认值为 10。
min_ratio (float):
最小重复比例，默认值为 0.0。
max_ratio (float):
最大重复比例，默认值为 0.5。



算子名称：conversation_hash_filter

功能介绍：

conversation_hash_filter 算子用于对数据集中的对话内容进行去重，利用 SimHash 或其他哈希算法检测相似的对话，并根据设定的相似度阈值移除重复的对话条目。该算子支持并行处理，适用于大规模数据集的快速去重。

参数详情：

参数名称	类型	默认值	说明
method	str	"simhash"	去重方法，支持 "simhash"（默认）。
threshold	float	0.8	相似度阈值：对于 SimHash 表示汉明距离比例，值越高表示更严格的去重。
max_workers	int	8	并行处理的线程数，用于加速大规模数据集的去重。
输出：

过滤后的数据集 (MMDataset): 仅保留唯一的对话条目，移除了相似度超过阈值的重复对话。
功能详情：

文本预处理：

对对话文本进行清理，例如移除 <image> 占位符及多余的换行符。
重复检测算法：

SimHash:  模拟哈希：
计算每条对话的 SimHash 值，比较汉明距离来判断相似性。
使用设定的 threshold 控制去重的敏感度。


算子名称：image_clip_filter

功能介绍：

image_clip_filter 算子使用 CLIP 模型对数据集中的问答对进行过滤，根据图像与文本的相似度，移除低置信度的问答对。该算子还支持跳过包含坐标形式（例如 [x1, y1, x2, y2]）的问答对，并可选择保存低置信度的样本图像以便后续分析。


使用说明：
# 配置 CLIP 过滤参数
config = CLIPFilterConfig(
    threshold=0.3,  # 置信度阈值
    batch_size=8,  # 批量大小
    save_images=True,  # 是否保存低置信度图像
    save_dir="./filtered_images"  # 保存图像的目录
)

# 应用 image_clip_filter 算子
dataset = dataset.image_clip_filter(config=config)


可选参数：
model_name (str):
model_name （ str ）：
使用的 CLIP 模型名称，默认为 "paddlemix/CLIP/CLIP-ViT-L-14-laion2B-s32B-b82K"。
threshold (float):
图像与文本相似度的置信度阈值，低于此值的问答对将被过滤，默认值为 0.25。
batch_size (int):
批量大小，用于一次性处理的问答对数量，默认为 8。
save_images (bool):
是否保存低置信度的样本图像，默认为 False。
save_dir (str):
保存低置信度图像的目录，默认为 "./low_confidence_images"。


功能：
文本预处理：

对问答对中的问题和答案进行清理，移除 <image> 占位符以及多余的换行符，确保输入文本格式化为自然语言。
跳过坐标形式的问答对：

自动识别问答对中包含坐标形式（例如 [x1, y1, x2, y2]）的文本，并跳过这些问答对，避免干扰模型的处理。
CLIP 模型计算：

使用 CLIP 模型计算图像与问答对文本的相似度，低于设定阈值的问答对将被标记为低置信度。
过滤低置信度样本：

移除低置信度的问答对，保留高置信度的内容。
保存低置信度图像（可选）：

对低置信度的样本生成包含问答对和相似度的拼接图片，并保存到指定目录，便于后续分析。



算子名称：image_filesize_filter

功能介绍：

image_filesize_filter 用于过滤数据集中图像文件大小不符合要求的样本。用户可以通过设置图像文件的最小和最大大小（以 KB 为单位）来控制数据集的质量，移除异常或无效的图像文件。


# 应用图像文件大小过滤算子
dataset = dataset.image_filesize_filter(
    min_size_kb=10,  # 图像文件的最小大小（单位：KB）
    max_size_kb=1024  # 图像文件的最大大小（单位：KB）
)

可选参数
min_size_kb (Optional[float]):
图像文件的最小大小（单位：KB）。小于该大小的文件将被过滤。默认值为 10 KB。

max_size_kb (Optional[float]):
图像文件的最大大小（单位：KB）。大于该大小的文件将被过滤。默认值为 None，表示不限制最大大小。

功能：
文件大小检测：

使用 os.path.getsize 获取图像文件的大小，并转换为 KB。
检查文件大小是否在 [min_size_kb, max_size_kb] 范围内。


算子名称：image_hash_filter

功能介绍：

image_hash_filter 算子用于通过图像哈希值对数据集中的图像去重。支持多种哈希算法（如 phash、dhash 和 average_hash），通过比较图像的哈希值来检测重复内容，并移除重复的图像。

使用介绍：
# 应用图像哈希过滤算子
dataset = dataset.image_hash_filter(
    hash_method="phash"  # 使用 "phash" 方法（默认）
)


可选参数：
hash_method (Optional[str], 默认值: "phash"):
图像哈希算法类型，可选值包括：

"phash": 感知哈希（Perceptual Hash），适用于检测图像内容的相似性。
"dhash": 差异哈希（Difference Hash），计算图像的梯度变化。
"average_hash": 平均哈希，基于图像的平均像素值计算哈希。



功能详情：

图像哈希计算：

对每张图像计算哈希值，支持多种哈希算法（如 phash、dhash 和 average_hash）。
通过哈希值检测重复图像，保留唯一图像。


算子名称：image_ration_filter

功能介绍：

image_ration_filter 算子用于过滤数据集中宽高比不符合指定范围的图像。通过设置图像的最小和最大宽高比，可以移除异常比例的图像样本，确保数据集的图像比例符合预期。


使用说明：
# 应用宽高比过滤算子
dataset = dataset.image_ration_filter(
    min_ratio=0.333,  # 最小宽高比
    max_ratio=3.0     # 最大宽高比
)

参数说明：

min_ratio (Optional[float], 默认值: 0.333):
图像的最小宽高比。小于该值的图像将被过滤。

max_ratio (Optional[float], 默认值: 3.0):
图像的最大宽高比。大于该值的图像将被过滤。

功能详情：

宽高比计算：
计算宽高比 width / height。如果宽高比小于 min_ratio 或大于 max_ratio，则过滤掉该图像。


算子名称：image_resolution_filter

功能介绍：

image_resolution_filter 算子用于过滤数据集中分辨率（宽度和高度）不符合指定范围的图像样本。通过设置图像的最小宽度、高度和可选的最大宽度、高度，可以移除分辨率异常的图像，确保数据集符合预期。


使用介绍：
# 应用分辨率过滤算子
dataset = dataset.image_resolution_filter(
    min_width=112,   # 最小宽度
    min_height=112,  # 最小高度
    max_width=1920,  # 最大宽度（可选）
    max_height=1080  # 最大高度（可选）
)


参数说明：

dataset (MMDataset):
输入的数据集，包含图像路径及相关信息的样本。

min_width (Optional[float], 默认值: 112):
图像的最小宽度。小于该值的图像将被过滤。

min_height (Optional[float], 默认值: 112):
图像的最小高度。小于该值的图像将被过滤。

max_width (Optional[float], 默认值: None):
图像的最大宽度。大于该值的图像将被过滤。如果设置为 None，则不限制最大宽度。

max_height (Optional[float], 默认值: None):
图像的最大高度。大于该值的图像将被过滤。如果设置为 None，则不限制最大高度。

功能详情：

分辨率检查：检查图像是否符合设定的宽度和高度范围。


算子名称：llm_judge_filter

功能介绍：

llm_judge_filter 算子利用llm模型分析数据集中的问答对，根据模型的评分过滤掉质量较差的问答对。通过给定的评分标准，保留评分大于等于 3 的问答对，从而提高数据集的整体质量。

使用介绍：
filtered_dataset = dataset.llm_judge_filter(
    model_name="Qwen/Qwen2.5-7B",  # llm model name
    batch_size=1                     # Batch size for processing
)

参数说明：

model_name (str, 默认值: "Qwen/Qwen2.5-7B"):
使用的llm模型名称。

batch_size (int, 默认值: 1):
每次处理的问答对数量。

功能详情：

使用指定的llm模型，对每个问答对生成评价，提取评分值。
评分范围为 1 到 4，保留评分大于等于 3 的问答对。



算子名称：language_id_filter

功能介绍：

language_id_filter 算子通过 FastText 模型检测数据集中样本的语言，根据指定的语言列表和最小置信分数过滤数据集中的样本。用户可以灵活选择允许的语言，并移除不符合语言要求或置信分数过低的样本。

# 应用语言过滤算子
filtered_dataset = dataset.language_id_filter(
    lang=["en", "fr"],  # 允许的语言为英语和法语
    min_score=0.9       # 最小语言置信分数
)

参数说明：

dataset (MMDataset):
输入的数据集，包含样本的问答对和相关信息。

lang (Optional[Union[str, List[str]]], 默认值: None):
允许的语言代码，可以是单个字符串（如 "en" 表示英语）、字符串列表（如 ["en", "fr"] 表示英语和法语）或 None（不限制语言）。

min_score (float, 默认值: 0.8):
最小语言置信分数，置信分数低于该值的样本将被过滤。


功能详情：

语言检测：

使用 FastText 模型检测样本的语言 ID 和对应的置信分数。
过滤逻辑：

如果指定了语言列表，保留语言属于列表中且置信分数大于等于 min_score 的样本。
如果未指定语言，仅保留置信分数大于等于 min_score 的样本。
自动下载模型：

如果本地没有 FastText 模型文件，算子会自动从指定 URL 下载模型。


算子名称：maximum_line_length_filter

功能介绍：

maximum_line_length_filter 算子用于根据问答对中最大行长度的范围过滤数据集。通过设置最小和最大行长度，可以移除问题或答案内容过短或过长的样本，从而提高数据集的质量。

使用介绍：
# 应用最大行长度过滤算子
dataset = dataset.maximum_line_length_filter(
    min_length=10,  # 最小最大行长度
    max_length=128  # 最大最大行长度
)

参数说明：

min_length (Optional[int], 默认值: 10):
问答对中最大行长度的最小值。小于该长度的样本将被过滤。

max_length (Optional[float], 默认值: float('inf')):
问答对中最大行长度的最大值。大于该长度的样本将被过滤。

功能详情：

对每个问答对中的问题和答案计算长度，取其最大值作为最大行长度。如果最大行长度小于 min_length 或大于 max_length，则过滤该样本。


算子名称：special_characters_filter

功能介绍：

special_characters_filter 算子用于根据样本中特殊字符的比例过滤数据集。通过设置允许的最小和最大特殊字符比例，可以过滤掉特殊字符过多或过少的样本，以提高数据集的质量。


使用介绍：
# 应用特殊字符过滤算子
dataset = dataset.special_characters_filter(
    min_ratio=0.0,  # 最小特殊字符比例
    max_ratio=0.25  # 最大特殊字符比例
)


参数说明：

min_ratio (Optional[float], 默认值: 0.0):
样本中特殊字符比例的最小值。特殊字符比例低于该值的样本将被过滤。

max_ratio (Optional[float], 默认值: 0.25):
样本中特殊字符比例的最大值。特殊字符比例高于该值的样本将被过滤。

功能详情：
计算特殊字符占总字符数的比例。如果特殊字符比例小于 min_ratio 或大于 max_ratio，则过滤掉该样本。



算子名称：stopwords_ratio_filter

功能介绍：

stopwords_ratio_filter 算子用于根据样本中的停用词比例对数据集进行过滤。通过设置最小停用词比例，筛选出停用词比例大于或等于指定值的样本，从而优化数据集的语言特征。



使用介绍：
# 应用停用词比例过滤算子
dataset = dataset.stopwords_ratio_filter(
    min_ratio=0.25  # 最小停用词比例
)


参数说明：
min_ratio (Optional[float], 默认值: 0.25):
样本中停用词比例的最小值。停用词比例低于该值的样本将被过滤。

功能详情：


使用 NLTK 的 stopwords 资源获取英语停用词列表。对样本中的问答对内容进行分词，统计停用词的数量。如果停用词比例低于 min_ratio，则过滤掉该样本。


算子名称：text_action_filter

功能介绍：

text_action_filter 算子通过检测样本中的动词数量，根据指定的最小动词数量过滤数据集。该算子使用 spaCy 模型进行语言处理，支持基于英语的动词检测规则。


使用介绍：
# 应用动词数量过滤算子
filtered_dataset = dataset.text_action_filter(
    lang="en",  # 文本的语言
    min_action_num=2  # 最小动词数量
)


参数说明：


lang (str, 默认值: 'en'):
文本的语言。当前支持 'en'（英语）。

min_action_num (Optional[int], 默认值: 1):
样本中动词的最小数量。动词数量小于该值的样本将被过滤。

功能详情：

使用 spaCy 模型处理样本中的文本内容，提取动词。通过 pos_ 和 tag_ 属性检测动词。例如，VERB 表示动词的词性。如果样本中的动词数量小于 min_action_num，则过滤掉该样本。


算子名称：text_entity_dependency_filter

功能介绍：

text_entity_dependency_filter 算子通过检测样本中的实体依赖关系，根据每个实体的依赖边数量对数据集进行过滤。用户可以选择不同的筛选策略（any 或 all），并设置依赖边的最小数量，从而筛选出符合条件的样本。


# 应用实体依赖关系过滤算子
filtered_dataset = dataset.text_entity_dependency_filter(
    lang="en",               # 文本的语言
    min_dependency_num=2,    # 每个实体的最小依赖边数量
    any_or_all="any"         # 筛选策略：'any' 或 'all'
)


参数说明：

lang (str, 默认值: 'en'):
文本的语言。当前支持 'en'（英语）。

min_dependency_num (Optional[int], 默认值: 1):
每个实体的最小依赖边数量。实体的依赖边数量小于该值的样本将被过滤。

any_or_all (str, 默认值: 'any'):
筛选策略：

'any'：只要有一个实体满足条件即可。
'all'：所有实体都必须满足条件。
功能详情：

文本处理：

使用 spaCy 模型处理样本中的文本内容。通过 POS 和 Tag 的规则识别实体，例如名词、专有名词和代词。统计每个实体的依赖边数量，包括实体本身的依赖关系和其他词对实体的依赖关系。

根据用户指定的策略（any 或 all）以及最小依赖边数量筛选样本。


算子名称：token_num_filter

功能介绍：

token_num_filter 算子用于根据样本（会话）的 token 数量过滤数据集。通过设置最小和最大 token 数量范围，可以筛选出符合要求的样本，从而优化数据集的质量。


使用介绍：
# 应用 token 数量过滤算子
filtered_dataset = dataset.token_num_filter(
    tokenizer_model="Qwen/Qwen2.5-7B",  # 使用的 tokenizer 模型
    min_tokens=10,                      # 最小 token 数量
    max_tokens=512                      # 最大 token 数量
)


参数说明：

tokenizer_model (str, 默认值: "Qwen/Qwen2.5-7B"):
使用的 tokenizer 模型名称。例如 "Qwen/Qwen2.5-7B"。

min_tokens (Optional[int], 默认值: 10):
样本的最小 token 数量。样本中 token 数量小于该值的将被过滤。

max_tokens (Optional[int], 默认值: sys.maxsize):
样本的最大 token 数量。样本中 token 数量大于该值的将被过滤。

功能详情：


统计会话的 token 数量。如果样本的 token 数量小于 min_tokens 或大于 max_tokens，则过滤掉该样本。


算子名称：word_ngram_repetition_filter

功能介绍：

word_ngram_repetition_filter 算子通过计算样本中的词 n-gram 重复比例，根据指定的重复比例范围过滤数据集。该算子可以有效去除重复内容过多或不足的样本，从而提升数据集的多样性和质量。

使用介绍：
# 应用词 n-gram 重复比例过滤算子
filtered_dataset = dataset.word_ngram_repetition_filter(
    rep_len=10,         # n-gram 的长度
    min_ratio=0.0,      # 最小重复比例
    max_ratio=0.2       # 最大重复比例
)

参数说明：
rep_len (int, 默认值: 10):
n-gram 的长度。

min_ratio (Optional[float], 默认值: 0.0):
样本中词 n-gram 的最小重复比例。重复比例低于该值的样本将被过滤。

max_ratio (Optional[float], 默认值: 0.2):
样本中词 n-gram 的最大重复比例。重复比例高于该值的样本将被过滤。

功能详情：

词 n-gram 生成：

将样本中的文本按空格分词，并生成长度为 rep_len 的词 n-gram。
n-gram 重复比例计算：

统计每个 n-gram 的出现频率。
计算重复 n-gram 的比例。
过滤逻辑：

如果样本中的 n-gram 重复比例不在 [min_ratio, max_ratio] 范围内，则过滤掉该样本。



基础分析算子（5个）
功能描述：

count_data_statistics 用于统计数据集的基本信息，包括有效项和无效项的数量。通过验证每个样本的字段完整性（如 image 和 conversations），筛选出有效项，并计算对话相关的统计数据。

返回一个包含以下统计信息的字典：
total_records: 数据集的总记录数。
unique_images: 数据集中唯一图片的数量。
total_conversations: 所有样本的总对话数量（Q&A 对数量）。
max_conversations: 单个样本中最大对话数量。
min_conversations: 单个样本中最小对话数量。
avg_conversations: 样本的平均对话数量。
invalid_item_count: 无效数据项的数量。

analyze_language_distribution分析数据集中语言的分布情况，并统计人类消息和助手消息的语言是否一致。
返回包含以下信息的字典：
human_message_count (int): 数据集中人类消息的总数量。
assistant_message_count (int): 数据集中助手消息的总数量。
mismatched_language_pairs_count (int): 人类消息和助手消息语言不一致的对话对数量。
languages_distribution (dict): 数据集中语言的分布情况（语言代码及其出现次数）。

validate_image_paths_in_dataset 函数验证数据集中图片路径的分布和存在性。
返回包含以下信息的字典：
total_images (int): 数据集中图片的总数量。
missing_images (int): 数据集中缺失图片的数量。
path_distribution (dict): 图片路径的分布情况（文件夹路径及其图片数量）。

detect_data_anomalies 函数检测数据集中的异常项，例如缺少必要字段或对话为空的样本。
返回包含以下信息的字典：
missing_field_count (int): 缺少必要字段的样本数量。
empty_conversation_count (int): 对话为空的样本数量。


run_token_analysis 函数分析数据集中问答对的 token 分布，包括高频和低频 token 的统计。
返回包含以下信息的字典：
human: 人类消息的 token 分析结果。
total_tokens (int): 总 token 数量。
high_freq_tokens (Counter): 高频 token 的分布。
low_freq_tokens (Counter): 低频 token 的分布。
assistant: 助手消息的 token 分析结果（同上）。


上述5个算子使用方法：
analysis_flags = {
    "data_statistics": True,
    "field_distribution": True,
    "path_validation": True,
    "anomaly_detection": True,
    "token_analysis": True
}

# Run the base analysis
results = dataset.run_analysis_pipeline(analysis_flags=analysis_flags, output_dir="analysis_results")




算子名称：description_analysis

功能介绍：

description_analysis 算子用于分析数据集中问答对的内容，并提取对话中提到的属性信息（如颜色、形状、位置、大小、方向等）。通过调用指定的模型对对话内容进行分析，生成结构化的属性数据，并对结果进行清洗和统计。

参数说明：

model_name (str, 默认值: "Qwen/Qwen2.5-7B"):
指定用于分析对话内容的模型名称。

batch_size (int, 默认值: 1):
处理数据时的批处理大小。

功能详情：

对话内容处理：

将数据集中每个样本的所有问答对拼接为完整的对话内容。
构造输入模型的提示模板（prompt），引导模型生成结构化的属性数据。
模型推理：

使用指定的模型对拼接的对话内容进行分析，提取出颜色、形状、位置、大小、方向、关系、动作和类别等属性。
结果解析与清洗：

解析模型输出的 JSON 格式数据。
对缺失的属性填充默认值（如 "N/A"）。
统计与过滤：

对提取的属性数据进行清洗和统计，生成每个类别的高频统计信息。



算子名称：quality_analysis

功能介绍：

quality_analysis 算子用于评估数据集中文本描述（caption）的质量。通过对多轮对话进行分析，使用预定义的评估指标（如图文匹配、对象描述详细程度等），生成每个样本的评估结果。



使用介绍：
quality_analysis_flags = {
    "image_text_matching": True,  # 是否启用图文匹配分析
    "object_detail_fulfillment": False,  # 是否启用对象详细描述分析
    "caption_text_quality": False,  # 是否启用文字质量分析
    "semantic_understanding": False,  # 是否启用语义理解分析
}

# Apply the image caption metrics analysis operator
dataset_results = dataset.quality_analysis(
    model_name="Qwen/Qwen2.5-7B",  # Specify the model name
    quality_analysis_flags=quality_analysis_flags  # Pass the analysis flags
)


参数说明：

model_name (str, 默认值: "Qwen/Qwen2.5-7B"):
指定用于分析对话内容的模型名称。


评估指标：

图文匹配（image_text_matching）：评估文本描述是否准确反映图像的主要特征。
对象详细描述（object_detail_fulfillment）：评估文本是否详细描述了对象的颜色、形状、位置等。
文本质量（caption_text_quality）：评估文本的语法正确性、单词多样性、流畅性等。
语义理解（semantic_understanding）：评估文本是否提供了额外的语义信息。


算子名称

lda_topic_clustering

算子功能

lda_topic_clustering 是一个用于对对话文本进行主题建模和降维可视化的算子。本算子主要功能包括：

主题建模（LDA）：基于 Latent Dirichlet Allocation (LDA) 算法，分析数据集中每段对话文本的主题分布，提取隐藏的语义主题。
降维与可视化（T-SNE）：将高维的主题分布通过 t-SNE 算法降维到二维，并生成可视化散点图，展示不同对话在语义主题上的聚类情况。
输出主题分布矩阵、降维后的二维坐标以及每条文本的主要主题类别。

使用介绍：
results = lda_topic_clustering(
    dataset=dataset,
    num_topics=5,                
    tsne_perplexity=30,          
    tsne_learning_rate=200,      
    tsne_n_iter=1000,             
    random_state=42,              
    output_plot="lda_tsne_plot.png"  
)

参数说明


num_topics
可选参数，默认值为 5。用于指定主题建模中要提取的主题数量。例如，如果设置为 5，LDA 会尝试从数据中提取 5 个主要主题。

tsne_perplexity
可选参数，默认值为 30。t-SNE 算法的困惑度参数，控制降维时的平衡性。适合样本数量在 5-50 * perplexity 的范围内。

tsne_learning_rate
可选参数，默认值为 200。t-SNE 算法的学习率，控制优化过程的速度。

tsne_n_iter
可选参数，默认值为 1000。t-SNE 算法的迭代次数。更多的迭代次数可能带来更好的降维效果，但计算时间会增加。

random_state
可选参数，默认值为 42。用于设置随机种子，保证结果的可重复性。

output_plot
可选参数，默认值为 "lda_tsne_plot.png"。指定 T-SNE 降维结果的可视化散点图的保存路径。

功能详情

文本数据预处理
算子首先从数据集中提取对话文本，包括问题和答案部分，将它们拼接为一段完整的文本。
这些文本数据会被转换为词频矩阵（Bag of Words）以供主题建模使用。

LDA主题建模
使用 Latent Dirichlet Allocation (LDA) 算法对文本数据进行主题建模，提取给定数量的主题（由 num_topics 参数指定）。
每段文本的主题分布以概率矩阵的形式输出，矩阵大小为 [n_samples, num_topics]。

T-SNE降维
使用 t-SNE 算法对 LDA 生成的主题分布进行降维，将高维数据映射到二维空间。
每条文本在二维空间中的位置由其主题分布决定，从而可以通过可视化展示不同文本的主题聚类情况。

可视化
算子生成一张散点图，每个点代表一条对话文本，不同颜色表示不同主题聚类。
图像文件默认保存为 "lda_tsne_plot.png"，但可以通过 output_plot 参数指定路径。

输出结果

lda_result：每条对话文本的主题分布矩阵，表示各文本属于不同主题的概率。
tsne_result：每条对话文本在二维空间的坐标，用于可视化。
topics：每条对话文本的主要主题（取决于主题分布矩阵中最大的概率值对应的主题索引）。