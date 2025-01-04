# PaddleMix 算子文档

## 目录
- [1. 转换算子](#1-转换算子)
- [2. 过滤算子](#2-过滤算子)
  - [2.1 基础过滤算子](#21-基础过滤算子)
  - [2.2 文本过滤算子](#22-文本过滤算子)
  - [2.3 图像过滤算子](#23-图像过滤算子)
- [3. 分析算子](#3-分析算子)
  - [3.1 基础分析算子](#31-基础分析算子)
  - [3.2 高级分析算子](#32-高级分析算子)
- [4. 可视化算子](#4-可视化算子)
- [5. 生成算子](#5-生成算子)

## 1. 转换算子

### 1.1 llava_convert

**功能介绍**:  
将 llava 数据集转换为 paddlemix 标准格式，处理图像路径、对话配对，并过滤无效数据。

**参数说明**:
- `image_path_prefix` (str, 可选): 图像路径的前缀，用于拼接 json 中的 image 路径

**使用示例**:
```python
# 加载json格式的数据集
dataset = MMDataset.from_json(anno_path)
# 基础用法
dataset = dataset.llava_convert()
# 指定图像路径前缀，添加到转换后的数据集中
dataset = dataset.llava_convert(image_path_prefix='datasets/llava/valid_images/')
```

**输入输出**:
- 输入：包含 image（图像相对路径）和 conversations（llava 格式对话列表）的原始数据集
- 输出：包含 image（图像绝对路径）和 conversations（paddlemix 支持的对话列表）的转换后数据集

## 2. 过滤算子

**输入输出**:
- 输入：
- 输出：过滤后的MMDataset

### 2.1 基础过滤算子

#### 2.1.1 image_compliance_operator

**功能介绍**:  
过滤数据集中无效的图像数据，确保数据符合使用要求。

**功能详情**:
- 图像有效性检查：验证图片文件存在且可加载


#### 2.1.2 conversation_compliance_operator

**功能介绍**:  
过滤数据集中无效的对话数据，确保数据符合使用要求。

**功能详情**:
- 对话合规性检查：
  - 确保 conversations 为有效的列表结构
  - 每条对话必须是 [human_message, gpt_message] 格式
  - 对话内容不能包含 USER 或 ASSISTANT 关键字
  - 确保对话内容为非空字符串

注：`image_compliance_operator` 以及 `conversation_compliance_operator` 通过 `valid_data_filter()` 调用，因此无需单独调用。

**使用示例**:
```python
dataset = dataset.valid_data_filter()
```


### 2.2 文本过滤算子


#### 2.2.1 conversation_length_filter

**功能介绍**:  
过滤数据集中会话内容过长的条目。

**参数说明**:
- `max_length` (int, 默认值: 2048): 会话的最大字符长度

**使用示例**:
```python
dataset = dataset.conversation_length_filter(max_length=2048)
```

#### 2.2.2 conversation_percentage_filter

**功能介绍**:  
根据对话数量的百分位数范围过滤数据集中的条目。

**参数说明**:
- `min_percentile` (float): 最小百分位数
- `max_percentile` (float): 最大百分位数

**使用示例**:
```python
dataset = dataset.conversation_percentage_filter(
    min_percentile=5,
    max_percentile=95
)
```


#### 2.2.3 alphanumeric_ratio_filter

**功能介绍**:  
根据文本中字母或数字字符占总字符数的比例过滤样本。

**参数说明**:
- `min_ratio` (float, 默认值: 0.25): 最小字母数字比例
- `max_ratio` (float, 默认值: 正无穷): 最大字母数字比例

**使用示例**:
```python
dataset = dataset.alphanumeric_ratio_filter(
    min_ratio=0.25,
    max_ratio=0.75
)
```

#### 2.2.4 average_line_length_filter

**功能介绍**:  
根据会话的平均行长度过滤数据集中的样本。

**参数说明**:
- `min_length` (int, 默认值: 10): 每行文本的最小平均长度
- `max_length` (float, 默认值: 无穷大): 每行文本的最大平均长度

**使用示例**:
```python
dataset = dataset.average_line_length_filter(
    min_length=15,
    max_length=50
)
```

#### 2.2.5 char_ngram_repetition_filter

**功能介绍**:  
根据会话中字符 n-gram 的重复比例过滤数据集。

**参数说明**:
- `rep_len` (int, 默认值: 10): n-gram 的长度
- `min_ratio` (float, 默认值: 0.0): 最小重复比例
- `max_ratio` (float, 默认值: 0.5): 最大重复比例

**使用示例**:
```python
dataset = dataset.char_ngram_repetition_filter(
    rep_len=10,
    min_ratio=0.1,
    max_ratio=0.4
)
```

#### 2.2.6 simhash_duplicate_operator

**功能介绍**:  
使用 SimHash 或其他哈希算法检测相似对话并去重。

**参数说明**:
- `method` (str, 默认值: "simhash"): 去重方法
- `threshold` (float, 默认值: 0.8): 相似度阈值
- `max_workers` (int, 默认值: 8): 并行处理线程数


#### 2.2.6 minhash_duplicate_operator

**功能介绍**:  
使用 SimHash 或其他哈希算法检测相似对话并去重。

**参数说明**:
- `method` (str, 默认值: "simhash"): 去重方法
- `threshold` (float, 默认值: 0.8): 相似度阈值
- `max_workers` (int, 默认值: 8): 并行处理线程数



**使用示例**:
```python
dataset = dataset.conversation_hash_filter(
    method="simhash",
    threshold=0.8,
    max_workers=8
)
```

#### language_id_filter

**功能介绍**:  
使用 FastText 模型检测数据集中样本的语言，根据指定的语言列表和置信分数过滤样本。

**参数说明**:
- `lang` (str 或 List[str], 默认值: None): 允许的语言代码
- `min_score` (float, 默认值: 0.8): 最小语言置信分数

**使用示例**:
```python
dataset = dataset.language_id_filter(
    lang=["en", "fr"],
    min_score=0.9
)
```

#### maximum_line_length_filter

**功能介绍**:  
根据问答对中最大行长度的范围过滤数据集。

**参数说明**:
- `min_length` (int, 默认值: 10): 最小最大行长度
- `max_length` (float, 默认值: float('inf')): 最大最大行长度

**使用示例**:
```python
dataset = dataset.maximum_line_length_filter(
    min_length=10,
    max_length=128
)
```

#### special_characters_filter

**功能介绍**:  
根据样本中特殊字符的比例过滤数据集。

**参数说明**:
- `min_ratio` (float, 默认值: 0.0): 最小特殊字符比例
- `max_ratio` (float, 默认值: 0.25): 最大特殊字符比例

**使用示例**:
```python
dataset = dataset.special_characters_filter(
    min_ratio=0.0,
    max_ratio=0.25
)
```

#### stopwords_ratio_filter

**功能介绍**:  
根据样本中的停用词比例过滤数据集。

**参数说明**:
- `min_ratio` (float, 默认值: 0.25): 最小停用词比例

**使用示例**:
```python
dataset = dataset.stopwords_ratio_filter(
    min_ratio=0.25
)
```

#### text_action_filter

**功能介绍**:  
检测样本中的动词数量，根据最小动词数量过滤数据集。

**参数说明**:
- `lang` (str, 默认值: 'en'): 文本语言
- `min_action_num` (int, 默认值: 1): 最小动词数量

**使用示例**:
```python
dataset = dataset.text_action_filter(
    lang="en",
    min_action_num=2
)
```

#### text_entity_dependency_filter

**功能介绍**:  
通过检测样本中的实体依赖关系过滤数据集。

**参数说明**:
- `lang` (str, 默认值: 'en'): 文本语言
- `min_dependency_num` (int, 默认值: 1): 最小依赖边数量
- `any_or_all` (str, 默认值: 'any'): 筛选策略

**使用示例**:
```python
dataset = dataset.text_entity_dependency_filter(
    lang="en",
    min_dependency_num=2,
    any_or_all="any"
)
```

#### token_num_filter

**功能介绍**:  
根据样本的 token 数量过滤数据集。

**参数说明**:
- `tokenizer_model` (str, 默认值: "Qwen/Qwen2.5-7B"): tokenizer 模型名称
- `min_tokens` (int, 默认值: 10): 最小 token 数量
- `max_tokens` (int): 最大 token 数量

**使用示例**:
```python
dataset = dataset.token_num_filter(
    tokenizer_model="Qwen/Qwen2.5-7B",
    min_tokens=10,
    max_tokens=512
)
```

#### word_ngram_repetition_filter

**功能介绍**:  
根据样本中词 n-gram 的重复比例过滤数据集。

**参数说明**:
- `rep_len` (int, 默认值: 10): n-gram 长度
- `min_ratio` (float, 默认值: 0.0): 最小重复比例
- `max_ratio` (float, 默认值: 0.2): 最大重复比例

**使用示例**:
```python
dataset = dataset.word_ngram_repetition_filter(
    rep_len=10,
    min_ratio=0.0,
    max_ratio=0.2
)
```

### 2.3 图像过滤

#### image_clip_filter

**功能介绍**:  
使用 CLIP 模型对数据集中的问答对进行过滤，根据图像与文本的相似度过滤低置信度的样本。

**参数说明**:
- `model_name` (str, 默认值: "paddlemix/CLIP/CLIP-ViT-L-14-laion2B-s32B-b82K"): CLIP 模型名称
- `threshold` (float, 默认值: 0.25): 置信度阈值
- `batch_size` (int, 默认值: 8): 批处理大小
- `save_images` (bool, 默认值: False): 是否保存低置信度图像
- `save_dir` (str, 默认值: "./low_confidence_images"): 保存目录

**使用示例**:
```python
config = CLIPFilterConfig(
    threshold=0.3,
    batch_size=8,
    save_images=True,
    save_dir="./filtered_images"
)
dataset = dataset.image_clip_filter(config=config)
```

#### image_filesize_filter

**功能介绍**:  
根据图像文件大小过滤数据集。

**参数说明**:
- `min_size_kb` (float, 默认值: 10): 最小文件大小(KB)
- `max_size_kb` (float): 最大文件大小(KB)

**使用示例**:
```python
dataset = dataset.image_filesize_filter(
    min_size_kb=10,
    max_size_kb=1024
)
```

#### image_hash_filter

**功能介绍**:  
使用图像哈希算法对数据集中的图像去重。

**参数说明**:
- `hash_method` (str, 默认值: "phash"): 哈希算法类型

**使用示例**:
```python
dataset = dataset.image_hash_filter(
    hash_method="phash"
)
```

#### image_ration_filter

**功能介绍**:  
根据图像宽高比过滤数据集。

**参数说明**:
- `min_ratio` (float, 默认值: 0.333): 最小宽高比
- `max_ratio` (float, 默认值: 3.0): 最大宽高比

**使用示例**:
```python
dataset = dataset.image_ration_filter(
    min_ratio=0.333,
    max_ratio=3.0
)
```

#### image_resolution_filter

**功能介绍**:  
根据图像分辨率过滤数据集。

**参数说明**:
- `min_width` (float, 默认值: 112): 最小宽度
- `min_height` (float, 默认值: 112): 最小高度
- `max_width` (float): 最大宽度
- `max_height` (float): 最大高度

**使用示例**:
```python
dataset = dataset.image_resolution_filter(
    min_width=112,
    min_height=112,
    max_width=1920,
    max_height=1080
)
```

## 3. 分析算子

### 3.1 基础分析

#### count_data_statistics

**功能介绍**:  
统计数据集的基本信息。

**返回信息**:
- total_records: 总记录数
- unique_images: 唯一图片数量
- total_conversations: 总对话数量
- max_conversations: 最大对话数量
- min_conversations: 最小对话数量
- avg_conversations: 平均对话数量
- invalid_item_count: 无效数据项数量

#### analyze_language_distribution

**功能介绍**:  
分