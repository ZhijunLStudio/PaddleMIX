# PaddleMix 算子文档

## 目录
- [1. 转换算子](#1-转换算子)
  - [1.1 llava转换算子](#11-llava转换算子)
    - [1.1.1 llava_convert](#111-llava_convert)
      - [1.1.1 llava_convert](#111-llava_convert)
- [2. 过滤算子](#2-过滤算子)
  - [2.1 基础过滤算子](#21-基础过滤算子)
    - [2.1.1 image_compliance_operator](#211-image_compliance_operator)
      - [2.1.2 conversation_compliance_operator](#212-conversation_compliance_operator)
  - [2.2 文本过滤算子](#22-文本过滤算子)
    - [2.2.1 conversation_length_filter](#221-conversation_length_filter)
    - [2.2.2 conversation_percentage_filter](#222-conversation_percentage_filter)
    - [2.2.3 alphanumeric_ratio_filter](#223-alphanumeric_ratio_filter)
    - [2.2.4 average_line_length_filter](#224-average_line_length_filter)
    - [2.2.5 char_ngram_repetition_filter](#225-char_ngram_repetition_filter)
    - [2.2.6 conversation_hash_filter](#226-conversation_hash_filter)
    - [2.2.7 language_id_filter](#227-language_id_filter)
    - [2.2.8 maximum_line_length_filter](#228-maximum_line_length_filter)
    - [2.2.9 special_characters_filter](#229-special_characters_filter)
  - [2.3 图像过滤算子](#23-图像过滤算子)
    - [2.3.1 image_clip_filter](#231-image_clip_filter)
    - [2.3.2 image_filesize_filter](#232-image_filesize_filter)
    - [2.3.3 image_hash_filter](#233-image_hash_filter)
    - [2.3.4 image_ration_filter](#234-image_ration_filter)
    - [2.3.5 image_resolution_filter](#235-image_resolution_filter)
    - [2.3.6 llm_judge_filter](#236-llm_judge_filter)

## 1. 转换算子

### 1.1 llava转换算子

#### 1.1.1 llava_convert

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
- 输入：llava 格式对话列表的原始数据集，形如：
```json
[
  {
    "id": "000000033471",
    "image": "coco/train2017/000000033471.jpg",
    "conversations": [
        {
        "from": "human",
        "value": "<image>\nWhat are the colors of the bus in the image?"
      },
      {
        "from": "gpt",
        "value": "The bus in the image is white and red."
      }
    ]
  }
]```

- 输出：paddlemix 支持的对话列表的数据集，形如:
```json
[
  {
    "image": "image_path_prefix/coco/train2017/000000033471.jpg",
    "conversations": [
      [
        "<image>\nWhat are the colors of the bus in the image?",
        "The bus in the image is white and red."
      ]
    ]
  }
]```

## 2. 过滤算子


过滤算子中的输入和输出都是包含 image 和 conversations（paddlemix 支持的对话列表）的数据集，
并且所有算子都可使用默认参数直接调用
调用示例：`dataset = dataset.filter_name()`
另外，训练前建议先使用2.1的基础过滤算子进行异常数据清洗，防止训练过程中报错。

### 2.1 基础过滤算子

#### 2.1.1 valid_data_filter

`valid_data_filter()` 下联合了 `image_compliance_operator` 以及 `conversation_compliance_operator`算子，分别用于图像和文本的无效数据过滤，通过调用 `valid_data_filter()` 同时过滤这两种模态的异常数据。

**使用示例**:
```python
dataset = dataset.valid_data_filter()
```

##### 2.1.1.1 image_compliance_operator

**功能介绍**:  
过滤数据集中无效的图像数据，确保数据符合使用要求。

**功能详情**:
- 图像有效性检查：验证图片文件存在且可加载

##### 2.1.1.2 conversation_compliance_operator

**功能介绍**:  
过滤数据集中无效的对话数据，确保数据符合使用要求。

**功能详情**:
- 对话合规性检查：
  - 确保 conversations 为有效的列表结构
  - 每条对话必须是 [human_message, gpt_message] 格式
  - 对话内容不能包含 USER 或 ASSISTANT 关键字
  - 确保对话内容为非空字符串



### 2.2 文本过滤算子

#### 2.2.1 conversation_length_filter

**功能介绍**:  
过滤数据集中会话内容过长的条目。

**功能详情**:
- 将 conversations 中的所有内容拼接成一个字符串
- 去除 <image> 占位符及其换行符
- 检查拼接后的字符串长度是否小于 max_length

**参数说明**:
- `max_length` (int, 默认值: 2048): 会话的最大字符长度

**使用示例**:
```python
dataset = dataset.conversation_length_filter(max_length=2048)
```

#### 2.2.2 average_line_length_filter

**功能介绍**:  
根据会话的平均行长度过滤数据集中的样本。

**参数说明**:
- `min_length` (int, 默认值: 10): 每行文本的最小平均长度
- `max_length` (float, 默认值: float('inf')): 每行文本的最大平均长度

**使用示例**:
```python
dataset = dataset.average_line_length_filter(
    min_length=15,  
    max_length=50  
)
```

#### 2.2.3 conversation_percentage_filter 

**功能介绍**:  
根据对话数量的百分位数范围，过滤数据集中对话数量过少或过多的条目。

**参数说明**:
- `min_percentile` (float, 默认值: 5): 最小百分位数，保留对话数量大于等于该百分位数的条目
- `max_percentile` (float, 默认值: 95): 最大百分位数，保留对话数量小于等于该百分位数的条目

**使用示例**:
```python
dataset = dataset.conversation_percentage_filter(
    min_percentile=5, 
    max_percentile=95 
)
```

#### 2.2.4 token_num_filter

**功能介绍**:  
用于根据会话的 token 数量过滤数据集。

**功能详情**:
- 加载指定的 tokenizer 模型
- 统计会话的 token 数量
- 如果样本的 token 数量小于 min_tokens 或大于 max_tokens，则过滤掉该样本

**参数说明**:
- `tokenizer_model` (str, 默认值: "Qwen/Qwen2.5-7B"): 使用的 tokenizer 模型名称
- `min_tokens` (int, 默认值: 10): 样本的最小 token 数量
- `max_tokens` (int, 默认值: sys.maxsize): 样本的最大 token 数量

**使用示例**:
```python
filtered_dataset = dataset.token_num_filter(
    tokenizer_model="Qwen/Qwen2.5-7B",  
    min_tokens=10,                   
    max_tokens=512                      
)
```


#### 2.2.5 alphanumeric_ratio_filter

**功能介绍**:  
根据文本中字母或数字字符占总字符数的比例过滤数据集中的样本。

**参数说明**:
- `min_ratio` (float, 默认值: 0.25): 样本中文本的最小字母或数字字符比例
- `max_ratio` (float, 默认值: float('inf')): 样本中文本的最大字母或数字字符比例

**使用示例**:
```python
dataset = dataset.alphanumeric_ratio_filter(
    min_ratio=0.25,  
    max_ratio=0.75  
)
```

#### 2.2.6 stopwords_ratio_filter

**功能介绍**:  
根据样本中的停用词比例对数据集进行过滤，通过设置最小停用词比例，筛选出停用词比例大于或等于指定值的样本。

**功能详情**:
- 使用 NLTK 的 stopwords 资源获取英语停用词列表
- 对样本中的问答对内容进行分词，统计停用词的数量
- 如果停用词比例低于 min_ratio，则过滤掉该样本

**参数说明**:
- `min_ratio` (float, 默认值: 0.25): 样本中停用词比例的最小值。停用词比例低于该值的样本将被过滤。

**使用示例**:
```python
dataset = dataset.stopwords_ratio_filter(
    min_ratio=0.25 
)
```


#### 2.2.7 text_action_filter

**功能介绍**:  
通过检测样本中的动词数量，根据指定的最小动词数量过滤数据集。使用 spaCy 模型进行语言处理，支持基于英语的动词检测规则。

**功能详情**:
- 使用 spaCy 模型处理样本中的文本内容，提取动词
- 通过 pos_ 和 tag_ 属性检测动词，例如 VERB 表示动词的词性
- 如果样本中的动词数量小于 min_action_num，则过滤掉该样本

**参数说明**:
- `lang` (str, 默认值: 'en'): 文本的语言。当前支持 'en'（英语）
- `min_action_num` (int, 默认值: 1): 样本中动词的最小数量。动词数量小于该值的样本将被过滤。

**使用示例**:
```python
filtered_dataset = dataset.text_action_filter(
    lang="en",  
    min_action_num=2  
)
```



#### 2.2.8 text_entity_dependency_filter

**功能介绍**:  
通过检测样本中的实体依赖关系，根据每个实体的依赖边数量对数据集进行过滤。用户可以选择不同的筛选策略（any 或 all），并设置依赖边的最小数量。

**功能详情**:  
- 使用 spaCy 模型处理样本中的文本内容
- 通过 POS 和 Tag 的规则识别实体，例如名词、专有名词和代词
- 统计每个实体的依赖边数量，包括实体本身的依赖关系和其他词对实体的依赖关系
- 根据用户指定的策略（any 或 all）以及最小依赖边数量筛选样本

**参数说明**:
- `lang` (str, 默认值: 'en'): 文本的语言。当前支持 'en'（英语）
- `min_dependency_num` (int, 默认值: 1): 每个实体的最小依赖边数量
- `any_or_all` (str, 默认值: 'any'): 筛选策略，可选值为 'any'（只要有一个实体满足条件）或 'all'（所有实体都必须满足条件）

**使用示例**:
```python
filtered_dataset = dataset.text_entity_dependency_filter(
    lang="en",              
    min_dependency_num=2,   
    any_or_all="any"      
)
```


### 2.2.9 char_ngram_repetition_filter

**功能介绍**:
根据会话中字符 n-gram 的重复比例过滤数据集。

**功能详情**:
- 将样本中的文本按字符级别切分，并生成长度为 rep_len 的字符 n-gram。
- 统计每个 n-gram 的出现频率。
- 计算重复 n-gram 的比例，即 n-gram 出现频率大于 1 的 n-gram 数量占总 n-gram 数量的比例。
- 如果样本中的字符 n-gram 重复比例不在 [min_ratio, max_ratio] 范围内，则过滤掉该样本。

**参数说明**:
rep_len (int, 默认值: 10): n-gram 的长度
min_ratio (float, 默认值: 0.0): 最小重复比例
max_ratio (float, 默认值: 0.5): 最大重复比例

使用示例:
```python
dataset = dataset.char_ngram_repetition_filter(
    rep_len=10, 
    min_ratio=0.1, 
    max_ratio=0.4
)
```

#### 2.2.10 word_ngram_repetition_filter

**功能介绍**:  
通过计算样本中的词 n-gram 重复比例，根据指定的重复比例范围过滤数据集。

**功能详情**:  
- 将样本中的文本按空格分词，并生成长度为 rep_len 的词 n-gram
- 统计每个 n-gram 的出现频率
- 计算重复 n-gram 的比例
- 如果样本中的 n-gram 重复比例不在 [min_ratio, max_ratio] 范围内，则过滤掉该样本

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


#### 2.2.11 conversation_hash_filter

**功能介绍**:  
`conversation_hash_filter` 是一个用于去除重复问答对的算子，它统一调用 `simhash_duplicate_operator` 或 `minhash_duplicate_operator` 算子进行处理。

**参数说明**:
- `method` (str, 默认值: "simhash"): 去重方法
  - "simhash": 使用 SimHash 算法（基于汉明距离）检测文本重复。
  - "minhash": 使用 MinHashLSH 算法（基于 Jaccard 相似度）检测文本重复。
- `threshold` (float, 默认值: 0.8): 相似度阈值，影响去重的严格程度：
  - SimHash 汉明距离的比例，1 - threshold 是允许的最大汉明距离。例如，threshold=0.8 表示允许最多 20% 的汉明距离。
- `num_perm` (int, 默认值: 128):
  - MinHash 的置换次数，仅在 method="minhash" 时有效。值越大，MinHash 签名的精度越高，但计算开销也会增加。


**使用示例**:
```python
dataset = dataset.conversation_hash_filter(
    method="simhash",
    threshold=0.8,
)
```

#### 2.2.11.1 simhash_duplicate_operator

**功能介绍**:  
使用simhash算法对问答对进行去重。

#### 2.2.11.2 minhash_duplicate_operator

**功能介绍**:  
使用minhash算法对问答对进行去重。


#### 2.3.12 llm_judge_filter

**功能介绍**:  
利用 LLM 模型分析数据集中的问答对，根据模型的评分过滤掉质量较差的问答对。

**参数说明**:
- `model_name` (str, 默认值: "Qwen/Qwen2.5-7B"): 使用的 LLM 模型名称
- `batch_size` (int, 默认值: 1): 每次处理的问答对数量

**功能详情**:
- 使用指定的 LLM 模型对每个问答对生成评价并提取评分
- 评分范围为 1 到 4，保留评分大于等于 3 的问答对

**使用示例**:
```python
filtered_dataset = dataset.llm_judge_filter(
    model_name="Qwen/Qwen2.5-7B",
    batch_size=1
)
```



#### 2.3.6 llm_judge_filter

**功能介绍**:  
利用 LLM 模型分析数据集中的问答对，根据模型的评分过滤掉质量较差的问答对。

**参数说明**:
- `model_name` (str, 默认值: "Qwen/Qwen2.5-7B"): 使用的 LLM 模型名称
- `batch_size` (int, 默认值: 1): 每次处理的问答对数量

**功能详情**:
- 使用指定的 LLM 模型对每个问答对生成评价并提取评分
- 评分范围为 1 到 4，保留评分大于等于 3 的问答对

**使用示例**:
```python
filtered_dataset = dataset.llm_judge_filter(
    model_name="Qwen/Qwen2.5-7B",
    batch_size=1
)
```



### 2.3 图像过滤算子

#### 2.3.1 image_filesize_filter

**功能介绍**:  
过滤数据集中图像文件大小不符合要求的样本。

**参数说明**:
- `min_size_kb` (float, 默认值: 10): 图像文件的最小大小（KB）
- `max_size_kb` (float, 默认值: None): 图像文件的最大大小（KB）

**使用示例**:
```python
dataset = dataset.image_filesize_filter(
    min_size_kb=10,
    max_size_kb=1024
)
```

#### 2.3.2 image_ration_filter

**功能介绍**:  
过滤数据集中宽高比不符合指定范围的图像。

**参数说明**:
- `min_ratio` (float, 默认值: 0.333): 图像的最小宽高比
- `max_ratio` (float, 默认值: 3.0): 图像的最大宽高比

**使用示例**:
```python
dataset = dataset.image_ration_filter(
    min_ratio=0.333,
    max_ratio=3.0
)
```

#### 2.3.3 image_resolution_filter

**功能介绍**:  
过滤数据集中分辨率不符合指定范围的图像样本。

**参数说明**:
- `min_width` (float, 默认值: 112): 图像的最小宽度
- `min_height` (float, 默认值: 112): 图像的最小高度
- `max_width` (float, 默认值: None): 图像的最大宽度
- `max_height` (float, 默认值: None): 图像的最大高度

**使用示例**:
```python
dataset = dataset.image_resolution_filter(
    min_width=112,
    min_height=112,
    max_width=1920,
    max_height=1080
)
```

#### 2.3.4 image_hash_filter

**功能介绍**:  
通过图像哈希值对数据集中的图像去重。

**参数说明**:
- `hash_method` (str, 默认值: "phash"): 图像哈希算法类型，支持 "phash"、"dhash" 和 "average_hash"

**功能详情**:
- 支持多种哈希算法：
  - phash: 感知哈希，适用于检测图像内容相似性
  - dhash: 差异哈希，计算图像梯度变化
  - average_hash: 平均哈希，基于图像平均像素值计算

**使用示例**:
```python
dataset = dataset.image_hash_filter(hash_method="phash")
```


### 2.4 图文过滤算子

#### 2.4.1 image_clip_filter

**功能介绍**:  
使用 CLIP 模型对数据集中的问答对进行过滤，根据图像与文本的相似度移除低置信度的问答对。

**功能详情**:
- 文本预处理：对问答对进行清理，移除占位符和多余换行符
- 自动跳过包含坐标形式的问答对
- 使用 CLIP 模型计算图像-问答对相似度
- 可选保存低置信度样本图像

**参数说明**:
- `model_name` (str, 默认值: "paddlemix/CLIP/CLIP-ViT-L-14-laion2B-s32B-b82K"): 使用的 CLIP 模型名称
- `threshold` (float, 默认值: 0.25): 图像与文本相似度的置信度阈值
- `batch_size` (int, 默认值: 8): 批量处理的问答对数量
- `save_images` (bool, 默认值: False): 是否保存低置信度的样本图像
- `save_dir` (str, 默认值: "./low_confidence_images"): 保存低置信度图像的目录


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