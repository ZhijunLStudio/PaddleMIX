> 本文档为 PaddleMIX 开发者任务 RFC(**R**equest **F**or **C**omment) 模板

# 任务名

> RFC 文档相关记录信息

|              |                    |
| ------------ | -----------------  |
| 提交作者      |         李志军      |
| 提交时间      |       2024-11-13   |
| RFC 版本号    | v1.0               |
| 文件名        | lzj-data.md        |

## 1. 概述

### 1.1 相关背景

> 多模态大模型套件PaddleMIX集成了多模态大模型与飞桨框架的高性能技术，支持高效训练、推理与部署，广泛适用于金融、医疗、电商和教育等领域。
尽管PaddleMIX套件提供了丰富的模型库和极高的性能优势，用户在专有数据集上进行微调时，仍然需要对数据进行复杂的分析与清洗操作以保证模型的适用性与准确性。
本次大赛的目标是为飞桨多模态大模型套件（PaddleMIX）丰富数据分析与处理能力，通过开发高效的数据清洗、分析、过滤等组件，以更好地支持多模态大模型的训练，进而减少用户的数据预处理成本，提升开发效率。


### 1.2 功能目标

- 1. **数据分析模块**：实现单条数据和数据集的质量分析功能，针对图文多模态数据集中的图文相关性、文本质量、图片质量、合理性等多维度展开分析。
- 2. **数据清洗与过滤**：设计并实现一系列基础的数据清洗算子，支持数据过滤功能，并在数据集上验证有效性。
- 3. **数据配比与均衡分析**：通过数据处理策略，评估数据集整体的均衡性和多样性，以优化模型在LLaVA1.5 sft阶段的训练效果。


### 1.3 意义

> 该方案将为PaddleMIX套件提供端到端的数据分析与清洗能力，提升其在不同领域的适用性，并降低用户在多模态模型微调和部署中的数据处理成本，进一步拓展飞桨框架的多模态数据生态。

## 2.  方案背景

> 随着多模态数据在多领域的广泛应用，PaddleMIX多模态大模型套件在保证性能的同时，也面临数据质量不一致、数据多样性不足等问题。通过设计灵活的数据分析与清洗模块，将支持用户根据领域需求定制数据预处理流程，确保多模态大模型在实际应用中的可靠性和精确性。


## 3. 目标调研

在 **data-juicer** 中，LLaVA pretrain (LCS-558k) 数据过滤算子（压缩率 89.65%）包含以下几个关键过滤方法：

1. **图像宽高比过滤（image_aspect_ratio_filter）**  
   根据图像的宽高比（即宽度除以高度）筛选样本，宽高比范围为：  
   - `min_ratio = 0.333`  
   - `max_ratio = 3.0`  

2. **图像分辨率过滤（image_shape_filter）**  
   基于图像的分辨率（宽度和高度）筛选样本，最大宽度为 `727.88`，最大高度为 `606.24`。

3. **图像文件大小过滤（image_size_filter）**  
   通过图像的文件大小筛选样本，最大图像大小为 `124KB`。

4. **图像-文本相似度过滤（image_text_similarity_filter）**  
   使用 Hugging Face 的 CLIP 模型 `openai/clip-vit-base-patch32` 计算图像和文本之间的相似度，要求相似度分数不低于 `min_score = 0.20315419`。  
   - [clip-score:https://github.com/Taited/clip-score](https://github.com/Taited/clip-score)  
   - 该代码结构清晰，但缺乏鲁棒性检查和灵活性配置，适合基础功能，但扩展性和容错性有待提升。

5. **图像-文本匹配度过滤（image_text_matching_filter）**  
   使用 Hugging Face 的 BLIP 模型 `Salesforce/blip-itm-base-coco`，计算图像和文本之间的匹配得分，要求匹配分数不低于 `min_score = 0.44930778`。

### 两篇论文，提出了以下过滤方法：

#### 第一篇：https://arxiv.org/abs/2401.12225
- **CLIP SCORE过滤**：  
  选择图像和文本相似度得分前 x% 的样本，确保图像内容与描述的相关性。
  
- **对象检测过滤**：  
  使用 **Grounding DINO** 的零样本对象检测模型，依据以下三个条件进行数据过滤：  
  - **预测的logit分数**：根据Grounding DINO的logit分数判断对象检测的置信度，低于阈值的图像会被去除。  
  - **检测到的对象数量**：过滤掉对象数量过多或过少的图像。  
  - **对象的宽高比**：计算每个检测对象的宽高比，筛选掉不符合比例标准的图像。

文章使用弱监督技术对多个过滤方法的结果进行集成，具体方法如下：  
- **多数投票法**：集成多个过滤器的结果。  
- **弱监督算法**：使用 Snorkel 框架建模过滤器的准确性和相关性，从而优化标签集成结果。

过滤条件

1. **Grounding DINO 无对象检测**：如果未检测到任何对象，则该图像会被剔除。  
   - 约 38% 的图像将被剔除。  
2. **logit 分数过滤器**：选择前 30% 得分最高的图像。  
3. **长宽比过滤器**：丢弃长宽比小于 5% 或大于 95% 的图像。  
4. **图文匹配过滤**：确保图像与文本的匹配度通过 CLIP 分数过滤器验证，通常与 30%、50% 和 55% 的 CLIP L/14 配合使用。


#### 第二篇：https://arxiv.org/abs/2403.02677，代码仓库https://github.com/Victorwz/MLM_Filter
文章使用GPT4生成图像-文本数据，定义四个独立的评分指标，通过指令微调数据集，使得微调后的大模型可以对以下四块内容进行独立评估：
- **图文匹配度（ITM）**：评估图像和文本是否整体匹配，关注文本是否准确反映图像的主要内容。
- **细节完备度（ODF）**：检查文本是否准确描述了图像中对象的细节（如颜色、大小、位置等）。
- **文本质量（CTQ）**：评估文本的语言质量，包括语法、词汇多样性和流畅性。
- **语义理解（SU）**：判断文本是否提供图像无法直接呈现的附加语义信息（如地点、职业、社交关系等）。


## 4. 设计思路与实现方案

### 数据过滤

1. **图像去重**  
   基于可选的哈希算法，对数据集中的图像进行去重，移除重复的图像样本。

2. **不适当内容检测**  
   使用 NSFW 模型筛除含有敏感内容的图片。

3. **图文匹配去重**  
   基于 clip/blip 模型，对数据集中的图文匹配进行去重，移除重复的图文匹配样本。

4. **图像过滤**  
   移除分辨率过低、模糊、占用空间过小、长宽比不符合要求的图像样本。

5. **Grounding DINO 对象检测**  
   使用 Grounding DINO 模型，对数据集中的图像进行对象检测，过滤掉不适合训练的图像样本。

6. **重复字符过滤**  
   检测并过滤包含高比例重复字符的文本。

7. **文本质量过滤**  
   使用语言模型，对文本进行质量评估，过滤低质量的文本。

8. **特殊字符过滤**  
   删除文本中含有过多特殊字符的样本，确保语义的纯净度。

### 数据配比

1. **图像与问答对的配比**  
   实验得出图像与问答对的最优配比，通常为 1:2 或 1:3。

2. **基于问答难度的分层采样**  
   将数据集划分为简单、中等、困难三类，并对每一类进行采样。



### 开发思路，以图像去重为例：

1. **新建文件**  
   在 `paddlemix/datacopilot/ops/` 新建一个文件夹 `filter`，用来存放数据过滤相关的算子（如图像去重、文本去重等）。  
   在 `filter` 文件夹下新建 `remove_duplicates.py` 文件，将去重算子写在该文件中。

2. **使用 register 函数注册算子**  
   在去重函数上添加 `@register()` 完成注册。

3. **编写和使用算子示例代码**  
   在 `remove_duplicates.py` 中编写图像哈希去重的代码，使用 `register` 装饰器注册函数。  
   通过在 `MMDataset` 中调用该注册函数来完成去重操作。

```python
import os
import imagehash
from PIL import Image
from paddlemix.datacopilot.core import MMDataset
from paddlemix.datacopilot.ops import register

@register()
def remove_duplicates_by_hash(dataset: MMDataset, hash_method="phash") -> MMDataset:
    """
    使用图像哈希方法对数据集进行去重，移除重复的图像样本。

    参数：
    - dataset: MMDataset，待去重的数据集
    - hash_method: str, 可选的哈希算法类型 ('phash', 'dhash', 'average_hash', etc.)

    返回：
    - 去重后的数据集 MMDataset 实例
    """
    
    # 用于存储已计算过的图像哈希值
    hash_set = set()
    unique_items = []

    def compute_image_hash(item):
        # 提取图像的 URL 或路径
        image_path = item.get("image_url")
        if not image_path or not os.path.exists(image_path):
            return None

        # 打开图像并计算哈希
        try:
            with Image.open(image_path) as img:
                if hash_method == "phash":
                    hash_value = imagehash.phash(img)
                elif hash_method == "dhash":
                    hash_value = imagehash.dhash(img)
                elif hash_method == "average_hash":
                    hash_value = imagehash.average_hash(img)
                else:
                    raise ValueError("Unsupported hash method")

            return str(hash_value)
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None

    # 遍历数据集中的每个 item
    for item in dataset:
        # 计算图像的哈希值
        hash_value = compute_image_hash(item)
        
        # 判断哈希值是否已存在
        if hash_value and hash_value not in hash_set:
            hash_set.add(hash_value)
            unique_items.append(item)  # 添加到去重后的列表

    # 将去重后的样本重新载入为 MMDataset
    return MMDataset(unique_items)
```


## 5. 测试和验收的考量

> 在 MMDataset 中调用算子：
> 可以在 MMDataset 中直接调用这个去重算子，例如：

```python
from paddlemix.datacopilot.core import MMDataset

# 假设已经有一个json文件数据集
dataset = MMDataset.from_json('./path/to/your/json/file')
print(f"Original dataset size: {len(dataset)}")

# 执行去重操作
deduplicated_dataset = dataset.remove_duplicates_by_hash(hash_method="phash")
print(f"Deduplicated dataset size: {len(deduplicated_dataset)}")

# 将去重后的数据集导出
deduplicated_dataset.export_json('/path/to/your/output/deduplicated_file.json')
```

## 6. 可行性分析和排期规划

- 里程碑 1：图像过滤算子开发（11月27日 - 11月29日）
开发图像宽高比、分辨率、文件大小过滤算子。
开发图像去重功能（基于哈希算法）。
- 里程碑 2：文本过滤算子开发（11月30日 - 12月2日）
开发文本质量、重复字符、特殊字符过滤算子。
- 里程碑 3：图像-文本匹配过滤算子开发（12月3日 - 12月5日）
开发图像-文本相似度与匹配度过滤算子，集成CLIP和BLIP模型。
- 里程碑 4：集成与功能测试（12月6日 - 12月8日）
集成所有过滤算子并进行初步功能测试。
验证算子在数据集上的效果，确保正确性。
- 里程碑 5：性能优化与调优（12月9日 - 12月11日）
对各个算子进行性能优化，确保处理大规模数据集时的效率。
- 里程碑 6：系统测试与验收（12月12日 - 12月13日）
完成系统集成测试，确保各算子协同工作。
进行最终的性能和准确性验证。

## 7. 影响面

> 本方案将提升数据集的质量，减少低质量样本，确保训练数据更加精准。通过优化图像与文本的匹配度，能显著提高多模态任务的模型效果。



## 8.已实现算子

过滤算子
1. 图像去重：使用图像哈希算法对图像进行去重（phash/dhash/average_hash），可选择是否合并重复对话内容。
2. 图像宽高比过滤、最大宽高尺寸过滤、图像文件大小过滤
3. 基于clip-score过滤
4. 基于grounding_dino的过滤，可选只取前n%得分的图像、每张图像中最大最小检测数量、检测框的宽高比限制等。
5. 文本的SimHash和MinHash过滤

可视化算子
1. T-SNE可视化

分析算子
1. 基础数据分析功能（数量统计/字段分布/文件检查/内容分析/异常检测）
2. 基于token的数据分析功能（基于paddlenlp的tokenizer，统计token数量，分析token分布）
3. 添加多个维度的数据分析(基于paddlenlp输出颜色、形状、位置、大小、方向、关系、状态、类别)
4. 四种评估指标进行图文数据质量分析


### 过滤算子
#### 1. 图像去重：使用图像哈希算法对图像进行去重（phash/dhash/average_hash），可选择是否合并重复对话内容。
```python
import os
from typing import Optional, List, Dict
import imagehash
from PIL import Image
from ...core import T, MMDataset, register

@register()
def compute_hash(
    item: T,
    hash_method: str = "phash",
) -> Optional[str]:
    """计算图像的感知哈希值。"""
    image_path = item.get('image')
    if not image_path or not os.path.exists(image_path):
        return None

    try:
        with Image.open(image_path) as img:
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

def merge_conversations(conversations_list: List[List[Dict]]) -> List[Dict]:
    """合并多个对话列表，去除重复的对话。"""
    merged = []
    seen_pairs = set()  # 用于追踪已经见过的问答对
    
    for conversations in conversations_list:
        for i in range(0, len(conversations), 2):  # 每次处理一个问答对
            if i + 1 < len(conversations):
                # 创建问答对的唯一标识
                qa_pair = (
                    conversations[i]['value'].strip(),
                    conversations[i+1]['value'].strip()
                )
                
                if qa_pair not in seen_pairs:
                    merged.extend([conversations[i], conversations[i+1]])
                    seen_pairs.add(qa_pair)
    
    return merged

@register()
def remove_duplicates(
    dataset: MMDataset,
    hash_method: str = "phash",
    merge_text: bool = False
) -> MMDataset:
    """使用感知哈希算法移除重复图像，可选择是否合并对话内容。
    
    Args:
        dataset (MMDataset): 输入数据集
        hash_method (str): 使用的哈希算法类型，默认为 "phash"
        merge_text (bool): 是否合并重复图像的对话内容，默认为 False
        
    Returns:
        MMDataset: 处理后的数据集
    """
    # 用于存储已经出现的哈希值及其相关信息
    hash_dict: Dict[str, List] = {}
    filtered_items = []
    
    # 计算所有图像的哈希值
    hash_values = dataset.map(
        lambda x: compute_hash(x, hash_method),
        max_workers=8,
        progress=True
    )

    # 遍历数据集，处理重复图像
    for item, hash_value in zip(dataset, hash_values):
        if not hash_value:
            continue
            
        if hash_value not in hash_dict:
            # 新的哈希值，初始化
            hash_dict[hash_value] = {
                'item': item,
                'conversations_list': [item['conversations']]
            }
        elif merge_text:
            # 已存在的哈希值，且需要合并文本
            hash_dict[hash_value]['conversations_list'].append(item['conversations'])
    
    # 处理结果
    for hash_value, data in hash_dict.items():
        new_item = data['item'].copy()
        if merge_text and len(data['conversations_list']) > 1:
            # 合并对话内容
            new_item['conversations'] = merge_conversations(data['conversations_list'])
        filtered_items.append(new_item)
            
    # 返回新的数据集实例
    return MMDataset(filtered_items)
```
调用方式如下：

```python
from paddlemix.datacopilot.core import MMDataset

# 加载数据集
dataset = MMDataset.from_json('./my.json')
print(f"原始数据集大小: {len(dataset)}")

# 使用不同的哈希方法去重
deduped_phash = dataset.remove_duplicates(hash_method="phash", merge_text=True)
print(f"使用phash去重后的数据集大小: {len(deduped_phash)}")

# 链式调用示例
processed_dataset = (
    dataset
    .remove_duplicates(hash_method="phash", merge_text=False)  # 使用phash去重
    .nonempty()  # 移除空值
)
```

#### 2. 图像宽高比过滤、最大宽高尺寸过滤、图像文件大小过滤
```python
import os
from PIL import Image
from typing import Optional, List
from ...core import T, MMDataset, register

@register()
def filter_by_aspect_ratio(
    dataset: MMDataset,
    min_ratio: float = 0.333,
    max_ratio: float = 3.0
) -> MMDataset:
    """
    根据图像宽高比过滤样本。
    
    Args:
        dataset (MMDataset): 输入数据集。
        min_ratio (float): 最小宽高比，默认为 0.333。
        max_ratio (float): 最大宽高比，默认为 3.0。
        
    Returns:
        MMDataset: 过滤后的数据集。
    """
    filtered_items = []
    
    for item in dataset:
        image_path = item.get('image')
        if not image_path or not os.path.exists(image_path):
            continue
        
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                ratio = width / height
                if min_ratio <= ratio <= max_ratio:
                    filtered_items.append(item)
        except Exception as e:
            print(f"处理图像时出错 {image_path}: {e}")
    
    return MMDataset(filtered_items)


@register()
def filter_by_resolution(
    dataset: MMDataset,
    max_width: float = 727.88,
    max_height: float = 606.24
) -> MMDataset:
    """
    根据图像分辨率过滤样本。
    
    Args:
        dataset (MMDataset): 输入数据集。
        max_width (float): 最大宽度，默认为 727.88。
        max_height (float): 最大高度，默认为 606.24。
        
    Returns:
        MMDataset: 过滤后的数据集。
    """
    filtered_items = []
    
    for item in dataset:
        image_path = item.get('image')
        if not image_path or not os.path.exists(image_path):
            continue
        
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                if width <= max_width and height <= max_height:
                    filtered_items.append(item)
        except Exception as e:
            print(f"处理图像时出错 {image_path}: {e}")
    
    return MMDataset(filtered_items)


@register()
def filter_by_file_size(
    dataset: MMDataset,
    max_size_kb: float = 124
) -> MMDataset:
    """
    根据图像文件大小过滤样本。
    
    Args:
        dataset (MMDataset): 输入数据集。
        max_size_kb (float): 最大文件大小（以 KB 为单位），默认为 124。
        
    Returns:
        MMDataset: 过滤后的数据集。
    """
    filtered_items = []
    
    for item in dataset:
        image_path = item.get('image')
        if not image_path or not os.path.exists(image_path):
            continue
        
        try:
            file_size_kb = os.path.getsize(image_path) / 1024  # 转换为 KB
            if file_size_kb <= max_size_kb:
                filtered_items.append(item)
        except Exception as e:
            print(f"处理图像文件大小时出错 {image_path}: {e}")
    
    return MMDataset(filtered_items)

```
#### 3. 基于clip-score过滤
```python
@dataclass
class CLIPFilterConfig:
    """用于CLIP相似度过滤的配置。"""
    model_name: str = "paddlemix/CLIP/CLIP-ViT-L-14-laion2B-s32B-b82K"
    threshold: float = 0.25  # 相似度阈值


def clip_process_single_image(
    image_path: str,
    text_prompt: str,
    model: CLIP,
    processor: CLIPProcessor,
    config: CLIPFilterConfig
) -> Optional[float]:
    """使用CLIP模型处理单张图片并计算相似度。"""
    try:
        if not os.path.isfile(image_path):
            return None

        # 处理图片和文本
        processed_inputs = processor(
            images=[image_path],
            text=[text_prompt],
            max_length=77,
            return_tensors="pd",
            return_attention_mask=False,
            mode="eval",
            do_resize=True,
            do_crop=True,
            padding_zero=True,
        )

        # 提取图片和文本的输入
        image_tensor = processed_inputs["image"]
        input_ids = processed_inputs["input_ids"]

        # 计算相似度
        with paddle.no_grad():
            similarity = model.clip_score(
                image=image_tensor, 
                input_ids=input_ids
            )
        
        return float(similarity.item())  # 转换为Python浮点数
    except Exception as e:
        print(f"处理图片 {image_path} 时出错：{e}")
        return None


@register()
def filter_by_clip(
    dataset: MMDataset,
    text_prompt: str,
    config: Optional[CLIPFilterConfig] = None,
) -> MMDataset:
    """使用CLIP相似度分数过滤数据集。"""
    if config is None:
        config = CLIPFilterConfig()

    # 初始化模型和处理器
    model = CLIP.from_pretrained(config.model_name, ignore_mismatched_sizes=False)
    model.eval()
    image_processor = CLIPImageProcessor.from_pretrained(os.path.join(config.model_name, "processor", "eval"))
    text_processor = CLIPTextProcessor.from_pretrained(os.path.join(config.model_name, "processor", "eval"))
    tokenizer = SimpleTokenizer()
    processor = CLIPProcessor(image_processor, text_processor, tokenizer)

    filtered_items = []

    # 使用tqdm显示进度条
    for item in tqdm(dataset, desc="使用CLIP过滤图片"):
        image_path = item.get('image')
        if not image_path or not os.path.exists(image_path):
            continue

        # 计算相似度分数
        similarity = clip_process_single_image(
            image_path=image_path,
            text_prompt=text_prompt,
            model=model,
            processor=processor,
            config=config
        )

        # 如果相似度达到阈值，则保留该项
        if similarity is not None and similarity >= config.threshold:
            filtered_items.append(item)

    return MMDataset(filtered_items)

```

#### 4. 基于grounding_dino的过滤，可选只取前n%得分的图像、每张图像中最大最小检测数量、检测框的宽高比限制等。
```python
from tqdm import tqdm
from typing import Optional, Dict, Tuple
from PIL import Image
import os
import paddle
import paddle.nn.functional as F
from dataclasses import dataclass

from ....models.groundingdino.modeling import GroundingDinoModel
from ....processors.groundingdino_processing import GroundingDinoProcessor
from ...core import T, MMDataset, register

@dataclass
class GroundingDinoConfig:
    """Configuration for Grounding DINO filtering."""
    model_name: str = "GroundingDino/groundingdino-swint-ogc"
    box_threshold: float = 0.3
    min_objects: int = 1
    max_objects: int = 4
    min_aspect_ratio: float = 0.05  # 5%
    max_aspect_ratio: float = 0.95  # 95%
    top_percentage: float = 0.3  # Percentage for top detections


def dino_process_single_image(
    image_path: str,
    prompt: str,
    model: GroundingDinoModel,
    processor: GroundingDinoProcessor,
    config: GroundingDinoConfig
) -> Optional[Dict]:
    """Process a single image with Grounding DINO model."""
    try:
        # 加载图像
        if not os.path.isfile(image_path):
            return None
        image = Image.open(image_path).convert("RGB")
        
        # 处理图像
        image_tensor, mask, tokenized_out = processor(images=image, text=prompt)
        if image_tensor is None or image_tensor.shape[0] == 0:
            return None

        # 获取模型预测
        with paddle.no_grad():
            outputs = model(
                image_tensor,
                mask,
                input_ids=tokenized_out["input_ids"],
                attention_mask=tokenized_out["attention_mask"],
                text_self_attention_masks=tokenized_out["text_self_attention_masks"],
                position_ids=tokenized_out["position_ids"],
            )

        # 处理输出
        logits = F.sigmoid(outputs["pred_logits"])[0]  # [nq, 256]
        boxes = outputs["pred_boxes"][0]  # [nq, 4]

        # 计算置信度分数
        scores = logits.max(axis=1)
        
        # 应用 box_threshold 过滤低置信度框
        high_confidence_mask = scores >= config.box_threshold
        high_confidence_indices = paddle.nonzero(high_confidence_mask).flatten()
        
        if len(high_confidence_indices) == 0:
            print("No boxes passed the confidence threshold")
            return None
            
        scores = scores[high_confidence_indices]
        boxes = boxes[high_confidence_indices]

        # 取前配置的百分比
        if len(scores) > 0:
            num_to_keep = max(1, int(config.top_percentage * len(scores)))
            sorted_indices = paddle.argsort(scores, descending=True)[:num_to_keep]
            scores = scores[sorted_indices]
            boxes = boxes[sorted_indices]

        # 计算框的属性
        widths = boxes[:, 2] - boxes[:, 0]  # 修正宽度计算
        heights = boxes[:, 3] - boxes[:, 1]  # 修正高度计算
        aspect_ratios = widths / heights


        return {
            "num_objects": len(boxes),
            "scores": scores,
            "aspect_ratios": aspect_ratios
        }
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None


@register()
def filter_by_dino(
    dataset: MMDataset,
    prompt: str,
    config: Optional[GroundingDinoConfig] = None,
) -> MMDataset:
    """Filter dataset using Grounding DINO object detection."""
    if config is None:
        config = GroundingDinoConfig()

    # 初始化模型和处理器
    processor = GroundingDinoProcessor.from_pretrained(config.model_name)
    model = GroundingDinoModel.from_pretrained(config.model_name)
    model.eval()

    filtered_items = []

    # 使用 tqdm 显示进度条
    for item in tqdm(dataset, desc="Filtering images"):
        image_path = item.get('image')
        if not image_path or not os.path.exists(image_path):
            continue

        # 处理图像并获取检测结果
        detection_results = dino_process_single_image(
            image_path=image_path,
            prompt=prompt,
            model=model,
            processor=processor,
            config=config
        )
        
        if detection_results is None:
            continue

        # 应用过滤条件
        if (
            config.min_objects <= detection_results["num_objects"] <= config.max_objects and
            all(config.min_aspect_ratio <= ar <= config.max_aspect_ratio 
                for ar in detection_results["aspect_ratios"])
        ):
            filtered_items.append(item)

    return MMDataset(filtered_items)

```

#### 5. 文本的SimHash和MinHash过滤
```python
from typing import Optional, List, Dict
from datasketch import MinHash, MinHashLSH
from simhash import Simhash
from paddlemix.datacopilot.core import MMDataset, register

def compute_simhash(text: str) -> int:
    """计算文本的 SimHash 值，返回整数值"""
    return Simhash(text).value  # 直接返回整数值

def compute_minhash(text: str, num_perm: int = 128) -> MinHash:
    """计算文本的 MinHash 值。"""
    minhash = MinHash(num_perm=num_perm)
    for word in text.split():
        minhash.update(word.encode('utf8'))
    return minhash

def extract_conversation_texts(conversations: List[Dict]) -> List[str]:
    """
    从对话中提取文本对
    每个文本对由连续的human和assistant消息组成
    """
    texts = []
    for i in range(0, len(conversations)-1, 2):
        if (conversations[i]['from'] == 'human' and 
            conversations[i+1]['from'] == 'assistant'):
            text = conversations[i]['value'].strip() + ' ' + conversations[i+1]['value'].strip()
            texts.append(text)
    return texts

@register()
def remove_text_duplicates(
    dataset: MMDataset,
    method: str = "simhash",
    threshold: float = 0.8,
    merge_text: bool = False,
    num_perm: int = 128
) -> MMDataset:
    """基于 SimHash 或 MinHashLSH 去除文本级别的重复样本"""
    filtered_items = []
    hash_dict = {}
    
    if method == "simhash":
        for idx, item in enumerate(dataset):
            # 提取所有对话文本对
            texts = extract_conversation_texts(item['conversations'])
            
            for text in texts:
                if not text:
                    continue
                
                # 直接使用整数值作为哈希键
                simhash_value = compute_simhash(text)
                found_similar = False
                
                for existing_hash, existing_data in hash_dict.items():
                    # 计算汉明距离
                    distance = bin(simhash_value ^ existing_hash).count('1')
                    if distance <= int((1 - threshold) * 64):  # 64 是 SimHash 的长度
                        found_similar = True
                        if merge_text:
                            existing_data['items'].append(item)
                        break
                
                if not found_similar:
                    hash_dict[simhash_value] = {
                        'items': [item],
                        'texts': [text]
                    }
    
    elif method == "minhash":
        lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        
        for idx, item in enumerate(dataset):
            texts = extract_conversation_texts(item['conversations'])
            
            for text in texts:
                if not text:
                    continue
                
                minhash = compute_minhash(text, num_perm)
                similar_items = list(lsh.query(minhash))
                
                if similar_items:
                    if merge_text:
                        for sim_idx in similar_items:
                            hash_dict[sim_idx]['items'].append(item)
                else:
                    lsh.insert(idx, minhash)
                    hash_dict[idx] = {
                        'items': [item],
                        'texts': [text]
                    }
    
    # 去重并保留第一个相似项
    unique_items = {}
    for data in hash_dict.values():
        representative_item = data['items'][0]
        unique_items[representative_item['id']] = representative_item
    
    # 将唯一项转换为列表
    filtered_items = list(unique_items.values())
    
    return MMDataset(filtered_items)
```



### 可视化算子
#### 1. T-SNE可视化
```python
import os
import json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
from ...core import T, MMDataset, register

@register()
def extract_text_for_lda(item: T) -> Optional[str]:
    """从对话中提取文本，用于主题建模."""
    conversations = item.get("conversations", [])
    text = []
    for convo in conversations:
        text.append(convo["value"])
    return " ".join(text)

@register()
def lda_topic_clustering(
    dataset: MMDataset,
    num_topics: int = 5,
    tsne_perplexity: int = 30,
    tsne_learning_rate: int = 200,
    tsne_n_iter: int = 1000,
    random_state: int = 42,
    output_plot: str = "lda_tsne_plot.png"
):
    """对对话文本进行LDA主题聚类，并使用T-SNE进行可视化."""
    # 提取文本数据
    texts = dataset.map(extract_text_for_lda)
    texts = [text for text in texts if text.strip()]  # 移除空文本

    # 文本向量化
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    text_matrix = vectorizer.fit_transform(texts)

    # LDA 主题建模
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=random_state)
    lda_result = lda.fit_transform(text_matrix)

    # 使用 T-SNE 降维
    tsne = TSNE(
        n_components=2,
        perplexity=tsne_perplexity,
        learning_rate=tsne_learning_rate,
        n_iter=tsne_n_iter,
        random_state=random_state
    )
    tsne_result = tsne.fit_transform(lda_result)

    # 可视化结果
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        tsne_result[:, 0], tsne_result[:, 1], c=np.argmax(lda_result, axis=1), cmap='tab10', alpha=0.7
    )
    plt.colorbar(scatter, label="Topic Cluster")
    plt.title("LDA Topic Clustering with T-SNE Visualization")
    plt.xlabel("T-SNE Dimension 1")
    plt.ylabel("T-SNE Dimension 2")
    plt.savefig(output_plot)
    plt.show()

    return {
        "lda_result": lda_result,
        "tsne_result": tsne_result,
        "topics": np.argmax(lda_result, axis=1).tolist()
    }

```



### 数据分析算子
#### 1. 基础数据分析功能（数量统计/字段分布/文件检查/内容分析/异常检测）
```python
from collections import Counter
import jieba
from collections import Counter
from langdetect import detect, DetectorFactory
from typing import Dict
from paddlemix.datacopilot.core import MMDataset, register
import os
DetectorFactory.seed = 0  # 保证语言检测结果可复现

def count_data_statistics(dataset: MMDataset) -> Dict:
    """统计数据集的基本数量信息"""
    total_records = len(dataset)
    unique_images = len(set(item['image'] for item in dataset))
    conversation_counts = [len(item['conversations']) for item in dataset]

    return {
        "total_records": total_records,
        "unique_images": unique_images,
        "conversation_counts": conversation_counts
    }


def analyze_field_distribution(dataset: MMDataset) -> Dict:
    """分析字段分布信息"""
    human_msgs = []
    assistant_msgs = []
    languages = Counter()

    for item in dataset:
        for conv in item["conversations"]:
            if conv["from"] == "human":
                human_msgs.append(conv["value"])
            elif conv["from"] == "assistant":
                assistant_msgs.append(conv["value"])

            try:
                lang = detect(conv["value"])
                languages[lang] += 1
            except:
                languages["unknown"] += 1

    return {
        "human_message_count": len(human_msgs),
        "assistant_message_count": len(assistant_msgs),
        "languages_distribution": dict(languages)
    }



def validate_image_paths(dataset: MMDataset) -> Dict:
    """验证图片路径的分布和文件存在性"""
    all_paths = [item['image'] for item in dataset]
    missing_paths = [path for path in all_paths if not os.path.exists(path)]

    path_distribution = Counter(os.path.dirname(path) for path in all_paths)

    return {
        "total_images": len(all_paths),
        "missing_images": len(missing_paths),
        "missing_paths": missing_paths,
        "path_distribution": dict(path_distribution)
    }



def analyze_content(dataset: MMDataset) -> Dict:
    """分析对话内容"""
    human_msgs = []
    assistant_msgs = []

    for item in dataset:
        for conv in item["conversations"]:
            if conv["from"] == "human":
                human_msgs.append(conv["value"])
            elif conv["from"] == "assistant":
                assistant_msgs.append(conv["value"])

    # 分词统计
    human_word_count = Counter(jieba.lcut(" ".join(human_msgs)))
    assistant_word_count = Counter(jieba.lcut(" ".join(assistant_msgs)))

    return {
        "human_word_count": human_word_count.most_common(10),
        "assistant_word_count": assistant_word_count.most_common(10)
    }



def detect_anomalies(dataset: MMDataset) -> Dict:
    """检测数据集中的异常项"""
    missing_fields = []
    empty_conversations = []

    for item in dataset:
        if not all(key in item for key in ["id", "image", "conversations"]):
            missing_fields.append(item)

        if not item["conversations"] or any(not conv["value"].strip() for conv in item["conversations"]):
            empty_conversations.append(item)

    return {
        "missing_field_count": len(missing_fields),
        "empty_conversation_count": len(empty_conversations),
        "examples_missing_fields": missing_fields[:5],  # 提供部分异常样本
        "examples_empty_conversations": empty_conversations[:5]
    }


@register()
def run_all_analysis(dataset: MMDataset) -> Dict:
    """统一调用所有分析功能"""
    results = {}

    # 1. 数据数量统计
    stats = count_data_statistics(dataset)
    results["data_statistics"] = stats

    # 2. 字段分布分析
    field_dist = analyze_field_distribution(dataset)
    results["field_distribution"] = field_dist

    # 3. 文件路径和图片检查
    path_validation = validate_image_paths(dataset)
    results["path_validation"] = path_validation

    # 4. 内容分析
    content_analysis = analyze_content(dataset)
    results["content_analysis"] = content_analysis

    # 5. 异常检测
    anomalies = detect_anomalies(dataset)
    results["anomaly_detection"] = anomalies

    return results

```

#### 2. 基于token的数据分析功能（基于paddlenlp的tokenizer，统计token数量，分析token分布）
```python
from collections import Counter
import json
import jieba
import matplotlib.pyplot as plt
from paddlenlp.transformers import AutoTokenizer
import paddle
from matplotlib import rcParams
from matplotlib import font_manager
from paddlemix.datacopilot.core import MMDataset, register
from typing import Dict

# 设置字体路径
font_path = '/home/lizhijun/PaddleMIX-develop/PaddleNLP/font/SimHei.ttf'  

# 手动添加字体到 matplotlib 字体管理器
font_manager.fontManager.addfont(font_path)

# 设置 matplotlib 使用 SimHei 字体
plt.rcParams['font.family'] = 'SimHei'  # 使用 SimHei 字体
rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 初始化分词器
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

@register()
def load_data(file_path: str) -> MMDataset:
    """加载 JSON 数据并转换为 MMDataset 格式"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return MMDataset.from_list(data)

def analyze_tokens(dataset: MMDataset) -> Dict:
    """分析 Token 统计信息"""
    human_tokens = []
    assistant_tokens = []

    for item in dataset:
        for conv in item["conversations"]:
            tokens = tokenizer(conv["value"], truncation=True, return_tensors="pd")["input_ids"].numpy().flatten()
            if conv["from"] == "human":
                human_tokens.extend(tokens)
            elif conv["from"] == "assistant":
                assistant_tokens.extend(tokens)

    # 计算频率分布
    human_token_counts = Counter(human_tokens)
    assistant_token_counts = Counter(assistant_tokens)
    
    # 计算总 token 数
    human_total_tokens = len(human_tokens)
    assistant_total_tokens = len(assistant_tokens)

    return {
        "human": {
            "total_tokens": human_total_tokens,
            "token_distribution": human_token_counts,
        },
        "assistant": {
            "total_tokens": assistant_total_tokens,
            "token_distribution": assistant_token_counts,
        },
        "overall": {
            "total_tokens": human_total_tokens + assistant_total_tokens,
            "human_ratio": human_total_tokens / (human_total_tokens + assistant_total_tokens),
            "assistant_ratio": assistant_total_tokens / (human_total_tokens + assistant_total_tokens),
        }
    }

def decode_token_ids(token_counts: Counter) -> Counter:
    """解码 Token ID 为原始文字"""
    decoded_counts = Counter()
    for token_id, count in token_counts.items():
        decoded_text = tokenizer.decode([token_id]).strip()
        decoded_counts[decoded_text] += count
    return decoded_counts

def plot_token_distribution(token_counts: Counter, title: str, output_path: str) -> None:
    """绘制 Token 分布图"""
    most_common = token_counts.most_common(20)
    tokens, frequencies = zip(*most_common)

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(tokens)), frequencies, tick_label=tokens)
    plt.xticks(rotation=45, fontsize=10)
    plt.xlabel("Decoded Tokens")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_sentence_length_distribution(dataset: MMDataset, output_path: str) -> None:
    """绘制句子长度分布图"""
    lengths = []
    for item in dataset:
        for conv in item["conversations"]:
            tokens = tokenizer(conv["value"], truncation=True, return_tensors="pd")["input_ids"].numpy().flatten()
            lengths.append(len(tokens))
    
    plt.figure(figsize=(10, 5))
    plt.hist(lengths, bins=20, color='blue', alpha=0.7)
    plt.xlabel("Sentence Length (Tokens)")
    plt.ylabel("Frequency")
    plt.title("Sentence Length Distribution")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def analyze_field_distribution(dataset: MMDataset) -> Dict:
    """分析字段分布信息"""
    human_msgs = []
    assistant_msgs = []
    for item in dataset:
        for conv in item["conversations"]:
            if conv["from"] == "human":
                human_msgs.append(conv["value"])
            elif conv["from"] == "assistant":
                assistant_msgs.append(conv["value"])

    human_word_count = Counter(jieba.lcut(" ".join(human_msgs)))
    assistant_word_count = Counter(jieba.lcut(" ".join(assistant_msgs)))

    return {
        "human_word_count": human_word_count.most_common(10),
        "assistant_word_count": assistant_word_count.most_common(10)
    }

@register()
def run_token_analysis(dataset: MMDataset) -> Dict:
    """统一调用所有分析功能"""
    results = {}

    # 1. Token 统计分析
    token_results = analyze_tokens(dataset)
    results["token_analysis"] = token_results

    # 2. 字段分布分析
    field_dist = analyze_field_distribution(dataset)
    results["field_distribution"] = field_dist

    # 3. 绘制分布图
    human_decoded_counts = decode_token_ids(token_results["human"]["token_distribution"])
    assistant_decoded_counts = decode_token_ids(token_results["assistant"]["token_distribution"])

    plot_token_distribution(human_decoded_counts, "Human Token Distribution", "human_token_distribution.png")
    plot_token_distribution(assistant_decoded_counts, "Assistant Token Distribution", "assistant_token_distribution.png")
    plot_sentence_length_distribution(dataset, "sentence_length_distribution.png")

    return results

```

#### 3. 添加多个维度的数据分析(基于paddlenlp输出颜色、形状、位置、大小、方向、关系、状态、类别)
```python
from paddlenlp.transformers import AutoTokenizer, AutoModelForCausalLM
from paddlemix.datacopilot.core import MMDataset, register
import os
from collections import Counter
from tqdm import tqdm
from typing import Dict


# 加载模型的函数，支持传入模型名称
def load_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype="float16")
    return tokenizer, model

# 定义用于提取图像特征的 prompt
prompt = """You will be given a description of an image. Extract the following details if mentioned:

Color: List colors mentioned.
Shape: List shapes mentioned.
Position: Describe the object's position (relative to background/other objects).
Size: Describe the object's size (e.g., large, small).
Direction: Describe the object's orientation (e.g., tilt, front/back).
Relationship: Describe relationships between objects (e.g., on, near, etc.).
Action/State: Describe any actions or states (e.g., moving, still).
Category: List object types (e.g., cars, flowers).
Return the information in the following format:

Color: [list colors]
Shape: [list shapes]
Position: [position]
Size: [size description]
Direction: [direction]
Relationship: [relationship]
Action/State: [action/state]
Category: [category]

Text: "[text_input]"
"""



def clean_and_count(all_info):
    """清理并统计每个类别的出现频率"""
    cleaned_info = {}

    for category, items in all_info.items():
        # 清理无效项，例如 'list colors' 和 'None'
        valid_items = [item.strip() for item in items if item not in ['list colors', 'list shapes', 'position', 'None', 'size description', 'direction', 'action/state', 'relationship', 'category']]
        
        # 统计每个类别项的频率
        item_counts = Counter(valid_items)
        
        # 保存清理后的频率统计
        cleaned_info[category] = item_counts

    return cleaned_info


@register()
def analyze_gpt_responses(dataset: MMDataset, model_name: str = "Qwen/Qwen2.5-0.5B") -> Dict:
    """分析数据集中的所有 'gpt' 对话内容，并提取每个类别的信息"""
    results = {}
    all_info = {
        "Color": [],
        "Shape": [],
        "Position": [],
        "Size": [],
        "Direction": [],
        "Relationship": [],
        "Action/State": [],
        "Category": []
    }

    # 加载指定的模型
    tokenizer, model = load_model(model_name)

    for item in tqdm(dataset):
        gpt_responses = []

        # 获取所有 'gpt' 的对话内容
        for conversation in item["conversations"]:
            if conversation["from"] == "gpt":
                gpt_responses.append(conversation["value"])

        # 将所有 'gpt' 对话拼接为一个文本块
        gpt_text = "\n".join(gpt_responses)

        # 替换 prompt 中的占位符
        splice_prompt = prompt.replace("text_input", gpt_text)

        # 使用 tokenizer 对输入文本进行编码
        input_features = tokenizer(splice_prompt, return_tensors="pd")

        # 生成模型的输出
        outputs = model.generate(**input_features, max_length=128)

        # 解码并获取分析结果
        analysis_result = tokenizer.batch_decode(outputs[0], skip_special_tokens=True)[0]

        # 提取具体信息并分类存储
        for category in all_info.keys():
            # 查找并提取该类别的信息
            start_idx = analysis_result.find(f"{category}: [")
            if start_idx != -1:
                start_idx += len(f"{category}: [")
                end_idx = analysis_result.find("]", start_idx)
                if end_idx != -1:
                    info = analysis_result[start_idx:end_idx]
                    all_info[category].extend(info.split(","))
        
        # 存储结果
        results[item['id']] = analysis_result

    # 使用clean_and_count函数清理all_info并统计频率
    cleaned_info = clean_and_count(all_info)

    # 输出每个类别及其项的频率
    for category, counts in cleaned_info.items():
        print(f"{category}:")
        for item, count in counts.items():
            print(f"  {item}: {count}")
        print("-" * 50)
    # 返回合并后的结果，包括每个类别的实际信息
    return cleaned_info

```


#### 4. 四种评估指标进行图文数据质量分析
```python
from paddlenlp.transformers import AutoTokenizer, AutoModelForCausalLM
from paddlemix.datacopilot.core import MMDataset, register
import os
from collections import Counter
from tqdm import tqdm
from typing import Dict, List

# 预置的四种评估指标及其提示词
CRITERIA_PROMPTS = {
    "image_text_matching": """Please evaluate if the provided text caption accurately represents the main features and objects of the image. The caption doesn't need to detail every aspect of the image, but it should capture its primary theme. Rate the overall quality of the text caption's match to the image on a scale of 1-100, considering the criteria mentioned.""",
    "object_detail_fulfillment": """Please evaluate the text caption to determine if it provides detailed descriptions of objects that align with the image. Specifically, assess if the caption sufficiently describes the color, size, position, shape, material, etc., of the objects. Afterward, rate the caption's overall accuracy in capturing object details from the image on a scale of 1-100, based on the criteria provided.""",
    "caption_text_quality": """Please evaluate the text caption based on the following criteria: Grammatical Correctness, Diversity of Vocabulary (e.g., the range and uniqueness of words used), Fluency (e.g., smoothness and natural flow of sentences), Readability, Length, and Structure. Assign an overall quality score on a scale of 1-100.""",
    "semantic_understanding": """Evaluate the given text caption in relation to its corresponding image. Your goal is to determine if the text caption provides additional semantic information that isn't readily apparent just from the image itself. Rate the text caption's semantic depth on a scale from 1 to 100.""",
}

DEFAULT_PROMPT_TEMPLATE = """Text Caption: {caption}

{criteria}
A higher score indicates a higher level of {aspect}. Ensure that your scoring is nuanced and uses the entire range from 0 to 100, reflecting the subtle differences. The score should be given as an integer, with each number between 0 and 100 considered as a potential score, avoiding the tendency to round to multiples of 10. Please first output a single line containing the value indicating the scores. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias."""

# 加载模型的函数，支持传入模型名称
def load_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype="float16")
    return tokenizer, model

def evaluate_image_caption(
    dataset: MMDataset, 
    model_name: str = "Qwen/Qwen2.5-0.5B", 
    selected_metrics: List[str] = None
) -> Dict:
    """
    根据指定的指标评估图文质量。
    :param dataset: MMDataset 数据集
    :param model_name: 模型名称
    :param selected_metrics: 指定要使用的指标，默认为 all
    :return: 每个数据项的评估结果
    """
    # 如果未指定指标，默认使用所有指标
    if selected_metrics is None:
        selected_metrics = list(CRITERIA_PROMPTS.keys())
    
    # 加载模型
    tokenizer, model = load_model(model_name)
    
    # 存储最终结果
    results = {}

    for item in tqdm(dataset):
        item_id = item["id"]
        conversations = item["conversations"]
        
        # 遍历每个问答对
        for idx, conversation in enumerate(conversations):
            question = conversation["from"]  # 获取提问者
            answer = conversation["value"]  # 获取回答
            
            if question != "gpt":
                continue
            
            # 对每个选定的指标生成评估
            for metric in selected_metrics:
                criteria = CRITERIA_PROMPTS[metric]
                aspect = metric.replace("_", " ")
                caption = answer
                
                # 生成完整的 prompt
                full_prompt = DEFAULT_PROMPT_TEMPLATE.format(
                    caption=caption, 
                    criteria=criteria, 
                    aspect=aspect
                )
                
                # 使用 tokenizer 编码输入
                input_features = tokenizer(full_prompt, return_tensors="pd")
                
                # 模型生成输出
                outputs = model.generate(**input_features, max_length=256)
                
                # 解码生成结果
                decoded_output = tokenizer.batch_decode(outputs[0], skip_special_tokens=True)[0]

                
                # 存储结果
                if item_id not in results:
                    results[item_id] = {}
                if idx not in results[item_id]:
                    results[item_id][idx] = {}
                results[item_id][idx][metric] = decoded_output

    return results

@register()
def analyze_image_caption_with_metrics(dataset: MMDataset, model_name: str, selected_metrics: List[str] = None):
    """
    分析多轮对话的图文描述质量。
    """
    results = evaluate_image_caption(dataset, model_name, selected_metrics)
    
    # 打印或存储最终结果
    for item_id, conversations in results.items():
        print(f"Item ID: {item_id}")
        for idx, metrics in conversations.items():
            print(f"  Round {idx}:")
            for metric, output in metrics.items():
                print(f"    {metric}: {output}")
    return results

```



## 9.TODO
1. blip2模型随机打分问题（BUG未修复）
2. 数据生成prompt模板（论文查阅中）
3. 验证上述所有算子的有效性，部分算子未调整为最优状态，待验证
4. 代码规范化（注释、格式、变量命名）
5. 算子并行计算
6. 基于文本的质量分析prompt模板