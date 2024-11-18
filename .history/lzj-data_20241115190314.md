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
