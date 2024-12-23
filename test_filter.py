import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

from paddlemix.datacopilot.core import MMDataset
from paddlemix.datacopilot.ops.convert._llava_convert import llava_convert
from paddlemix.datacopilot.ops.filter._base_filter import valid_data_filter
from paddlemix.datacopilot.ops.filter._image_clip_filter import CLIPFilterConfig
# from paddlemix.datacopilot.ops.filter._conversation_percentage_filter import conversation_percentage_filter
# from paddlemix.datacopilot.ops.filter._conversation_hash_filter import remove_text_duplicates
# from paddlemix.datacopilot.ops.filter._image_filesize_filter import image_filesize_filter
# from paddlemix.datacopilot.ops.filter._image_hash_filter import image_hash_filter
# from paddlemix.datacopilot.ops.filter._image_ration_filter import image_ration_filter
# from paddlemix.datacopilot.ops.filter._image_resolution_filter import image_resolution_filter
# from paddlemix.datacopilot.ops.filter._conversation_length_filter import conversation_length_filter
# from paddlemix.datacopilot.ops.filter._alphanumeric_ratio_filter import alphanumeric_ratio_filter
# from paddlemix.datacopilot.ops.filter._average_line_length_filter import average_line_length_filter
# from paddlemix.datacopilot.ops.filter._char_ngram_repetition_filter import char_ngram_repetition_filter
# from paddlemix.datacopilot.ops.filter._language_id_filter import language_id_filter
# from paddlemix.datacopilot.ops.filter._maximum_line_length_filter import maximum_line_length_filter
# from paddlemix.datacopilot.ops.filter._perplexity_filter import perplexity_filter
# from paddlemix.datacopilot.ops.filter._special_characters_filter import special_characters_filter
# from paddlemix.datacopilot.ops.filter._stopwords_ratio_filter import stopwords_ratio_filter
# from paddlemix.datacopilot.ops.filter._text_action_filter import text_action_filter
# from paddlemix.datacopilot.ops.filter._text_entity_dependency_filter import text_entity_dependency_filter
# from paddlemix.datacopilot.ops.filter._token_num_filter import token_num_filter
# from paddlemix.datacopilot.ops.filter._word_ngram_repetition_filter import word_ngram_repetition_filter
# from paddlemix.datacopilot.ops.filter._word_num_filter import word_num_filter

# 数据集路径
anno_path = 'datasets/llava/12_train_chatml_filter_clip.json'

# 加载数据集
print("Loading dataset...")
dataset = MMDataset.from_json(anno_path)
print("初始数据集数量为:", len(dataset))


# 转换算子
# dataset = dataset.llava_convert()

# 0.过滤无效图像和文本
# dataset = dataset.valid_data_filter()

# 1.配置CLIP过滤器
# clip_config = CLIPFilterConfig(
#     model_name="paddlemix/CLIP/CLIP-ViT-L-14-laion2B-s32B-b82K",
#     threshold=0.15,  # 设置相似度阈值
#     batch_size=2560,  # 批量大小
#     save_images=False  # 控制是否保存低置信度图像
# )

# # 使用过滤器处理数据集并保存图片
# dataset = dataset.image_clip_filter(config=clip_config)

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

# 9.过滤掉非字母数字字符的文本
# dataset = dataset.alphanumeric_ratio_filter()

# 10.过滤掉平均行长度
# dataset = dataset.average_line_length_filter()

# 11.n-gram过滤
# dataset = dataset.char_ngram_repetition_filter()

# 12.语言id过滤
# dataset = dataset.language_id_filter()

# 13.过滤掉最大行长度过小的会话
# dataset = dataset.maximum_line_length_filter()

# 14.文本困惑度计算
# dataset = dataset.perplexity_filter()

# 15.特殊字符过滤
# dataset = dataset.special_characters_filter()

# 16.停用词过滤
# dataset = dataset.stopwords_ratio_filter()

# 17.动词检测
# dataset = dataset.text_action_filter()

# 18.文本实体依赖性过滤
# dataset = dataset.text_entity_dependency_filter()

# 19.token数量过滤
# dataset = dataset.token_num_filter()


# 20.基于词的n-gram过滤
# dataset = dataset.word_ngram_repetition_filter()

# 21.基于单词数量过滤
# dataset = dataset.word_num_filter()



dataset = dataset.nonempty()

print("过滤后数据集数量为:", len(dataset))
print("Dataset validation complete.")
dataset.export_json(anno_path.replace('.json', '_filter-noempty.json'))
# dataset.export_json("test.json")