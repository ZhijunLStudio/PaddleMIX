import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
from paddlemix.datacopilot.core import MMDataset
from paddlemix.datacopilot.ops.generate._qa_pairs_generate import generate_qna_for_images


# 加载数据集
# anno_path = 'datasets/llava/12_train_chatml_filter_clip.json'
# dataset = MMDataset.from_json(anno_path)


# 调用分析函数
dataset = generate_qna_for_images(image_folder_path="datasets/random_1w")

dataset.export_json('test_qwen2_vl_1w.json')