from paddlemix.datacopilot.core import MMDataset
from paddlemix.datacopilot.ops.filter._judge_analysis import gpt_responses_judge
from paddlemix.datacopilot.ops.analysis._description_analysis import description_analysis
from paddlemix.datacopilot.ops.analysis._quality_analysis import analyze_image_caption_with_metrics


anno_path = 'datasets/llava/02_val_chatml_filter.json'
# 加载数据集
dataset = MMDataset.from_json(anno_path)

# 设置模型名称和词云保存目录
model_name = "Qwen/Qwen2.5-7B"  # 可根据需要修改


# 调用GPT回复分析
# dataset = dataset.gpt_responses_judge(model_name=model_name)

# 描述分析
# dataset = dataset.description_analysis(model_name=model_name)


# 4种属性分析
dataset = dataset.analyze_image_caption_with_metrics(model_name=model_name)



print("过滤后数据集数量为:", len(dataset))
print("Dataset validation complete.")
dataset.export_json(anno_path.replace('.json', '_filter-judge-1221.json'))