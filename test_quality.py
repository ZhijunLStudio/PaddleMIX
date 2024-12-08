from paddlemix.datacopilot.core import MMDataset

# 加载数据集
dataset = MMDataset.from_json('./my.json')

# 设置模型名称
model_name = "Qwen/Qwen2.5-0.5B"  # 选择要使用的模型名称

# 指定评估指标，可以选择多个指标或使用默认的所有指标
selected_metrics = ["image_text_matching", "object_detail_fulfillment", "caption_text_quality", "semantic_understanding"]  # 可修改或设置为 None

# 调用 analyze_image_caption_with_metrics 函数进行评估
analysis_results = dataset.analyze_image_caption_with_metrics(
    model_name=model_name,
    selected_metrics=selected_metrics  # 如果不传则默认为所有指标
)