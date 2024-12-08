from paddlemix.datacopilot.core import MMDataset

# 加载数据集
dataset = MMDataset.from_json('./my.json')

# 设置模型名称和词云保存目录
model_name = "Qwen/Qwen2.5-0.5B"  # 可根据需要修改


# 调用GPT回复分析
analysis_results = dataset.gpt_responses_judge(model_name=model_name)
