from paddlemix.datacopilot.core import MMDataset

# 加载数据集
# dataset = MMDataset.from_json('llava_v1_5_mix665k.json')
dataset = MMDataset.from_json('random_samples_newconv.json')
# 设置模型名称和词云保存目录
model_name = "Qwen/Qwen2.5-0.5B"  # 可根据需要修改


# model_name = "meta-llama/Llama-3.2-1B"


# 调用GPT回复分析
# analysis_results = dataset.analyze_gpt_responses(model_name=model_name)

# 分析标志，控制哪些分析开启，哪些关闭
analysis_flags = {
    "data_statistics": True,
    "field_distribution": True,  # 关闭字段分布分析
    "path_validation": True,
    "anomaly_detection": True,   # 关闭异常项检测
    "token_analysis": True
}

# 调用分析函数
results = dataset.run_base_analysis(analysis_flags=analysis_flags)
# print(results)