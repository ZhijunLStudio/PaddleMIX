from paddlemix.datacopilot.core import MMDataset

# 加载数据集
# dataset = MMDataset.from_json('llava_v1_5_mix665k.json')
dataset = MMDataset.from_json('datasets/llava/03_val_chatml_percentiles_filter.json')
# 设置模型名称和词云保存目录
model_name = "Qwen/Qwen2.5-0.5B"  # 可根据需要修改


# model_name = "meta-llama/Llama-3.2-1B"


# 调用GPT回复分析
# analysis_results = dataset.analyze_gpt_responses(model_name=model_name)

# 分析标志，控制哪些分析开启，哪些关闭
analysis_flags = {
    "data_statistics": True,
    "field_distribution": False, 
    "path_validation": False,
    "anomaly_detection": False,  
    "token_analysis": False
}

# 调用分析函数
results = dataset.run_base_analysis(analysis_flags=analysis_flags)
# print(results)