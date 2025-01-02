import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
from paddlemix.datacopilot.core import MMDataset
from paddlemix.datacopilot.ops.analysis._base_analysis import run_base_analysis
from paddlemix.datacopilot.ops.analysis._description_analysis import description_analysis

# 加载数据集
# dataset = MMDataset.from_json('llava_v1_5_mix665k.json')
dataset = MMDataset.from_json('test_qwen2_vl_1w.json')
# 设置模型名称和词云保存目录
model_name = "Qwen/Qwen2.5-7B"  # 可根据需要修改


# model_name = "meta-llama/Llama-3.2-1B"

dataset = dataset.description_analysis(model_name)

# 调用GPT回复分析
# analysis_results = dataset.analyze_gpt_responses(model_name=model_name)

# 分析标志，控制哪些分析开启，哪些关闭
# analysis_flags = {
#     "data_statistics": True,
#     "field_distribution": True, 
#     "path_validation": True,
#     "anomaly_detection": True,  
#     "token_analysis": True
# }

# 调用分析函数
# results = dataset.run_base_analysis(analysis_flags=analysis_flags)
# print(results)