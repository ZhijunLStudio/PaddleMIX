from paddlemix.datacopilot.core import MMDataset

# 加载数据集
dataset = MMDataset.from_json('./my.json')

# 调用所有分析功能
analysis_results = dataset.run_all_analysis()

# 输出分析结果
for analysis_type, result in analysis_results.items():
    print(f"{analysis_type}: {result}")
