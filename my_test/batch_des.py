import os
from paddlemix.datacopilot.core import MMDataset
from paddlemix.datacopilot.ops.analysis._base_analysis import run_base_analysis

# 定义输入目录和输出目录
input_dir = "datasets/llava"  # 替换为你的 JSON 文件所在目录
output_base_dir = "output_directory"  # 替换为你的结果保存目录

# 定义需要排除的 JSON 文件列表
excluded_files = {"00_llava_v1_5_mix665k.json", "01_train_chatml.json", "01_val_chatml.json", "chat_template.json"}  # 替换为你需要排除的文件名

# 分析标志，控制哪些分析开启，哪些关闭
analysis_flags = {
    "data_statistics": True,
    "field_distribution": False, 
    "path_validation": False,
    "anomaly_detection": False,  
    "token_analysis": False
}

# 遍历 JSON 文件并分析
for file_name in os.listdir(input_dir):
    if file_name.endswith(".json") and file_name not in excluded_files:
        input_file_path = os.path.join(input_dir, file_name)
        
        # 创建以 JSON 文件名命名的输出文件夹
        output_dir = os.path.join(output_base_dir, os.path.splitext(file_name)[0])
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            print(f"Analyzing {file_name}...")
            
            # 加载数据集
            dataset = MMDataset.from_json(input_file_path)
            
            # 调用分析函数并保存结果
            dataset.run_base_analysis(analysis_flags=analysis_flags, output_dir=output_dir)
            
            print(f"Analysis complete for {file_name}, results saved in {output_dir}")
        except Exception as e:
            print(f"Error analyzing {file_name}: {e}")
