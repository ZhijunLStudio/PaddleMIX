from paddlemix.datacopilot.core import MMDataset

# 加载数据集
dataset = MMDataset.from_json('../my.json')
print(f"原始数据集大小: {len(dataset)}")

# 1. 使用宽高比过滤
filtered_by_aspect_ratio = dataset.filter_by_aspect_ratio(min_ratio=0.5, max_ratio=2.0)
print(f"使用宽高比过滤后的数据集大小: {len(filtered_by_aspect_ratio)}")

# 2. 使用分辨率过滤
filtered_by_resolution = dataset.filter_by_resolution(max_width=800, max_height=600)
print(f"使用分辨率过滤后的数据集大小: {len(filtered_by_resolution)}")

# 3. 使用文件大小过滤
filtered_by_file_size = dataset.filter_by_file_size(max_size_kb=200)
print(f"使用文件大小过滤后的数据集大小: {len(filtered_by_file_size)}")

# 链式调用示例
processed_dataset = (
    dataset
    .filter_by_aspect_ratio(min_ratio=0.5, max_ratio=2.0)  # 使用宽高比过滤
    .filter_by_resolution(max_width=800, max_height=600)  # 使用分辨率过滤
    .filter_by_file_size(max_size_kb=200)  # 使用文件大小过滤
    .nonempty()  # 移除空值
)

# 导出处理后的数据集
processed_dataset.export_json('output_filtered.json')
