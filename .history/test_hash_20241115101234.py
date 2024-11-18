from paddlemix.datacopilot.core import MMDataset

# 加载数据集
dataset = MMDataset.from_json('./my.json')
print(f"原始数据集大小: {len(dataset)}")

# 使用不同的哈希方法去重
# deduped_phash = dataset.remove_duplicates(hash_method="phash")
# print(f"使用phash去重后的数据集大小: {len(deduped_phash)}")

# deduped_dhash = dataset.remove_duplicates(hash_method="dhash")
# print(f"使用dhash去重后的数据集大小: {len(deduped_dhash)}")

# 链式调用示例
processed_dataset = (
    dataset
    .remove_duplicates(hash_method="phash")  # 使用phash去重
    .nonempty()  # 移除空值
)

# 导出处理后的数据集
processed_dataset.export_json('output.json')