from paddlemix.datacopilot.core import MMDataset

# 加载数据集
dataset = MMDataset.from_json('./my.json')
print(f"原始数据集大小: {len(dataset)}")

# 使用 SimHash 去重
deduped_simhash = dataset.remove_text_duplicates(method="simhash", threshold=0.1, merge_text=True)
print(f"使用 SimHash 去重后的数据集大小: {len(deduped_simhash)}")

# # 使用 MinHash 去重
deduped_minhash = dataset.remove_text_duplicates(method="minhash", threshold=0.3, merge_text=False, num_perm=128)
print(f"使用 MinHash 去重后的数据集大小: {len(deduped_minhash)}")

# 链式调用示例
processed_dataset = (
    dataset
    .remove_text_duplicates(method="simhash", threshold=0.1, merge_text=False)  # 使用 SimHash 去重
    .nonempty()  # 移除空值
)

# 导出处理后的数据集
processed_dataset.export_json('output.json')
