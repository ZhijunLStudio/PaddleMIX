from paddlemix.datacopilot.core import MMDataset

# 处理每个数据项
def func(item):
    # 如果 item 中没有 'image' 键，则返回原始 item（不做任何修改）
    if 'image' not in item:
        return item  # 保持原数据，不返回 None
    
    # 修改图像路径，其他内容不变
    item['image'] = 'datasets/llava/valid_images/' + item['image']
    return item


# 数据集路径
anno_path = 'random_samples.json'
print('loading dataset...')
dataset = MMDataset.from_json(anno_path)

# 转换数据集
print('convert dataset...')
dataset = dataset.map(func, max_workers=8)

# 输出数据集长度
print(len(dataset))

# 导出新的 JSON 文件
print('export dataset...')
dataset.export_json(anno_path.replace('.json', '_newconv.json'))
print('done!')
