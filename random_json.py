import json
import random

# 加载原始 JSON 文件
with open('datasets/llava/llava_v1_5_mix665k.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 计算要选取的数量（1%）
num_samples = max(1, len(data) // 10000)  # 至少选取 1 条数据

# 随机选择 1% 的数据
random_samples = random.sample(data, num_samples)

# 将选取的数据保存到新的 JSON 文件
with open('random_samples.json', 'w', encoding='utf-8') as f:
    json.dump(random_samples, f, indent=2, ensure_ascii=False)

print(f"成功保存 {num_samples} 条数据到 'random_samples_66.json'")
