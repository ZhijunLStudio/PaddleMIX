import os
import json
from PIL import Image
from tqdm import tqdm
from modelscope.msdatasets import MsDataset

# 加载数据集
ds = MsDataset.load('lmms-lab/MMMU')
test_split = ds['dev']
print("数据加载完成...")

# 定义保存路径
base_dir = "datasets/mmmu"
image_dir = os.path.join(base_dir, "images")
os.makedirs(image_dir, exist_ok=True)

# 初始化文本数据
questions_data = []

# 遍历数据集并保存图片和文本
for example in tqdm(test_split):
    id = example['id']
    question = example['question']
    options = example['options']
    answer = example['answer']
    explanation = example['explanation']
    
    # 收集图片并保存
    image_filenames = []
    for i in range(7):
        image_key = f'image_{i+1}'
        if example.get(image_key):  # 检查图片是否存在
            image = example[image_key]
            
            # 处理 P 和 RGBA 模式的图像
            if image.mode in ["P", "RGBA"]:
                image = image.convert("RGB")  # 转换为 RGB 模式
            
            image_filename = f"{id}_{i+1}.png"
            image_path = os.path.join(image_dir, image_filename)
            
            # 保存图片为 PNG
            image.save(image_path, format="PNG")
            image_filenames.append(image_filename)
    
    # 保存文本数据
    questions_data.append({
        "id": id,
        "question": question,
        "options": options,
        "answer": answer,
        "explanation": explanation,
        "images": image_filenames
    })

# 保存文本到 JSON 文件
questions_path = os.path.join(base_dir, "questions.json")
with open(questions_path, "w", encoding="utf-8") as f:
    json.dump(questions_data, f, ensure_ascii=False, indent=4)

print(f"数据集处理完成！图片保存在 {image_dir}，文本保存在 {questions_path}")