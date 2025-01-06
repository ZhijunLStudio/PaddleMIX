import os
import json
import shutil

def extract_and_save_images(json_path, output_base_dir):
    # 确保输出目录存在
    os.makedirs(output_base_dir, exist_ok=True)
    
    # 加载 JSON 文件
    with open(json_path, "r") as f:
        data = json.load(f)
    
    # 遍历 JSON 数据
    for entry in data:
        image_path = entry.get("image")
        if not image_path:
            continue
        
        # 构造原始图像的路径和目标保存路径
        output_image_path = os.path.join(output_base_dir, image_path)
        
        # 创建保存目录（如果不存在）
        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
        
        try:
            # 复制图像文件
            shutil.copy(image_path, output_image_path)
            print(f"Saved: {output_image_path}")
        except Exception as e:
            print(f"Failed to save {image_path}: {e}")

# 使用方法
json_file = "random_samples.json"  # 替换为你的 JSON 文件路径
output_dir = "random_samples_img"   # 替换为你的目标输出目录
extract_and_save_images(json_file, output_dir)
