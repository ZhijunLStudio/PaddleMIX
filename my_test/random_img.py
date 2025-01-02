import os
import random
import shutil

def select_random_images(source_folder, target_folder, num_images=6):
    """
    从源文件夹中随机选择图片复制到目标文件夹。

    :param source_folder: 源文件夹路径
    :param target_folder: 目标文件夹路径
    :param num_images: 要选择的图片数量
    """
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # 获取源文件夹中所有图片文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    images = [
        file for file in os.listdir(source_folder)
        if os.path.isfile(os.path.join(source_folder, file)) and
        os.path.splitext(file)[1].lower() in image_extensions
    ]

    if len(images) < num_images:
        print(f"源文件夹中的图片不足 {num_images} 张，仅有 {len(images)} 张。")
        num_images = len(images)

    # 随机选择图片
    selected_images = random.sample(images, num_images)

    # 复制到目标文件夹
    for image in selected_images:
        source_path = os.path.join(source_folder, image)
        target_path = os.path.join(target_folder, image)
        shutil.copy2(source_path, target_path)

    print(f"已随机选择并复制 {len(selected_images)} 张图片到目标文件夹：{target_folder}")


# 示例使用
source_folder = "/home/lizhijun/llm/PaddleMix/datasets/mmmu/images"  # 替换为源文件夹路径
target_folder = "/home/lizhijun/llm/PaddleMix/datasets/mmmu/random_images"  # 替换为目标文件夹路径
select_random_images(source_folder, target_folder)