import os
import random
import shutil

def collect_images(src_dir, extensions):
    """
    收集指定目录下的所有图片文件（包括子目录）
    :param src_dir: 源文件夹路径
    :param extensions: 图片文件的扩展名列表
    :return: 图片文件的完整路径列表
    """
    image_paths = []
    for root, _, files in os.walk(src_dir):
        for file in files:
            if file.lower().endswith(tuple(extensions)):
                image_paths.append(os.path.join(root, file))
    return image_paths

def copy_random_images(image_paths, target_dir, num_images):
    """
    随机选择图片并复制到目标文件夹
    :param image_paths: 图片路径列表
    :param target_dir: 目标文件夹路径
    :param num_images: 要复制的图片数量
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # 随机选择图片
    selected_images = random.sample(image_paths, min(num_images, len(image_paths)))

    for idx, img_path in enumerate(selected_images, 1):
        target_path = os.path.join(target_dir, os.path.basename(img_path))
        shutil.copy2(img_path, target_path)  # 复制图片，保留元数据
        print(f"[{idx}/{len(selected_images)}] Copied: {img_path} -> {target_path}")

if __name__ == "__main__":
    # 配置参数
    src_dir = "datasets/llava/valid_images"  # 源文件夹路径
    target_dir = "datasets/random_1w"  # 目标文件夹路径
    num_images = 10000  # 随机选择的图片数量
    extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp']  # 支持的图片格式

    # 收集所有图片
    print("Collecting images...")
    image_paths = collect_images(src_dir, extensions)
    print(f"Found {len(image_paths)} images in '{src_dir}'.")

    # 复制随机选择的图片
    print("Copying random images...")
    copy_random_images(image_paths, target_dir, num_images)
    print(f"Finished copying images to '{target_dir}'.")
