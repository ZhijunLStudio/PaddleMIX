import os
import json
from PIL import Image, ImageDraw, ImageFont
import hashlib

# 配置
FONT_PATH = "PaddleNLP/font/SimHei.ttf"  # 替换为你的字体路径
WHITE_SPACE_HEIGHT = 50  # 初始白布高度
TEXT_PADDING = 10  # 文本与图像的间距
MAX_LINE_WIDTH = 30  # 每行最大字符数（中文字符较宽）
OUTPUT_DIR = "model_results"  # 输出文件夹

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

def add_white_space(image: Image.Image, text: str, font: ImageFont.FreeTypeFont) -> Image.Image:
    """
    在图像下方添加白布，并绘制文本。
    """
    draw = ImageDraw.Draw(image)
    lines = split_text(text, MAX_LINE_WIDTH)
    white_space_height = WHITE_SPACE_HEIGHT + len(lines) * 25

    # 创建新图像
    new_img = Image.new("RGB", (image.width, image.height + white_space_height), "white")
    new_img.paste(image, (0, 0))

    # 绘制文本
    draw = ImageDraw.Draw(new_img)
    text_start_y = image.height + TEXT_PADDING
    for line in lines:
        draw.text((TEXT_PADDING, text_start_y), line, fill="black", font=font)
        text_start_y += 25

    return new_img

def split_text(text: str, max_line_width: int) -> list:
    """
    将文本按长度分割为多行。
    """
    lines = []
    current_line = ""
    for char in text:
        current_line += char
        if len(current_line) >= max_line_width:
            lines.append(current_line)
            current_line = ""
    if current_line:
        lines.append(current_line)
    return lines

def hash_string(s: str) -> str:
    """
    对字符串进行哈希处理，避免文件名过长。
    """
    return hashlib.md5(s.encode('utf-8')).hexdigest()

def process_json_files(json_dir: str):
    """
    读取文件夹中的 JSON 文件，处理每个图像。
    """
    font = ImageFont.truetype(FONT_PATH, size=18)  # 使用更大的字体以保证可读性

    for json_file in os.listdir(json_dir):
        if not json_file.endswith(".json"):
            continue

        json_path = os.path.join(json_dir, json_file)
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for entry in data:
            image_path = entry.get("image")
            if not image_path or not os.path.exists(image_path):
                print(f"图像路径无效: {image_path}")
                continue

            image = Image.open(image_path)

            for idx, result in enumerate(entry.get("results", [])):
                question = result.get("question", "No question provided.")
                answer = result.get("response", "No response provided.")
                text = f"Q: {question}\nA: {answer}"

                # 为当前问题创建新的图像
                output_image = add_white_space(image, text, font)

                # 保存文件，文件名基于 JSON 文件路径、问题索引和问题内容的哈希
                base_name = os.path.splitext(os.path.basename(json_file))[0]
                question_hash = hash_string(question)
                output_name = f"{base_name}_Q{idx}_{question_hash}.png"
                output_path = os.path.join(OUTPUT_DIR, output_name)
                output_image.save(output_path)
                print(f"保存处理后的图像: {output_path}")

# 示例调用
process_json_files("path_to_your_json_folder")