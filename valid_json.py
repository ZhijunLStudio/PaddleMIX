# import json

# # 输入 JSON 文件路径
# input_file = 'datasets/llava/train_chatml.json'  # 替换为您的 JSON 文件路径
# output_file = 'output.json'

# # 指定范围
# start_index = 1501 * 16
# end_index = 1503 * 16

# def extract_data(input_file, output_file, start_index, end_index):
#     try:
#         # 读取 JSON 文件
#         with open(input_file, 'r', encoding='utf-8') as f:
#             data = json.load(f)
        
#         # 提取指定范围的数据
#         extracted_data = data[start_index:end_index]
        
#         # 保存为新的 JSON 文件
#         with open(output_file, 'w', encoding='utf-8') as f:
#             json.dump(extracted_data, f, ensure_ascii=False, indent=4)
        
#         print(f"提取完成，保存到文件: {output_file}")
#     except Exception as e:
#         print(f"处理时出现错误: {e}")

# # 调用函数
# extract_data(input_file, output_file, start_index, end_index)


# import json

# # 文件路径
# input_file = 'datasets/llava/train_chatml.json'  # 输入 JSON 文件路径

# # 读取和检查 JSON 数据
# def check_conversations(input_file):
#     try:
#         # 读取 JSON 文件
#         with open(input_file, 'r', encoding='utf-8') as f:
#             data = json.load(f)
        
#         # 标记是否有不正常的对话
#         has_invalid = False

#         # 遍历每个项目
#         for idx, item in enumerate(data):
#             image = item.get('image', '未知图片路径')
#             conversations = item.get('conversations', [])
            
#             # 遍历每个子对话
#             for conv_idx, conversation in enumerate(conversations):
#                 print("conversation:", conversation)
#                 if not isinstance(conversation, list):
#                     print(f"第 {idx} 项，第 {conv_idx} 子对话不是列表格式：{conversation}")
#                     has_invalid = True
#                     continue
#                 print(len(conversation))
#                 if len(conversation) != 2:
#                     print(f"不正常的对话发现 - 图片: {image}")
#                     print(f"子对话索引: {conv_idx}")
#                     print(f"对话内容: {conversation}\n")
#                     has_invalid = True
#             print("-"*50)
        
#         if not has_invalid:
#             print("所有对话都是正常的，每个子对话都有问题和回答。")
    
#     except FileNotFoundError:
#         print(f"文件未找到: {input_file}")
#     except json.JSONDecodeError as e:
#         print(f"JSON 解码错误: {e}")
#     except Exception as e:
#         print(f"处理 JSON 文件时出错: {e}")

# # 执行检查
# if __name__ == "__main__":
#     check_conversations(input_file)



# 检查图像有效性
# import os
# import json
# from tqdm import tqdm
# from PIL import Image
# from concurrent.futures import ProcessPoolExecutor

# def check_single_image(image_path, base_folder):
#     """
#     Check the status of a single image.

#     Args:
#         image_path (str): Relative path to the image.
#         base_folder (str): Base folder where the image is located.

#     Returns:
#         str: Message indicating the status of the image.
#     """
#     full_path = os.path.join(base_folder, image_path)
#     try:
#         # Open and fully load the image to test for corruption
#         with Image.open(full_path) as img:
#             img.load()  # Force loading of image data
#         return None  # Return None if the image is fine
#     except Exception as e:
#         return f"Error loading {full_path}: {e}"


# def check_image_status_parallel(json_path, base_folder, num_workers=16):
#     """
#     Check the status of images listed in a JSON file using multiple processes.

#     Args:
#         json_path (str): Path to the JSON file.
#         base_folder (str): Base folder where images are located.
#         num_workers (int): Number of parallel workers.

#     Returns:
#         None
#     """
#     # Load the JSON file
#     with open(json_path, 'r') as f:
#         data = json.load(f)

#     # Extract image paths
#     image_paths = [item['image'] for item in data]

#     # Use ProcessPoolExecutor for parallel processing
#     with ProcessPoolExecutor(max_workers=num_workers) as executor:
#         # Create a tqdm progress bar
#         with tqdm(total=len(image_paths), desc="Checking images") as pbar:
#             # Submit tasks to the pool
#             futures = [executor.submit(check_single_image, path, base_folder) for path in image_paths]
            
#             # Process results as they complete
#             for future in futures:
#                 result = future.result()
#                 if result:  # Print any errors
#                     print(result)
#                 pbar.update(1)


# if __name__ == "__main__":
#     # Specify paths
#     json_file = "datasets/llava/train_chatml.json"  # Replace with your JSON file path
#     image_base_folder = "./"  # Replace with your image folder path

#     # Run the function
#     check_image_status_parallel(json_file, image_base_folder, num_workers=16)


# import json

# def validate_conversations(json_file):
#     def is_valid_conversation(conversations):
#         # 匹配不同角色的句子，假设是 "Human" 和 "GPT" 的交替模式
#         match_role_human = [c[0] for c in conversations if c[0]]
#         match_role_gpt = [c[1] for c in conversations if len(c) > 1 and c[1]]
#         return len(match_role_human) == len(match_role_gpt)
    
#     invalid_records = []
    
#     with open(json_file, 'r', encoding='utf-8') as f:
#         data = json.load(f)
    
#     for idx, record in enumerate(data):
#         if 'conversations' not in record or not isinstance(record['conversations'], list):
#             invalid_records.append((idx, "Missing or invalid 'conversations' key", record))
#             continue
        
#         conversations = record['conversations']
#         if not is_valid_conversation(conversations):
#             invalid_records.append((idx, "Mismatched roles in 'conversations'", record))
    
#     return invalid_records

# # 使用文件路径替换为你的 JSON 文件路径
# json_file_path = "datasets/llava/train_chatml_filter.json"

# # 检查 JSON 文件
# invalid_data = validate_conversations(json_file_path)

# if invalid_data:
#     print(f"Found {len(invalid_data)} invalid records:")
#     for idx, reason, record in invalid_data:
#         print(f"Record #{idx}: {reason}")
#         print(f"Full record:\n{json.dumps(record, indent=2, ensure_ascii=False)}\n")
# else:
#     print("All records are valid!")


import json

def validate_conversations_strict(json_file):
    """
    Validate the conversations in the given JSON file to ensure compatibility with training logic.
    Specifically checks if each conversation has matching lengths of 'Human' and 'GPT' exchanges.

    Args:
        json_file (str): Path to the JSON file to validate.

    Returns:
        list: A list of invalid records, each containing (index, error_reason, full_record).
    """
    def is_valid_conversation(conversations):
        """
        Validate if the conversation follows the alternating 'Human' and 'GPT' format,
        and checks if the 'Human' and 'GPT' message counts match.

        Args:
            conversations (list): The conversation list.

        Returns:
            bool: True if valid, otherwise False.
        """
        human_count = 0
        gpt_count = 0

        for conv in conversations:
            # Each conversation pair must be a list or tuple with exactly two elements
            if not isinstance(conv, (list, tuple)) or len(conv) != 2:
                return False
            # Both parts of the pair should be strings
            if not all(isinstance(part, str) for part in conv):
                return False
            # Content should not be empty
            if not all(part.strip() for part in conv):
                return False
            # Count occurrences of human and gpt
            if 'Human' in conv[0]:
                human_count += 1
            if 'GPT' in conv[1]:
                gpt_count += 1
        if human_count != gpt_count:
            print("conversations:", conversations)
            print(f"Human count: {human_count}, GPT count: {gpt_count}")
        
        # Ensure that the counts of 'Human' and 'GPT' are the same
        return human_count == gpt_count

    invalid_records = []

    # Load the JSON data
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for idx, record in enumerate(data):
        try:
            # Check for 'conversations' key and its structure
            if 'conversations' not in record or not isinstance(record['conversations'], list):
                invalid_records.append((idx, "Missing or invalid 'conversations' key", record))
                continue

            # Check conversation structure
            conversations = record['conversations']
            if not is_valid_conversation(conversations):
                invalid_records.append((idx, "Invalid 'conversations' structure (Human and GPT counts don't match)", record))
                continue

            # Simulate training constraints
            sources = ["".join(conv) for conv in conversations]
            if len(sources) == 0:
                invalid_records.append((idx, "Empty conversation content", record))
            elif not all(isinstance(source, str) and source.strip() for source in sources):
                invalid_records.append((idx, "Invalid conversation strings", record))

        except Exception as e:
            invalid_records.append((idx, f"Error during validation: {e}", record))

    return invalid_records

# Example usage:
json_file_path = "datasets/llava/train_chatml_filter.json"

# Validate JSON file
invalid_data = validate_conversations_strict(json_file_path)

if invalid_data:
    print(f"Found {len(invalid_data)} invalid records:")
    for idx, reason, record in invalid_data:
        print(f"Record #{idx}: {reason}")
        print(f"Full record:\n{json.dumps(record, indent=2, ensure_ascii=False)}\n")
else:
    print("All records are valid!")
