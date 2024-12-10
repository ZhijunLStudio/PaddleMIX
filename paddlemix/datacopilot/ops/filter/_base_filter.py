from PIL import Image


def validate_image(item):
    """
    验证数据集项中的图片是否可加载。
    
    参数:
        item (dict): 包含 'image' 键的数据集项。
        
    返回:
        bool: 如果图片有效，则返回 True， 否则返回 False。
    """
    image_path = item['image']
    try:
        with Image.open(image_path) as img:
            img.load()  # 强制加载图片数据
        return True
    except Exception as e:
        print(f"无效图片: {image_path}, 错误: {e}")
        return False


# def validate_conversation(item):
#     """
#     验证数据集项中的 'conversations' 是否符合角色交替规则，
#     确保 'Human' 和 'GPT' 消息交替出现，并且数量匹配。

#     参数:
#         item (dict): 包含 'conversations' 键的数据集项。

#     返回:
#         bool: 如果对话有效，则返回 True，否则返回 False。
#     """
#     if 'conversations' not in item or not isinstance(item['conversations'], list):
#         print(f"数据项中的对话格式无效: {item}")
#         return False

#     conversations = item['conversations']
#     human_count = 0
#     gpt_count = 0

#     for conv in conversations:
#         # 每对对话必须是包含两个元素的列表或元组
#         if not isinstance(conv, (list, tuple)) or len(conv) != 2:
#             print(f"数据项中的对话对结构无效: {item}")
#             return False
#         # 对话对的每一部分应为字符串
#         if not all(isinstance(part, str) for part in conv):
#             print(f"对话对应包含字符串: {item}")
#             return False
#         # 内容不能为空
#         if not all(part.strip() for part in conv):
#             print(f"数据项中的对话内容为空: {item}")
#             return False
#         # 统计 'Human' 和 'GPT' 出现的次数
#         if 'Human' in conv[0]:
#             human_count += 1
#         if 'GPT' in conv[1]:
#             gpt_count += 1

#     # 确保 'Human' 和 'GPT' 的出现次数相同
#     if human_count != gpt_count:
#         print(f"对话中 'Human' 和 'GPT' 的数量不匹配: {item}")
#         return False

#     return True




def validate_conversation(item):
    """
    验证数据集项中的 'conversations' 是否符合角色交替规则，
    确保 'Human' 和 'GPT' 消息交替出现，并且数量匹配。
    如果包含 'USER' 或 'ASSISTANT'，则直接返回 False，并打印错误信息。

    参数:
        item (dict): 包含 'conversations' 键的数据集项。

    返回:
        bool: 如果对话有效，则返回 True，否则返回 False。
    """
    if 'conversations' not in item or not isinstance(item['conversations'], list):
        print(f"数据项中的对话格式无效: {item}")
        return False

    conversations = item['conversations']
    human_count = 0
    gpt_count = 0

    for conv in conversations:
        # 检查是否包含 'USER' 或 'ASSISTANT'
        if any('USER' in part or 'ASSISTANT' in part for part in conv):
            print(f"对话中含有 'USER' 或 'ASSISTANT' 的内容: {item}")
            return False
        
        # 每对对话必须是包含两个元素的列表或元组
        if not isinstance(conv, (list, tuple)) or len(conv) != 2:
            print(f"数据项中的对话对结构无效: {item}")
            return False
        
        # 对话对的每一部分应为字符串
        if not all(isinstance(part, str) for part in conv):
            print(f"对话对应包含字符串: {item}")
            return False
        
        # 内容不能为空
        if not all(part.strip() for part in conv):
            print(f"数据项中的对话内容为空: {item}")
            return False
        
        # 统计 'Human' 和 'GPT' 出现的次数
        if 'Human' in conv[0]:
            human_count += 1
        if 'GPT' in conv[1]:
            gpt_count += 1

    # 确保 'Human' 和 'GPT' 的出现次数相同
    if human_count != gpt_count:
        print(f"对话中 'Human' 和 'GPT' 的数量不匹配: {item}")
        return False

    return True
