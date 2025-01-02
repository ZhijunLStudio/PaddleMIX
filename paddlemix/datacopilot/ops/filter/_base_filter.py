from PIL import Image
from ...core import T, MMDataset, register


def image_compliance_operator(
    item
    ) -> bool:
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



def conversation_compliance_operator(
    item
    ) -> bool:
    """
    验证数据集项中的 'conversations' 是否合规，
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
        
    return True


@register()
def valid_data_filter(dataset: MMDataset) -> MMDataset:
    # 过滤不可加载图像
    print("Filtering invalid images...")
    dataset = dataset.filter(
        func=image_compliance_operator, 
        max_workers=8, 
        progress=True
    )

    # 过滤无效文本
    print("Filtering invalid conversations...")
    dataset = dataset.filter(
        func=conversation_compliance_operator, 
        max_workers=8, 
        progress=True
    )
    return dataset
