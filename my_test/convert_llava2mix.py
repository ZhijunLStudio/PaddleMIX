import json

# 读取原始的 JSON 数据
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

# 转换为目标格式
def transform_data(raw_data):
    result = []

    for item in raw_data:
        # 检查是否存在 'image' 键，如果没有，则设置为空
        image = item.get('image', '')  # 如果没有 'image' 键，默认为空字符串
        
        # 如果存在 'image' 键，则拼接路径
        if image:
            image = 'datasets/llava/valid_images/' + image
        
        conversations = []
        for i in range(0, len(item['conversations']), 2):
            human_message = item['conversations'][i]['value']
            gpt_message = item['conversations'][i+1]['value'] if i+1 < len(item['conversations']) else ''
            conversations.append([human_message, gpt_message])

        # 构造转换后的数据结构
        transformed_item = {
            "image": image,
            "conversations": conversations
        }

        result.append(transformed_item)

    return result

# 将转换后的数据写入到一个新的 JSON 文件
def save_json(data, output_path):
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

# 主函数
def main(input_file, output_file):
    # 加载原始数据
    raw_data = load_json(input_file)
    
    # 转换数据
    transformed_data = transform_data(raw_data)
    
    # 保存转换后的数据
    save_json(transformed_data, output_file)

# 调用主函数
if __name__ == "__main__":
    input_file = "random_samples.json"  # 替换为你的输入文件路径
    output_file = "datasets/llava/random_samples_newconv_1.json"  # 输出文件路径
    main(input_file, output_file)
