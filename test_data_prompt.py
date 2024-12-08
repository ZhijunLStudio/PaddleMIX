from paddlenlp.transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype="float16")


prompt = """你将得到一个描述某个图像的文本。我需要你从中提取出以下信息，并按照要求进行详细列举。**只有当文本中确实存在相关信息时，才需要列出该项**，如果某项信息在文本中没有提到，请跳过该项。

1. **颜色**：描述图像中的颜色信息。如果文本中提到了颜色，请列出这些颜色。
2. **形状**：描述图像中的物体形状。如果文本中提到了物体的形状，请列出这些形状。
3. **位置**：描述物体或场景的空间位置。如果文本中提到了物体的位置，请提供物体相对于背景或其他物体的位置（例如，左侧、右侧、居中、上方、下方、靠近等）。
4. **大小**：描述物体的相对尺寸。如果文本中提到了物体的尺寸（如大、小、高、低），请列出这些描述。
5. **方向**：描述物体的朝向或方向。如果文本中提到了物体的方向，请列出描述（如正面、背面、倾斜角度等）。
6. **关系**：描述物体之间的空间关系。如果文本中提到了物体之间的关系（如“在……上”，“靠近”，“相对位置”等），请列出这些关系。
7. **动作或状态**：描述物体的状态或动作。如果文本中提到了物体的动作或状态（如移动、静止、消失、打开、关闭等），请列出这些动作或状态。
8. **类别**：列出物体的类别或类型。如果文本中提到了物体的类别（如汽车、猫、桌子、花等），请列出这些类别。

请按照以下格式返回结果：
1. 颜色: [列出颜色]（如果没有提到颜色，请不列出此项）
2. 形状: [列出形状]（如果没有提到形状，请不列出此项）
3. 位置: [描述位置]（如果没有提到位置，请不列出此项）
4. 大小: [描述大小]（如果没有提到大小，请不列出此项）
5. 方向: [描述方向]（如果没有提到方向，请不列出此项）
6. 关系: [描述物体关系]（如果没有提到关系，请不列出此项）
7. 动作或状态: [描述动作或状态]（如果没有提到动作或状态，请不列出此项）
8. 类别: [列出类别]（如果没有提到类别，请不列出此项）

文本： "[text_input]"
"""


# # 定义你想要分析的文本
input_text = "在一张阳光明媚的照片中，一只橙色的猫正坐在白色的沙发上。猫的尾巴弯曲向上，眼睛注视着窗外。窗外有几棵绿色的树，背景是蓝色的天空。猫的位置在沙发的左侧，靠近窗户。"

# 拼接最终的 prompt
splice_prompt = prompt.replace("text_input", input_text)
print("splice_prompt:", splice_prompt)

input_features = tokenizer(splice_prompt, return_tensors="pd")
outputs = model.generate(**input_features, max_length=128)
print(tokenizer.batch_decode(outputs[0], skip_special_tokens=True))




# import json
# from paddlenlp.transformers import AutoTokenizer, AutoModelForCausalLM
# # 加载模型和 tokenizer
# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype="float16")

# # 加载配置文件
# with open('/home/lizhijun/PaddleMIX-develop/paddlemix/datacopilot/prompt/description_template.json', 'r', encoding='utf-8') as file:
#     config = json.load(file)

# # 定义你想要分析的文本
# input_text = "在一张阳光明媚的照片中，一只橙色的猫正坐在白色的沙发上。猫的尾巴弯曲向上，眼睛注视着窗外。窗外有几棵绿色的树，背景是蓝色的天空。猫的位置在沙发的左侧，靠近窗户。"

# # 拼接最终的 prompt
# prompt = config["prompt_body"].replace("text_input", input_text)
# print("prompt:", prompt)


# # 使用 tokenizer 对 prompt 进行编码
# input_features = tokenizer(prompt, return_tensors="pd")

# # 使用模型生成输出
# outputs = model.generate(**input_features, max_length=128)

# # 解码输出并打印
# print(tokenizer.batch_decode(outputs[0], skip_special_tokens=True))

