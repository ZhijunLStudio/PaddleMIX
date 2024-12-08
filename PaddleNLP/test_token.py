# from paddlenlp.transformers import AutoTokenizer, AutoModelForCausalLM
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype="float16")
# input_features = tokenizer("你好！请自我介绍一下。", return_tensors="pd")
# print("Tokens:", input_features)


# # 获取token的数量
# token_count = len(input_features['input_ids'][0])  # 输入是batch格式，这里取第一个样本
# print(f"Token count: {token_count}")




# print(input_features['input_ids'])
# outputs = model.generate(**input_features, max_length=128)
# print(tokenizer.batch_decode(outputs[0], skip_special_tokens=True))
# ['我是一个AI语言模型，我可以回答各种问题，包括但不限于：天气、新闻、历史、文化、科学、教育、娱乐等。请问您有什么需要了解的吗？']


# from paddlenlp.transformers import AutoTokenizer, AutoModelForCausalLM
# import paddle

# # 初始化分词器和模型
# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")


# # 模拟 8 个输入句子
# texts = [
#     "你好！请自我介绍一下。",
#     "今天的天气怎么样？",
#     "你能帮我写一篇关于人工智能的文章吗？",
#     "Can you help me write an article about artificial intelligence?",
#     "帮我翻译这段话。",
#     "最近有什么新闻值得关注？",
#     "Can you generate a poem?",
#     "你可以生成一段诗歌吗？"
# ]

# # 对输入的 8 个句子进行分词
# input_features = tokenizer(texts, padding=True, truncation=True, return_tensors="pd")

# # 打印分词结果
# print("Tokens:", input_features)

# # 使用 attention_mask 计算每句话实际的 token 数量
# # attention_mask 为 1 的位置表示实际 token
# attention_mask = input_features['attention_mask']
# token_counts = paddle.sum(attention_mask, axis=1).numpy()

# print(f"Batch size: {len(input_features['input_ids'])}")
# print(f"Token counts for each sentence: {token_counts}")



# token_0 = input_features['input_ids'][0]
# # 将单个 token 映射回原始文字
# for token_id in token_0:
#     decoded_text = tokenizer.decode([token_id])  # 这里传入单个 token 的 ID
#     print(f"Token ID: {token_id} -> Decoded text: '{decoded_text}'")


from paddlemix.datacopilot.core import MMDataset

# 加载数据集
dataset = MMDataset.from_json('./my.json')

# 调用所有分析功能
analysis_results = dataset.run_token_analysis()

# 输出分析结果
for analysis_type, result in analysis_results.items():
    print(f"{analysis_type}: {result}")
