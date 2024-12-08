from paddlenlp.transformers import AutoTokenizer, AutoModelForCausalLM
from paddlemix.datacopilot.core import MMDataset, register
import os
from collections import Counter
from tqdm import tqdm
from typing import Dict
import json


# 加载模型的函数，支持传入模型名称
def load_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype="float16")
    return tokenizer, model


# 读取配置文件
# with open('/home/lizhijun/PaddleMIX-develop/paddlemix/datacopilot/prompt/judge_prompt.json', 'r', encoding='utf-8') as f:
#     config = json.load(f)

# # 获取模板
# prompt_template = config["IMPROVED_JUDGE_PROMPT"]


prompt_template = """
        You will be given a user_question and system_answer couple. Your task is to provide a 'total rating' scoring how well the system_answer answers the user concerns expressed in the user_question. Give your answer on a scale of 1 to 4, where 1 means that the system_answer is not helpful at all, and 4 means that the system_answer completely and helpfully addresses the user_question. Here is the scale you should use to build your answer: 1: The system_answer is terrible: completely irrelevant to the question asked, or very partial 2: The system_answer is mostly not helpful: misses some key aspects of the question 3: The system_answer is mostly helpful: provides support, but still could be improved 4: The system_answer is excellent: relevant, direct, detailed, and addresses all the concerns raised in the question Provide your feedback as follows: Feedback::: Evaluation: (your rationale for the rating, as a text) Total rating: (your rating, as a number between 1 and 4) You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer. Now here are the question and answer. Question: {question} Answer: {answer} Provide your feedback.
"""

@register()
def gpt_responses_judge(dataset: MMDataset, model_name: str = "Qwen/Qwen2.5-0.5B") -> Dict:

    # 加载指定的模型
    tokenizer, model = load_model(model_name)

    for item in tqdm(dataset):
        gpt_responses = []

        # 获取所有 'gpt' 的对话内容
        qa_pairs = [] 

        for i in range(0, len(item["conversations"]), 2):
            # Ensure we are correctly pairing 'human' questions with 'gpt' answers
            if item["conversations"][i]["from"] == "human" and item["conversations"][i + 1]["from"] == "gpt":
                question = item["conversations"][i]["value"]
                answer = item["conversations"][i + 1]["value"]
                qa_pairs.append((question, answer))

        # 处理每个问题和答案
        for question, answer in qa_pairs:
            # 替换占位符
            filled_prompt = prompt_template.format(question=question, answer=answer)

            # 使用 tokenizer 对输入文本进行编码
            input_features = tokenizer(filled_prompt, return_tensors="pd")

            # 生成模型的输出
            outputs = model.generate(**input_features, max_length=128)

            # 解码并获取分析结果
            analysis_result = tokenizer.batch_decode(outputs[0], skip_special_tokens=True)[0]
            print(f"Question: {question}")
            print(f"Answer: {answer}")
            print(f"Model Output: {analysis_result}")
            print("-" * 50)


    return analysis_result
