# from deepeval import evaluate
# from deepeval.metrics import AnswerRelevancyMetric
# from deepeval.test_case import LLMTestCase

# answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.7)
# test_case = LLMTestCase(
#     input="What if these shoes don't fit?",
#     # Replace this with the actual output from your LLM application
#     actual_output="We offer a 30-day full refund at no extra costs.",
#     retrieval_context=["All customers are eligible for a 30 day full refund at no extra costs."]
# )
# evaluate([test_case], [answer_relevancy_metric])


# from paddlenlp.transformers import AutoTokenizer, AutoModelForCausalLM
# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype="float16")
# input_features = tokenizer("你好！请自我介绍一下。", return_tensors="pd")
# outputs = model.generate(**input_features, max_length=128)
# print(tokenizer.batch_decode(outputs[0], skip_special_tokens=True))



# from paddlenlp.transformers import AutoTokenizer, AutoModelForCausalLM
# from deepeval.models.base_model import DeepEvalBaseLLM
# import asyncio
# from deepeval.test_case import LLMTestCase

# from pydantic import BaseModel
# from lmformatenforcer import JsonSchemaParser
# from lmformatenforcer.integrations.transformers import (
#     build_transformers_prefix_allowed_tokens_fn,
# )




# class Qwen2_5_0_5B(DeepEvalBaseLLM):
#     def __init__(self, model, tokenizer):
#         self.model = model
#         self.tokenizer = tokenizer

#     def load_model(self):
#         return self.model

#     def generate(self, prompt: str) -> str:
#         model = self.load_model()
#         input_features = self.tokenizer(prompt, return_tensors="pd")
#         outputs = model.generate(**input_features, max_length=128)
#         output_json = self.tokenizer.batch_decode(outputs[0], skip_special_tokens=True)[0]
#         print("output_json:",output_json)
#         # output_text = output_json["statements"]
#         # print("output_text:",output_text)

#         return output_json

#     async def a_generate(self, prompt: str) -> str:
#         loop = asyncio.get_running_loop()
#         return await loop.run_in_executor(None, self.generate, prompt)

#     def get_model_name(self):
#         return "Qwen-2.5-0.5B"

# # 载入模型和分词器
# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype="float16")

# # 创建自定义 Qwen 模型
# qwen_model = Qwen2_5_0_5B(model=model, tokenizer=tokenizer)

# # 测试模型生成
# prompt = "你好！请自我介绍一下。"
# print(qwen_model.generate(prompt))  # 同步调用

# test_case = LLMTestCase(
#     input=prompt,
#     actual_output="我是人工智能助手，由计算机程序设计和人工智能技术开发的，可以回答各种问题，提供信息和帮助用户解决问题。我可以通过自然语言处理技术理解人类语言，并根据用户的需求提供相应的答案或建议。",
# )


# from deepeval.metrics import AnswerRelevancyMetric
# metric = AnswerRelevancyMetric(model=qwen_model)
# metric.measure(test_case)
# print(metric.score)
# # print(metric.reason)




# from deepeval.metrics import GEval
# from deepeval.test_case import LLMTestCase, LLMTestCaseParams
# from deepeval.models.base_model import DeepEvalBaseLLM
# from paddlenlp.transformers import AutoTokenizer, AutoModelForCausalLM
# import asyncio

# class Qwen2_5_0_5B(DeepEvalBaseLLM):
#     def __init__(self, model, tokenizer):
#         self.model = model
#         self.tokenizer = tokenizer

#     def load_model(self):
#         return self.model

#     def load_grammar(self):
#         # If needed, load any specific grammar for your task here.
#         # In this case, we don't need grammar, so just return None
#         return None

#     def _call(self, prompt: str) -> str:
#         model = self.load_model()

#         # Tokenize the prompt and generate the output
#         input_features = self.tokenizer(prompt, return_tensors="pd")
#         outputs = model.generate(**input_features, max_length=128)

#         # Decode and return the generated text
#         output_text = self.tokenizer.batch_decode(outputs[0], skip_special_tokens=True)[0]
#         print("模型的输入文本为：", prompt)
#         print("模型输出文本为：", output_text)
#         return output_text

#     def generate(self, prompt: str) -> str:
#         """
#         Implement the 'generate' method that DeepEvalBaseLLM expects.
#         This method calls the '_call' method to generate a response.
#         """
#         return self._call(prompt)

#     async def a_generate(self, prompt: str) -> str:
#         loop = asyncio.get_running_loop()
#         return await loop.run_in_executor(None, self.generate, prompt)

#     def get_model_name(self):
#         return "Qwen-2.5-0.5B"


# # Load the model and tokenizer
# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype="float16")

# # Create the custom Qwen model
# qwen_model = Qwen2_5_0_5B(model=model, tokenizer=tokenizer)


# # Define the test case
# test_case = LLMTestCase(
#     input="你好！请自我介绍一下。",
#     actual_output="我是llama。",
# )

# # GEval setup for a different metric (e.g., Coherence)
coherence_metric = GEval(
    name="Coherence",
    evaluation_steps=[
        "Compare the 'actual output' directly with the 'expected output' to determine if the main answer aligns factually."
        "Do not penalize for any missing explanation, details, reasoning, or verbosity.",
        "Ensure the 'actual output' does not introduce any factual errors or contradictions in relation to the 'expected output'.",
    ],
    model=qwen_model,
    criteria="Coherence - determine if the actual output is coherent with the input.",
    evaluation_params=[
        LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT
    ],
)

# # Example of applying coherence metric
# coherence_metric.measure(test_case)
# coherence = coherence_metric.score

# # Print the coherence score
# print(f"Coherence score: {coherence}")
# print(coherence_metric.reason)







from paddlenlp.transformers import AutoTokenizer, AutoModelForCausalLM
from deepeval.models.base_model import DeepEvalBaseLLM
import asyncio
from deepeval.test_case import LLMTestCase

from pydantic import BaseModel
# from lmformatenforcer import JsonSchemaParser
# from lmformatenforcer.integrations.transformers import (
#     build_transformers_prefix_allowed_tokens_fn,
# )
import instructor



class Qwen2_5_0_5B(DeepEvalBaseLLM):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def load_model(self):
        return self.model

    # def generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        # model = self.load_model()
        # input_features = self.tokenizer(prompt, return_tensors="pd")

        # # Create parser required for JSON confinement using lmformatenforcer
        # parser = JsonSchemaParser(schema.schema())
        # prefix_function = build_transformers_prefix_allowed_tokens_fn(
        #     pipeline.tokenizer, parser
        # )


        # outputs = model.generate(**input_features, max_length=128)
        # decode_output = self.tokenizer.batch_decode(outputs[0], skip_special_tokens=True)[0]
        # print("decode_output:",decode_output)
        # json_result = json.loads(decode_output)
        # # output_text = output_json["statements"]
        # # print("output_text:",output_text)

        # return schema(**json_result)



    def generate(self, prompt: str, schema: BaseModel) -> BaseModel:


        model = self.load_model()
        input_features = self.tokenizer(prompt, return_tensors="pd")

        # Create parser required for JSON confinement using lmformatenforcer
        # parser = JsonSchemaParser(schema.schema())
        # prefix_function = build_transformers_prefix_allowed_tokens_fn(
        #     pipeline.tokenizer, parser
        # )


        outputs = model.generate(**input_features, max_length=128)
        decode_output = self.tokenizer.batch_decode(outputs[0], skip_special_tokens=True)[0]
        print("decode_output:",decode_output)

        client = self.load_model()
        instructor_client = instructor.from_gemini(
            client=client,
            mode=instructor.Mode.GEMINI_JSON,
        )
        resp = instructor_client.messages.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            response_model=schema,
        )
        return resp

    async def a_generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.generate, prompt, schema)

    def get_model_name(self):
        return "Qwen-2.5-0.5B"

# 载入模型和分词器
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype="float16")

# 创建自定义 Qwen 模型
qwen_model = Qwen2_5_0_5B(model=model, tokenizer=tokenizer)

# 测试模型生成
prompt = "你好！请自我介绍一下。"
print(qwen_model.generate(prompt))  # 同步调用

test_case = LLMTestCase(
    input=prompt,
    actual_output="我是人工智能助手，由计算机程序设计和人工智能技术开发的，可以回答各种问题，提供信息和帮助用户解决问题。我可以通过自然语言处理技术理解人类语言，并根据用户的需求提供相应的答案或建议。",
)


from deepeval.metrics import AnswerRelevancyMetric
metric = AnswerRelevancyMetric(model=qwen_model)
metric.measure(test_case)
print(metric.score)
# print(metric.reason)