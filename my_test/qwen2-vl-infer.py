# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
from paddlenlp.transformers import Qwen2Tokenizer

from paddlemix.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration
from paddlemix.processors.qwen2_vl_processing import (
    Qwen2VLImageProcessor,
    Qwen2VLProcessor,
    process_vision_info,
)

MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"
model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_NAME, dtype="bfloat16")

image_processor = Qwen2VLImageProcessor()
tokenizer = Qwen2Tokenizer.from_pretrained(MODEL_NAME)
processor = Qwen2VLProcessor(image_processor, tokenizer)


messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "paddlemix/demo_images/examples_image1.jpg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]
image_inputs, video_inputs = process_vision_info(messages)

# New instructions
instruction = """
You are an AI visual assistant, and you are seeing a single image. What you see are provided with five sentences, describing the same image you are looking at. Answer all questions as you are seeing the image.

Design a conversation between you and a person asking about this photo. The answers should be in a tone that a visual AI assistant is seeing the image and answering the question.
Ask diverse questions and give corresponding answers.

Include questions asking about the visual content of the image, including the object types, counting the objects, object actions, object locations, relative positions between objects, etc. Only include questions that have definite answers:
(1) one can see the content in the image that the question asks about and can answer confidently;
(2) one can determine confidently from the image that it is not in the image.
Do not ask any question that cannot be answered confidently.

Also include complex questions that are relevant to the content in the image, for example, asking about background knowledge of the objects in the image, asking to discuss about events happening in the image, etc. Again, do not ask about uncertain details.
Provide detailed answers when answering complex questions. For example, give detailed examples or reasoning steps to make the content more convincing and well-organized. You can include multiple paragraphs if necessary.
"""

# Combine the new instruction and the question
image_pad_token = "<|vision_start|><|image_pad|><|vision_end|>"
text = f"<|im_start|>system\n{instruction}<|im_end|>\n<|im_start|>user\n{image_pad_token}<|im_end|>\n<|im_start|>assistant\n"

inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pd",
)

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=1280)
output_text = processor.batch_decode(generated_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
print("output_text:\n", output_text[0])

