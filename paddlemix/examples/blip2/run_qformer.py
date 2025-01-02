# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
from dataclasses import dataclass, field

import numpy as np
import paddle
import paddle.distributed as dist
import requests
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import get_rng_state_tracker
from paddlenlp.trainer import PdArgumentParser, TrainingArguments
from PIL import Image

from paddlemix.models.blip2.modeling import Blip2ForConditionalGeneration
from paddlemix.models.blip2.utils import create_tokenizer, load_model
from paddlemix.processors.blip_processing import (
    Blip2Processor,
    BlipImageProcessor,
    BlipTextProcessor,
)
from paddlemix.models.blip2.configuration import Blip2Config
from paddlemix.utils.log import logger


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `PdArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    input_image: str = field(
        default="/home/lizhijun/PaddleMIX-develop/paddlemix/demo_images/examples_image1.jpg", metadata={"help": "The name of input image."}
    )  # "http://images.cocodataset.org/val2017/000000039769.jpg"
    prompt: str = field(
        default="Red panda in the woods", metadata={"help": "The prompt of the image to be generated."}
    )  # "Question: how many cats are there? Answer:"


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="paddlemix/blip2-caption-opt2.7b",
        metadata={"help": "Path to pretrained model or model identifier"},
    )

    text_model_name_or_path: str = field(
        default="facebook/opt-2.7b",
        metadata={"help": "The type of text model to use (OPT, T5)."},
    )
    image_size: int = field(default=224, metadata={"help": " Image size for training. (default:224)"})
    train_mode = "stage1"
    qformer_tokenizer_name: str = field(default=None, metadata={"help": "qformer tokenizer name"})
    vision_name_or_path: str = field(
        default="",
        metadata={"help": "The type of text model to use (OPT, T5)."},
    )

def create_model(model_args, training_args=None):
    blip2_config = Blip2Config.from_pretrained(model_args.model_name_or_path)
    blip2_config.train_mode = model_args.train_mode
    print("blip2_config:", blip2_config)
    model = Blip2ForConditionalGeneration(blip2_config)
    return model


def main():
    parser = PdArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()
    image = Image.open("/home/lizhijun/PaddleMIX-develop/paddlemix/demo_images/examples_image1.jpg")

    # 加载处理器
    prompt = data_args.prompt
    tokenizer_class = create_tokenizer(model_args.text_model_name_or_path)
    image_processor = BlipImageProcessor.from_pretrained(
        os.path.join(model_args.model_name_or_path, "processor", "eval")
    )
    text_processor_class = BlipTextProcessor.from_pretrained(
        os.path.join(model_args.model_name_or_path, "processor", "eval")
    )
    text_processor_class.prompt = ""
    processor = Blip2Processor(image_processor, text_processor_class, tokenizer_class)

    # 处理输入
    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pd",
        return_attention_mask=False,
        mode="eval",
    )

    # 加载模型
    model = create_model(model_args)
    model.eval()
    
    # 获取 ITM 阶段 logits
    with paddle.no_grad():
        pixel_values = inputs["pixel_values"]
        text_input_stage1 = prompt  # 文本输入直接传递到 ITM 阶段
        print("text_input_stage1:", text_input_stage1)

        # 调用 Stage1 模型，直接获取匹配概率
        matching_probability = model.inference_stage1(pixel_values=pixel_values, text_input=text_input_stage1)



if __name__ == "__main__":
    main()