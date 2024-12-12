import os
import json
import paddle
from paddlenlp.generation import TextStreamer

from paddlemix.auto import (
    AutoConfigMIX,
    AutoModelMIX,
    AutoProcessorMIX,
    AutoTokenizerMIX,
)
from paddlemix.models.llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
)
from paddlemix.models.llava.conversation import conv_templates
from paddlemix.models.llava.mm_utils import load_image

from paddlemix.utils.log import logger


def main():
    paddle.seed(seed=0)

    # 模型路径列表
    model_paths = [
        "checkpoints/infer_ckpt/llava-v1.5-7b",
        "checkpoints/infer_ckpt/llava_sft_lora_merge_1210_origin",
        "checkpoints/infer_ckpt/llava_sft_lora_merge_1210_percentile"
    ]

    # 固定问题和图片路径
    questions = [
        "Can you describe the main content of this image?",
        "图片中是否有可见的文字？如果有，文字内容是什么？",
        # "图像中的主要人物或物体在做什么？"
    ]
    image_files = [
        "datasets/mmmu/select_img/dev_Physics_4_1.png",
        # "datasets/mmmu/select_img/dev_Diagnostics_and_Laboratory_Medicine_4_1.png",
        # "datasets/mmmu/select_img/dev_Manage_3_1.png",
        "datasets/mmmu/select_img/dev_History_4_1.png"
    ]

    # 参数组合
    temperatures = [0.2]
    max_new_tokens_list = [512]
    fp16_options = [True]


    # 输出json路径
    json_output_dir = "model_results"
    if not os.path.exists(json_output_dir):
        os.makedirs(json_output_dir)

    # 遍历模型路径
    for model_path in model_paths:
        model_name = os.path.basename(model_path)  # 提取路径最后一级名称
        results = []

        for fp16 in fp16_options:
            compute_dtype = "float16" if fp16 else "bfloat16"
            if "npu" in paddle.get_device():
                is_bfloat16_supported = True
            else:
                is_bfloat16_supported = paddle.amp.is_bfloat16_supported()
            if compute_dtype == "bfloat16" and not is_bfloat16_supported:
                logger.warning("bfloat16 is not supported on your device, changing to float32")
                compute_dtype = "float32"

            # 加载模型、配置和处理器
            try:
                tokenizer = AutoTokenizerMIX.from_pretrained(model_path)
                model_config = AutoConfigMIX.from_pretrained(model_path)
                model = AutoModelMIX.from_pretrained(model_path, dtype=compute_dtype)
                model.eval()
                processor, _ = AutoProcessorMIX.from_pretrained(
                    model_path,
                    eval="eval",
                    max_length=max(max_new_tokens_list),
                    image_aspect_ratio=model_config.image_aspect_ratio,
                )
                model.resize_token_embeddings(len(tokenizer))
                vision_tower = model.get_vision_tower()
                vision_tower.load_model()
                logger.info(f"Model and processor loaded successfully from {model_path} with fp16={fp16}")
            except Exception as e:
                logger.error(f"Error loading model or processor from {model_path}: {e}")
                continue

            for image_file in image_files:
                logger.info(f"Processing image: {image_file}")
                image_results = {"image": image_file, "results": []}

                # 图像预处理（只处理一次，复用结果）
                try:
                    image_size = load_image(image_file).size
                    logger.info(f"Image size: {image_size}")

                    record = {"image": image_file, "conversations": ""}
                    processed_data = processor(record=record, image_aspect_ratio=model_config.image_aspect_ratio)
                    logger.info(f"Processed data shape: images={processed_data['images'].shape}")
                except Exception as e:
                    logger.error(f"Error processing image: {e}")
                    continue

                for question in questions:
                    logger.info(f"Processing question: {question}")
                    for temperature in temperatures:
                        for max_new_tokens in max_new_tokens_list:
                            try:
                                logger.info(
                                    f"Running with temperature={temperature}, "
                                    f"max_new_tokens={max_new_tokens}, fp16={fp16}"
                                )

                                # 使用独立对话模板，避免叠加问题
                                conv_mode = "llava_v0"
                                conv = conv_templates[conv_mode].copy()

                                inp = DEFAULT_IMAGE_TOKEN + "\n" + question
                                conv.append_message(conv.roles[0], inp)
                                conv.append_message(conv.roles[1], None)
                                prompt = conv.get_prompt()

                                # 更新输入
                                processed_data["input_ids"] = processor(
                                    {"image": image_file, "conversations": prompt},
                                    image_aspect_ratio=model_config.image_aspect_ratio,
                                )["input_ids"]

                                logger.info(f"Prompt: {prompt}")
                                logger.info(f"Input shape: {processed_data['input_ids'].shape}")

                                streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

                                # 文本生成
                                with paddle.no_grad():
                                    output_ids = model.generate(
                                        input_ids=processed_data["input_ids"],
                                        images=paddle.cast(processed_data["images"], compute_dtype),
                                        image_sizes=[image_size],
                                        decode_strategy="sampling" if temperature > 0 else "greedy_search",
                                        temperature=temperature,
                                        max_new_tokens=max_new_tokens,
                                        streamer=streamer,
                                        use_cache=False,  # 避免缓存带来的问题
                                    )
                                outputs = tokenizer.decode(output_ids[0][0]).strip().split("<|im_end|>")[0].split("</s>")[0]
                                logger.info(f"Generated response: {outputs}")

                                image_results["results"].append({
                                    "question": question,
                                    "temperature": temperature,
                                    "max_new_tokens": max_new_tokens,
                                    "fp16": fp16,
                                    "response": outputs
                                })
                            except Exception as e:
                                logger.error(f"Error during generation: {e}")
                                continue

                results.append(image_results)

        # 保存结果到 JSON 文件
        output_file = os.path.join(json_output_dir, f"{model_name}.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        logger.info(f"Results for model {model_name} saved to {output_file}")


if __name__ == "__main__":
    main()