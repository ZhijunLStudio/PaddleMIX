# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"

# import paddle
# from ppdiffusers import StableDiffusion3Pipeline
# pipe = StableDiffusion3Pipeline.from_pretrained(
#     "stabilityai/stable-diffusion-3-medium-diffusers", paddle_dtype=paddle.float16
# )
# print("Loading model done")
# generator = paddle.Generator().manual_seed(42)
# prompt = "A cat holding a sign that says hello world"
# image = pipe(prompt, height=512, width=512, generator=generator).images[0]
# image.save("text_to_image_generation-stable_diffusion_3-result.png")


import os.
import paddle
from paddle.distributed import init_parallel_env
from ppdiffusers import StableDiffusion3Pipeline

# 初始化 Paddle 的多卡分布式环境
init_parallel_env()

# 设置模型在多卡上运行
pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers", 
    paddle_dtype=paddle.float16
)
pipe.to("gpu")  # 确保模型被放置在 GPU 上

# 打印加载完成消息
print("Loading model done")

# 设置随机种子和生成器
generator = paddle.Generator().manual_seed(42)

# 定义输入 Prompt
prompt = "A cat holding a sign that says hello world"

# 使用多卡并行生成图像
# 多卡中生成器通常由主卡管理，确保一致性
if paddle.distributed.get_rank() == 0:  # 确保只在主进程保存
    image = pipe(prompt, height=512, width=512, generator=generator).images[0]
    image.save("text_to_image_generation-stable_diffusion_3-result.png")
