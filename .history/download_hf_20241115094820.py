from ppdiffusers import StableDiffusion3Pipeline
import paddle

# 指定本地目录
local_dir = "./stable-diffusion-3-model"

# 从 Hugging Face 下载模型
pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    paddle_dtype=paddle.float16,
    cache_dir=local_dir  # 设置缓存目录为指定的本地目录
)

print("模型下载完成，并已存储在:", local_dir)
