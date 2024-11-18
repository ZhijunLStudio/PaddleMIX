from diffusers import StableDiffusionPipeline
import torch

# 设置模型保存的本地路径
local_model_path = "./stable-diffusion-3-medium"

# 从Hugging Face下载模型
pipeline = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    torch_dtype=torch.float16,  # 使用float16以节省内存
    use_safetensors=True,  # 使用更安全的safetensors格式
)

# 将模型保存到本地
pipeline.save_pretrained(local_model_path)