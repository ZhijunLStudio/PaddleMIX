from huggingface_hub import snapshot_download

# 指定模型和下载路径
model_name = "stabilityai/stable-diffusion-3-medium-diffusers"
local_dir = "./stable-diffusion-3-model"

# 下载模型快照到本地目录
snapshot_download(repo_id=model_name, local_dir=local_dir)

print("模型已下载到本地目录:", local_dir)
