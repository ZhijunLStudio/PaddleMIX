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


