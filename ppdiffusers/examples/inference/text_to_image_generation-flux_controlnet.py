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


import paddle
from ppdiffusers import FluxControlNetModel
from ppdiffusers.pipelines import FluxControlNetPipeline
from ppdiffusers.utils import load_image

# load pipeline
controlnet = FluxControlNetModel.from_pretrained("/data/home/lizhijun/llm/flux-hf/models/FLUX.1-dev-Controlnet-Canny-pd", paddle_dtype=paddle.float16)
pipe = FluxControlNetPipeline.from_pretrained(
    "/data/home/lizhijun/llm/flux-hf/models/flux-dev-pd", controlnet=controlnet, paddle_dtype=paddle.float16
)


# download an image
control_image = load_image("https://huggingface.co/InstantX/SD3-Controlnet-Canny/resolve/main/canny.jpg") #FIXME

# generate image
generator = paddle.Generator().manual_seed(0)
prompt = "Anime style illustration of a girl wearing a suit. A moon in sky. In the background we see a big rain approaching. text 'InstantX' on image"
n_prompt = "NSFW, nude, naked, porn, ugly"
image = pipe(
    prompt, 
    negative_prompt=n_prompt, 
    control_image=control_image, 
    controlnet_conditioning_scale=0.5,
    generator=generator,
).images[0]
image.save("image_to_image_text_guided_generation-stable_diffusion_3_controlnet-result.png")