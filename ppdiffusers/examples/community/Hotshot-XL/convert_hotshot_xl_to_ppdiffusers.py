# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import argparse
import os
import sys
import tempfile
from collections import OrderedDict

sys.path.append(".")
import paddle
import torch
from hf_hotshot_xl.pipelines.hotshot_xl_pipeline import (
    HotshotXLPipeline as DiffusersStableDiffusionPipeline,
)

from ppdiffusers import AutoencoderKL
from ppdiffusers.configuration_utils import FrozenDict
from ppdiffusers.models.hotshot_xl.unet import UNet3DConditionModel
from ppdiffusers.pipelines.hotshot_xl.hotshot_xl_pipeline import (
    HotshotXLPipeline as PPDiffusersStableDiffusionPipeline,
)
from ppdiffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from ppdiffusers.schedulers.scheduling_euler_ancestral_discrete import (
    EulerAncestralDiscreteScheduler,
)
from ppdiffusers.transformers import (
    CLIPFeatureExtractor,
    CLIPTextConfig,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionConfig,
)

paddle.set_device("cpu")


def convert_to_ppdiffusers(vae_or_unet, dtype=None):
    need_transpose = []
    for k, v in vae_or_unet.named_modules():
        if isinstance(v, torch.nn.Linear):
            need_transpose.append(k + ".weight")
    new_vae_or_unet = OrderedDict()
    for k, v in vae_or_unet.state_dict().items():
        if k not in need_transpose:
            new_vae_or_unet[k] = v.cpu().numpy()
            if dtype is not None:
                new_vae_or_unet[k] = new_vae_or_unet[k].astype(dtype)
        else:
            new_vae_or_unet[k] = v.t().cpu().numpy()
            if dtype is not None:
                new_vae_or_unet[k] = new_vae_or_unet[k].astype(dtype)
    return new_vae_or_unet


def convert_hf_clip_to_ppnlp_clip(clip, dtype=None, is_text_encoder=True):
    new_model_state = {}
    transformers2ppnlp = {}
    ignore_value = ["position_ids"]
    donot_transpose = ["embeddings", "norm", "concept_embeds", "special_care_embeds"]

    for name, value in clip.state_dict().items():
        # step1: ignore position_ids
        if any(i in name for i in ignore_value):
            continue
        # step2: transpose nn.Linear weight
        if value.ndim == 2 and not any(i in name for i in donot_transpose):
            value = value.t()
        # step3: hf_name -> ppnlp_name mapping
        for hf_name, ppnlp_name in transformers2ppnlp.items():
            name = name.replace(hf_name, ppnlp_name)
        # step4: 0d tensor -> 1d tensor
        if name == "logit_scale":
            value = value.reshape((1,))
        # step5: safety_checker need prefix "clip."
        if "vision_model" in name:
            name = "clip." + name
        new_model_state[name] = value.cpu().numpy()
        if dtype is not None:
            new_model_state[name] = new_model_state[name].astype(dtype)

    if is_text_encoder:
        new_config = {}
        """
        {
  "_name_or_path": "/home/ubuntu/models/Hotshot-XL",
  "architectures": [
    "CLIPTextModel"
  ],
  "attention_dropout": 0.0,
  "bos_token_id": 0,
  "dropout": 0.0,
  "eos_token_id": 2,
  "hidden_act": "quick_gelu",
  "hidden_size": 768,
  "initializer_factor": 1.0,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-05,
  "max_position_embeddings": 77,
  "model_type": "clip_text_model",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 1,
  "projection_dim": 768,
  "torch_dtype": "float32",
  "transformers_version": "4.33.3",
  "vocab_size": 49408
}
        """
        new_config.update(
            {
                "attention_dropout": clip.config.attention_dropout,
                "bos_token_id": clip.config.bos_token_id,
                "dropout": clip.config.dropout,
                "eos_token_id": clip.config.eos_token_id,
                "hidden_act": clip.config.hidden_act,
                "hidden_size": clip.config.hidden_size,
                "initializer_factor": clip.config.initializer_factor,
                "initializer_range": clip.config.initializer_range,
                "intermediate_size": clip.config.intermediate_size,
                "max_position_embeddings": clip.config.max_position_embeddings,
                "num_attention_heads": clip.config.num_attention_heads,
                "num_hidden_layers": clip.config.num_hidden_layers,
                "projection_dim": clip.config.projection_dim,
                "vocab_size": clip.config.vocab_size,
            }
        )
        new_config.update(
            {
                "paddle_dtype": str(clip.config.torch_dtype).replace("torch.", ""),
            }
        )
    else:
        new_config = {
            "image_resolution": clip.config.vision_config.image_size,
            "vision_layers": clip.config.vision_config.num_hidden_layers,
            "vision_heads": clip.config.vision_config.num_attention_heads,
            "vision_embed_dim": clip.config.vision_config.hidden_size,
            "vision_patch_size": clip.config.vision_config.patch_size,
            "vision_mlp_ratio": clip.config.vision_config.intermediate_size // clip.config.vision_config.hidden_size,
            "vision_hidden_act": clip.config.vision_config.hidden_act,
            "projection_dim": clip.config.projection_dim,
        }
    return new_model_state, new_config


def convert_diffusers_stable_diffusion_to_ppdiffusers(pretrained_model_name_or_path, output_path=None):
    # 0. load diffusers pipe and convert to ppdiffusers weights format
    diffusers_pipe = DiffusersStableDiffusionPipeline.from_pretrained(
        pretrained_model_name_or_path, use_auth_token=True
    )
    requires_safety_checker = getattr(diffusers_pipe, "requires_safety_checker", False)
    vae_state_dict = convert_to_ppdiffusers(diffusers_pipe.vae)
    unet_state_dict = convert_to_ppdiffusers(diffusers_pipe.unet)
    text_encoder_state_dict, text_encoder_config = convert_hf_clip_to_ppnlp_clip(
        diffusers_pipe.text_encoder, is_text_encoder=True
    )
    text_encoder_2_state_dict, text_encoder_2_config = convert_hf_clip_to_ppnlp_clip(
        diffusers_pipe.text_encoder_2, is_text_encoder=True
    )
    # 1. vae
    pp_vae = AutoencoderKL.from_config(diffusers_pipe.vae.config)

    pp_vae.set_state_dict(vae_state_dict)

    # 2. unet
    pp_unet = UNet3DConditionModel.from_config(diffusers_pipe.unet.config)

    pp_unet.set_state_dict(unet_state_dict)

    # 3. text_encoder
    pp_text_encoder = CLIPTextModel(CLIPTextConfig.from_dict(text_encoder_config))
    pp_text_encoder.set_state_dict(text_encoder_state_dict)
    pp_text_encoder_2 = CLIPTextModelWithProjection(CLIPTextConfig.from_dict(text_encoder_2_config))
    pp_text_encoder_2.set_state_dict(text_encoder_2_state_dict)

    # 4. scheduler
    beta_start = diffusers_pipe.scheduler.beta_start
    beta_end = diffusers_pipe.scheduler.beta_end
    scheduler_type = diffusers_pipe.scheduler._class_name.lower()
    if "euler" in scheduler_type:
        pp_scheduler = EulerAncestralDiscreteScheduler(
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule="scaled_linear",
            # Make sure the scheduler compatible with DDIM
            # clip_sample=False,
            # set_alpha_to_one=False,
            steps_offset=1,
            timestep_spacing="leading",
        )
    else:
        raise ValueError(f"Scheduler of type {scheduler_type} doesn't exist!")

    with tempfile.TemporaryDirectory() as tmpdirname:
        # 5. tokenizer
        diffusers_pipe.tokenizer.save_pretrained(tmpdirname)
        pp_tokenizer = CLIPTokenizer.from_pretrained(tmpdirname)

        tmpdirname2 = tmpdirname + "/token2"
        os.mkdir(tmpdirname2)
        diffusers_pipe.tokenizer_2.save_pretrained(tmpdirname2)
        pp_tokenizer_2 = CLIPTokenizer.from_pretrained(tmpdirname2)

        if requires_safety_checker:
            # 6. feature_extractor
            # diffusers_pipe.feature_extractor.save_pretrained(tmpdirname)
            pp_feature_extractor = CLIPFeatureExtractor.from_pretrained(
                "CompVis/stable-diffusion-v1-4/feature_extractor"
            )
            # 7. safety_checker
            safety_checker_state_dict, safety_checker_config = convert_hf_clip_to_ppnlp_clip(
                diffusers_pipe.safety_checker, is_text_encoder=False
            )
            pp_safety_checker = StableDiffusionSafetyChecker(CLIPVisionConfig.from_dict(safety_checker_config))
            pp_safety_checker.set_state_dict(safety_checker_state_dict)
            # 8. create ppdiffusers pipe
            paddle_pipe = PPDiffusersStableDiffusionPipeline(
                vae=pp_vae,
                text_encoder=pp_text_encoder,
                tokenizer=pp_tokenizer,
                text_encoder_2=pp_text_encoder_2,
                tokenizer_2=pp_tokenizer_2,
                unet=pp_unet,
                safety_checker=pp_safety_checker,
                feature_extractor=pp_feature_extractor,
                scheduler=pp_scheduler,
            )
        else:
            # 8. create ppdiffusers pipe
            paddle_pipe = PPDiffusersStableDiffusionPipeline(
                vae=pp_vae,
                text_encoder=pp_text_encoder,
                tokenizer=pp_tokenizer,
                text_encoder_2=pp_text_encoder_2,
                tokenizer_2=pp_tokenizer_2,
                unet=pp_unet,
                # safety_checker=None,
                # feature_extractor=None,
                scheduler=pp_scheduler,
                # requires_safety_checker=False,
            )
        if "runwayml/stable-diffusion-inpainting" in pretrained_model_name_or_path:
            _internal_dict = dict(paddle_pipe._internal_dict)
            if _internal_dict["_ppdiffusers_version"] == "0.0.0":
                _internal_dict.update({"_ppdiffusers_version": "0.6.0"})
            paddle_pipe._internal_dict = FrozenDict(_internal_dict)
        # 9. save_pretrained
        paddle_pipe.save_pretrained(output_path)
    return paddle_pipe


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pytorch model weights to Paddle model weights.")
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default="hotshotco/Hotshot-XL",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="hotshot_output",
        help="The model output path.",
    )
    args = parser.parse_args()
    ppdiffusers_pipe = convert_diffusers_stable_diffusion_to_ppdiffusers(args.pretrained_path, args.output_path)
