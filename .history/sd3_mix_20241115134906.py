import os

import argparse
import paddle
def parse_args():
    parser = argparse.ArgumentParser(
        description=" Use PaddleMIX to accelerate the Stable Diffusion3 image generation model."
    )
    parser.add_argument(
        "--benchmark",
        type=(lambda x: str(x).lower() in ["true", "1", "yes"]),
        default=False,
        help="if set to True, measure inference performance",
    )
    parser.add_argument(
        "--inference_optimize",
        type=(lambda x: str(x).lower() in ["true", "1", "yes"]),
        default=False,
        help="If set to True, all optimizations except Triton are enabled.",
    )
    parser.add_argument(
        "--inference_optimize_bp",
        type=(lambda x: str(x).lower() in ["true", "1", "yes"]),
        default=False,
        help="If set to True, batch parallel is enabled in DIT and dual-GPU acceleration is used.",
    )
    parser.add_argument("--height", type=int, default=512, help="Height of the generated image.")
    parser.add_argument("--width", type=int, default=512, help="Width of the generated image.")
    parser.add_argument("--num-inference-steps", type=int, default=50, help="Number of inference steps.")
    parser.add_argument("--dtype", type=str, default="float32", help="Inference data types.")

    return parser.parse_args()


args = parse_args()

if args.inference_optimize:
    os.environ["INFERENCE_OPTIMIZE"] = "True"
    os.environ["INFERENCE_OPTIMIZE_TRITON"] = "True"
if args.inference_optimize_bp:
    os.environ["INFERENCE_OPTIMIZE_BP"] = "True"
if args.dtype == "float32":
    inference_dtype = paddle.float32
elif args.dtype == "float16":
    inference_dtype = paddle.float16


if args.inference_optimize_bp:
    from paddle.distributed import fleet
    from paddle.distributed.fleet.utils import recompute
    import numpy as np
    import random
    import paddle.distributed as dist
    import paddle.distributed.fleet as fleet
    strategy = fleet.DistributedStrategy()
    model_parallel_size = 2
    data_parallel_size = 1
    strategy.hybrid_configs = {
    "dp_degree": data_parallel_size,
    "mp_degree": model_parallel_size,
    "pp_degree": 1
    }
    fleet.init(is_collective=True, strategy=strategy)
    hcg = fleet.get_hybrid_communicate_group()
    mp_id = hcg.get_model_parallel_rank()
    rank_id = dist.get_rank()

import datetime
from ppdiffusers import StableDiffusion3Pipeline


pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    paddle_dtype=inference_dtype,
)

pipe.transformer = paddle.incubate.jit.inference(
    pipe.transformer,
    save_model_dir="./tmp/sd3",
    enable_new_ir=True,
    cache_static_model=True,
    # V100环境下，需设置exp_enable_use_cutlass=False,
    exp_enable_use_cutlass=False,
    delete_pass_lists=["add_norm_fuse_pass"],
)

generator = paddle.Generator().manual_seed(42)
prompt = "A cat holding a sign that says hello world"


image = pipe(
    prompt, num_inference_steps=args.num_inference_steps, width=args.width, height=args.height, generator=generator
).images[0]

if args.benchmark:
    # warmup
    for i in range(3):
        image = pipe(
            prompt,
            num_inference_steps=args.num_inference_steps,
            width=args.width,
            height=args.height,
            generator=generator,
        ).images[0]

    repeat_times = 10
    sumtime = 0.0
    for i in range(repeat_times):
        paddle.device.synchronize()
        starttime = datetime.datetime.now()
        image = pipe(
            prompt,
            num_inference_steps=args.num_inference_steps,
            width=args.width,
            height=args.height,
            generator=generator,
        ).images[0]
        paddle.device.synchronize()
        endtime = datetime.datetime.now()
        duringtime = endtime - starttime
        duringtime = duringtime.seconds * 1000 + duringtime.microseconds / 1000.0
        sumtime += duringtime
        print("SD3 end to end time : ", duringtime, "ms")

    print("SD3 ave end to end time : ", sumtime / repeat_times, "ms")
    cuda_mem_after_used = paddle.device.cuda.max_memory_allocated() / (1024**3)
    print(f"Max used CUDA memory : {cuda_mem_after_used:.3f} GiB")

if args.inference_optimize_bp:
    if rank_id == 0:
        image.save("text_to_image_generation-stable_diffusion_3-result.png")
else:
    image.save("text_to_image_generation-stable_diffusion_3-result.png")