import os
import json
import argparse
from datetime import datetime
from PIL import Image
import torch
from diffsynth import ModelManager, WanVideoPipeline, save_video
import numpy as np
import torch.distributed as dist


GLOBAL_NEG = "色调艳丽,过曝,静态,细节模糊不清,字幕,风格,作品,画作,画面,静止,整体发灰,最差质量,低质量,JPEG压缩残留,丑陋的,残缺的,多余的手指,画得不好的手部,画得不好的脸部,畸形的,毁容的,形态畸形的肢体,手指融合,静止不动的画面,杂乱的背景,三条腿,背景人很多,倒着走"
SR_SIZE_MAP = {"720P": (1280, 720), "1080P": (1920, 1080)}


def expand_paths(input_paths):
    expanded = []
    if input_paths is None:
        return None
    for path in input_paths:
        if os.path.isdir(path):
            for f in os.listdir(path):
                if f.endswith(".safetensors"):
                    expanded.append(os.path.join(path, f))
        else:
            expanded.append(path)
    return expanded if len(expanded) > 0 else None


def prepare_inputs(prompt, image, task):
    inputs = []

    # 处理JSON输入
    if os.path.isfile(prompt) and prompt.endswith(".json"):
        assert os.path.isdir(image), "JSON input requires a directory for images"
        with open(prompt, "r", encoding="utf-8") as f:
            json_data = json.load(f)
            for item in json_data:
                image_name = item.get("image", "")
                prompt = item.get("prompt", "")
                if image_name and prompt:
                    inputs.append(
                        {"image": os.path.join(image, image_name), "prompt": prompt}
                    )
        return inputs

    # 处理原有的TXT输入或单个prompt
    if os.path.isfile(prompt) and prompt.endswith(".txt"):
        with open(prompt, "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f.readlines() if line.strip()]
    else:
        prompts = [prompt]

    # 处理图像输入
    if task == "i2v":
        images = []
        if os.path.isdir(image):
            for file in os.listdir(image):
                if file.endswith((".jpg", ".png", ".jpeg", ".webp", ".bmp")):
                    images.append(os.path.join(image, file))
        elif os.path.isfile(image):
            images.append(image)

        for image_path in images:
            for prompt in prompts:
                inputs.append({"image": image_path, "prompt": prompt})
    else:
        for prompt in prompts:
            inputs.append({"prompt": prompt})

    return inputs


def main():
    parser = argparse.ArgumentParser(description="Wan Video Generation Pipeline")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        nargs="+",
        default="models/Wan-AI/Wan2.1-I2V-14B-720P/diffusion_pytorch_model.safetensors",
        help="Path to main model checkpoint",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        nargs="+",
        default=None,
        help="Path to LoRA checkpoint (optional)",
    )
    parser.add_argument(
        "--t5_model",
        type=str,
        default="models/Wan-AI/Wan2.1-I2V-14B-720P/models_t5_umt5-xxl-enc-bf16.pth",
        help="Path to T5 model",
    )
    parser.add_argument(
        "--vae_model",
        type=str,
        default="models/Wan-AI/Wan2.1-I2V-14B-720P/Wan2.1_VAE.pth",
        help="Path to VAE model",
    )
    parser.add_argument(
        "--clip_model",
        type=str,
        default="models/Wan-AI/Wan2.1-I2V-14B-720P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        help="Path to CLIP model",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Input text prompt for video generation or path to a txt file containing prompts",
    )
    parser.add_argument(
        "--negative_prompt", type=str, default=None, help="Negative prompt for guidance"
    )
    parser.add_argument(
        "--output", type=str, default="output", help="Path to save generated video"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--steps", type=int, default=40, help="Number of inference steps"
    )
    parser.add_argument(
        "--lora_alpha", type=float, default=1.0, help="LoRA alpha value"
    )
    parser.add_argument(
        "--height", type=int, default=480, help="Height of the generated video"
    )
    parser.add_argument(
        "--width", type=int, default=832, help="Width of the generated video"
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=81,
        help="Number of frames in the generated video",
    )
    parser.add_argument(
        "--fps", type=int, default=30, help="Frames per second of the generated video"
    )
    parser.add_argument(
        "--task", type=str, default="t2v", choices=["t2v", "i2v"], help="Task type"
    )
    parser.add_argument(
        "--tea_cache_thresh", type=int, default=None, help="Threshold for Tea cache"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="Wan2.1-I2V-14B-720P",
        help="WanX Model ID",
        choices=[
            "Wan2.1-T2V-1.3B",
            "Wan2.1-T2V-14B",
            "Wan2.1-I2V-14B-480P",
            "Wan2.1-I2V-14B-720P",
        ],
    )
    parser.add_argument("--image", type=str, default=None, help="Path to input image")
    parser.add_argument(
        "--rife_model",
        type=str,
        default="Scripts/RIFE/flownet.pkl",
        help="Path to RIFE model",
    )
    parser.add_argument(
        "--use_rife", action="store_true", help="Use RIFE for video smoothing"
    )
    parser.add_argument(
        "--cugan_model", type=str, default="Scripts/CUGAN/pro-conservative-up2x.pth"
    )
    parser.add_argument(
        "--use_cugan",
        action="store_true",
        help="Use CuGan for video resolution enhancement",
    )
    parser.add_argument(
        "--sr_size",
        type=str,
        default="1080P",
        choices=["720P", "1080P"],
        help="Super resolution size",
    )
    parser.add_argument(
        "--export_dit_path", type=str, default=None, help="export dit path"
    )
    parser.add_argument(
        "--use_fp8", default=False, action="store_true", help="Use FP8 for inference"
    )
    parser.add_argument(
        "--num_persistent_param_in_dit",
        type=int,
        default=None,
        help="Number of persistent parameters in DIT, smaller number means less VRAM usage",
        choices=[None, 0, 6 * 10**9, 7 * 10**9],
    )
    parser.add_argument(
        "--prompt_extend_model",
        type=str,
        default=None,
        help="Path to prompt extend model",
    )
    args = parser.parse_args()

    if args.negative_prompt is not None:
        neg_prompt = GLOBAL_NEG + "," + args.negative_prompt
    else:
        neg_prompt = GLOBAL_NEG

    args.ckpt_path = expand_paths(args.ckpt_path) or args.ckpt_path
    args.lora_path = expand_paths(args.lora_path) or args.lora_path

    model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
    models = [args.ckpt_path, args.t5_model, args.vae_model]
    model_manager.load_models(
        models, torch_dtype=torch.float8_e4m3fn if args.use_fp8 else torch.bfloat16
    )
    if args.task == "i2v":
        model_manager.load_models([args.clip_model], torch_dtype=torch.bfloat16)

    if args.lora_path is not None:
        model_manager.load_lora(args.lora_path, lora_alpha=args.lora_alpha)

    if args.export_dit_path is not None:
        os.makedirs(args.export_dit_path, exist_ok=True)
        dit_model = model_manager.fetch_model("wan_video_dit")
        try:
            from safetensors.torch import save_file

            save_path = os.path.join(args.export_dit_path, "wan_video_dit.safetensors")
            save_file(dit_model.state_dict(), save_path)
        except ImportError:
            print("safetensors not found, exporting DIT model as PyTorch")
            save_path = os.path.join(args.export_dit_path, "wan_video_dit.pth")
            torch.save(dit_model.state_dict(), save_path)
        print(f"Exported DIT model to: {save_path}")
        return

    smoother = None
    if args.use_rife and args.rife_model is not None:
        try:
            from RIFE import RIFESmoother

            smoother = RIFESmoother(
                model=args.rife_model,
                device="cuda",
                interpolate=True,
                interpolate_times=1,
            )
            print("Using RIFE for video smoothing")
        except Exception as e:
            print(f"Failed to load RIFE model: {e}")

    sr_model = None
    if args.use_cugan and args.cugan_model is not None:
        try:
            from CUGAN import VideoRealWaifuUpScaler

            sr_model = VideoRealWaifuUpScaler(
                scale=2, weight_path=args.cugan_model, device="cuda"
            )
            print("Using CuGan for video resolution enhancement")
        except Exception as e:
            print(f"Failed to load CuGan model: {e}")

    try:
        from xfuser.core.distributed import (
            initialize_model_parallel,
            init_distributed_environment,
        )

        dist.init_process_group(backend="nccl", init_method="env://")
        init_distributed_environment(
            rank=dist.get_rank(), world_size=dist.get_world_size()
        )
        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=1,
            ulysses_degree=dist.get_world_size(),
        )
        torch.cuda.set_device(dist.get_rank())
        pipe = WanVideoPipeline.from_model_manager(
            model_manager,
            torch_dtype=torch.bfloat16,
            device=f"cuda:{dist.get_rank()}",
            use_usp=True if dist.get_world_size() > 1 else False,
        )
        print("Unified sequence parallel enabled")
        use_usp = True
    except Exception as e:
        use_usp = False
        print(f"Failed to initialize unified sequence parallel: {e}")
        pipe = WanVideoPipeline.from_model_manager(
            model_manager, torch_dtype=torch.bfloat16, device="cuda"
        )

    prompt_expander = None
    if args.prompt_extend_model is not None:
        try:
            from Prompt_extend import QwenPromptExpander, VL_EN_DIGITAL_HUMAN_PROMPT

            if use_usp:
                prompt_expander = QwenPromptExpander(
                    model_name=args.prompt_extend_model,
                    is_vl="i2v" in args.task,
                    device=dist.get_rank(),
                )
            else:
                prompt_expander = QwenPromptExpander(
                    model_name=args.prompt_extend_model,
                    is_vl="i2v" in args.task,
                    device="cuda",
                )
            print(f"Using Prompt extend model from {args.prompt_extend_model}")
        except Exception as e:
            print(f"Failed to load prompt extend model: {e}")

    pipe.enable_vram_management(
        num_persistent_param_in_dit=args.num_persistent_param_in_dit
    )

    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    # 读取输入信息
    inputs = prepare_inputs(args.prompt, args.image, args.task)

    # 生成视频
    for input_item in inputs:
        single_prompt = input_item["prompt"]
        print(f"Generating video for prompt: {single_prompt}")

        if args.task == "i2v":
            max_area = args.width * args.height
            image_path = input_item["image"]

            image = Image.open(image_path)
            if image.mode != "RGB":
                image = image.convert("RGB")

            main_process = (not use_usp) or (dist.get_rank() == 0)

            if prompt_expander is not None:
                expander_kwargs = {
                    "prompt": single_prompt,
                    "system_prompt": VL_EN_DIGITAL_HUMAN_PROMPT,
                    "tar_lang": "en",
                    "image": image,
                    "seed": args.seed,
                }

                if main_process:
                    prompt_output = prompt_expander(**expander_kwargs)
                    if prompt_output.status:
                        single_prompt = prompt_output.prompt
                        single_prompt = single_prompt.strip().replace(
                            "\n", " "
                        )  # 去除换行符
                        print(f"Prompt extended: {single_prompt}")

                # USP 模式需要广播结果
                if use_usp and dist.is_initialized():
                    if not isinstance(single_prompt, list):
                        single_prompt = [single_prompt]
                    dist.broadcast_object_list(single_prompt, src=0)
                    single_prompt = single_prompt[0]

            width, height = image.size
            aspect_ratio = height / width
            new_height = round(np.sqrt(max_area * aspect_ratio))
            new_width = round(np.sqrt(max_area / aspect_ratio))
            image = image.resize((new_width, new_height))

            # Image-to-video
            video = pipe(
                prompt=single_prompt,
                negative_prompt=neg_prompt,
                input_image=image,
                num_inference_steps=args.steps,
                height=new_height,
                width=new_width,
                num_frames=args.num_frames,
                seed=args.seed,
                tiled=True,
                tea_cache_l1_thresh=args.tea_cache_thresh,
                tea_cache_model_id=args.model_id,
            )

            if main_process:
                image_basename = os.path.splitext(os.path.basename(image_path))[0]
                formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                formatted_prompt = single_prompt.replace(" ", "_").replace("/", "_")[
                    :50
                ]
                path = os.path.join(
                    output_dir,
                    f"{image_basename}_{formatted_time}_{formatted_prompt}.mp4",
                )
                save_video(video, path, fps=args.fps, quality=5)

                if smoother is not None:
                    video = smoother(video)
                    path = os.path.join(
                        output_dir,
                        f"inter_{image_basename}_{formatted_time}_{formatted_prompt}.mp4",
                    )
                    save_video(video, path, fps=args.fps, quality=5)

                if sr_model is not None:
                    size = SR_SIZE_MAP[args.sr_size]
                    area = size[0] * size[1]
                    sr_height = round(np.sqrt(area * aspect_ratio))
                    sr_width = round(np.sqrt(area / aspect_ratio))
                    sr_model.process_video(
                        path,
                        output_dir,
                        out_name=f"sr_{image_basename}_{formatted_time}_{formatted_prompt}",
                        out_width=sr_width,
                        out_height=sr_height,
                    )

        elif args.task == "t2v":
            main_process = (not use_usp) or (dist.get_rank() == 0)

            if prompt_expander is not None:
                expander_kwargs = {
                    "prompt": single_prompt,
                    "tar_lang": "en",
                    "seed": args.seed,
                }

                if main_process:
                    prompt_output = prompt_expander(**expander_kwargs)
                    if prompt_output.status:
                        single_prompt = prompt_output.prompt
                        single_prompt = single_prompt.strip().replace(
                            "\n", " "
                        )  # 去除换行符
                        print(f"Prompt extended: {single_prompt}")

                # USP 模式需要广播结果
                if use_usp and dist.is_initialized():
                    if not isinstance(single_prompt, list):
                        single_prompt = [single_prompt]
                    dist.broadcast_object_list(single_prompt, src=0)
                    single_prompt = single_prompt[0]

            video = pipe(
                prompt=single_prompt,
                negative_prompt=neg_prompt,
                num_inference_steps=args.steps,
                seed=args.seed,
                tiled=True,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                tea_cache_l1_thresh=args.tea_cache_thresh,
                tea_cache_model_id=args.model_id,
            )

            if main_process:
                formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                formatted_prompt = single_prompt.replace(" ", "_").replace("/", "_")[
                    :50
                ]
                path = os.path.join(
                    output_dir, f"{formatted_time}_{formatted_prompt}.mp4"
                )
                save_video(video, path, fps=args.fps, quality=5)

                if smoother is not None:
                    video = smoother(video)
                    path = os.path.join(
                        output_dir,
                        f"inter_{image_basename}_{formatted_time}_{formatted_prompt}.mp4",
                    )
                    save_video(video, path, fps=args.fps, quality=5)

                if sr_model is not None:
                    size = SR_SIZE_MAP[args.sr_size]
                    area = size[0] * size[1]
                    aspect_ratio = args.height / args.width
                    sr_height = round(np.sqrt(area * aspect_ratio))
                    sr_width = round(np.sqrt(area / aspect_ratio))
                    sr_model.process_video(
                        path,
                        output_dir,
                        out_name=f"sr_{formatted_time}_{formatted_prompt}",
                        out_width=sr_width,
                        out_height=sr_height,
                    )


if __name__ == "__main__":
    main()
