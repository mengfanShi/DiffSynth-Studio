from examples.Diffutoon.diffutoon_toon_shading import config
from diffsynth import SDVideoPipelineRunner, SDImagePipelineRunner
from diffsynth.extensions.CUGAN.inference_video import VideoRealWaifuUpScaler
import argparse
import datetime
import os
import cv2


def parse_args():
    parser = argparse.ArgumentParser(description="Run Video Style Transfer with specified parameters.")

    parser.add_argument("--video", type=str, required=True, help="Path to the input video file.")
    parser.add_argument("--prompt", type=str, default="masterpiece", help="Prompt for the pipeline.")
    parser.add_argument("--height", type=int, default=1024, help="Height of the input video frames, multiple of 64.")
    parser.add_argument("--width", type=int, default=1024, help="Width of the input video frames, multiple of 64.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the pipeline on (e.g., 'cuda', 'cpu').")
    parser.add_argument("--start_frame_id", type=int, default=0, help="Starting frame ID for processing.")
    parser.add_argument("--end_frame_id", type=int, default=30, help="Ending frame ID for processing.")
    parser.add_argument("--total_video", action="store_true", help="whether process total video.")
    parser.add_argument("--origin_size", action="store_true", help="whether output origin size of input video.")
    parser.add_argument("--output_folder", type=str, default="data/toon_video", help="Folder to save the output frames.")
    parser.add_argument("--steps", type=int, default=10, help="Number of inference steps.")
    parser.add_argument("--seed", type=int, default=0, help="seed of generate video.")
    parser.add_argument("--cfg_scale", type=float, default=7.0, help="cfg_scale.")
    parser.add_argument("--fps", type=float, default=None, help="fps of generate video.")
    parser.add_argument("--model_path", type=str, default="models/stable_diffusion/aingdiffusion_v16.safetensors",
                        help="Stable Diffusion model path.")
    parser.add_argument("--lora", type=str, nargs='+', help="Lora model path.")
    parser.add_argument("--lora_alpha", type=float, nargs='+', help="Lora alpha.")
    parser.add_argument("--animatediff", type=str, default="models/AnimateDiff/mm_sd_v15_v3.ckpt",
                        help="Animatediff model path.")
    parser.add_argument("--animatediff_size", type=int, default=16, help="Animatediff batch size.")
    parser.add_argument("--animatediff_stride", type=int, default=8, help="Animatediff stride.")
    parser.add_argument("--denoise", type=float, default=1, help="Denoising strength.")
    parser.add_argument("--super_model", type=str, default="models/CUGAN/pro-no-denoise-up2x.pth",
                        help="Super Resolution model path.")
    parser.add_argument("--upscaler_scale", type=int, default=2, help="Super Resolution scalser scale.")
    parser.add_argument("--upscale_input", action="store_true", help="whether upscale input video.")
    parser.add_argument("--upscale_output", action="store_true", help="whether upscale output video.")
    parser.add_argument("--use_rife", action="store_true", help="whether use rife smooth.")
    parser.add_argument("--rife_model", type=str, default="models/RIFE/flownet.pkl")
    parser.add_argument("--use_fastblend", action="store_true", help="whether use FastBlend smooth.")
    parser.add_argument("--interpolate", type=int, default=0, help="interpolate times param.")
    parser.add_argument("--trans_first_frame", action="store_true", help="whether transform input video's first frame.")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    time_str = datetime.datetime.now().strftime("-%Y-%m-%dT%H-%M-%S")
    demo_config = config.copy()
    args = parse_args()

    if os.path.isfile(args.video):
        basename, ext = os.path.splitext(os.path.basename(args.video))
        output_dir = os.path.join(args.output_folder, basename + time_str + f"-{args.steps}steps")
        video = args.video

        interpolate = args.use_rife and args.interpolate > 0
        if interpolate:
            os.makedirs(output_dir, exist_ok=True)
            cap = cv2.VideoCapture(video)
            out = cv2.VideoWriter(
                os.path.join(output_dir, "video_sample" + ext),
                cv2.VideoWriter_fourcc(*'mp4v'), cap.get(cv2.CAP_PROP_FPS) / (args.interpolate + 1),
                (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            )

            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_count % (args.interpolate + 1) == 0:
                    out.write(frame)
                frame_count += 1
            cap.release()
            out.release()
            video = os.path.join(output_dir, "video_sample" + ext)

        cap = cv2.VideoCapture(video)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        is_1080p = min(frame_width, frame_height) >= 1024

        start_frame_id = 0 if args.total_video else args.start_frame_id
        end_frame_id = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if args.total_video else args.end_frame_id
        if args.fps is None:
            fps = cap.get(cv2.CAP_PROP_FPS) * (args.interpolate + 1) if interpolate else cap.get(cv2.CAP_PROP_FPS)
        else:
            fps = args.fps
        cap.release()

        if args.super_model is not None and os.path.exists(args.super_model):
            video_upscaler = VideoRealWaifuUpScaler(
                scale=args.upscaler_scale, weight_path=args.super_model, device=args.device)

        # if args.upscale_input or not is_1080p:
        if args.upscale_input:
            video_upscaler.start()
            video_upscaler(video, output_dir, "upscaled_input")
            video = os.path.join(output_dir, "upscaled_input" + ext)
            frame_width *= args.upscaler_scale
            frame_height *= args.upscaler_scale

        if os.path.isfile(args.model_path):
            demo_config["models"]["model_list"][0] = args.model_path

        if os.path.isfile(args.animatediff):
            demo_config["models"]["model_list"][1] = args.animatediff

        if args.lora and args.lora_alpha:
            assert len(args.lora) == len(args.lora_alpha), "lora and lora_alpha should have same length"
            for lora, alpha in zip(args.lora, args.lora_alpha):
                demo_config["models"]["lora_list"].append(lora)
                demo_config["models"]["lora_alphas"].append(alpha)

        width = ((args.width + 63) // 64) * 64
        height = ((args.height + 63) // 64) * 64

        if args.origin_size:
            width = ((frame_width + 63) // 64) * 64
            height = ((frame_height + 63) // 64) * 64

        demo_config["models"]["device"] = args.device
        demo_config["data"]["input_frames"] = {
            "video_file": video,
            "image_folder": None,
            "height": height,
            "width": width,
            "start_frame_id": start_frame_id,
            "end_frame_id": end_frame_id
        }
        demo_config["data"]["controlnet_frames"] = [demo_config["data"]["input_frames"], demo_config["data"]["input_frames"]]
        demo_config["data"]["output_folder"] = output_dir
        demo_config["data"]["fps"] = fps
        demo_config["pipeline"]["seed"] = args.seed
        demo_config["pipeline"]["pipeline_inputs"]["prompt"] = args.prompt
        demo_config["pipeline"]["pipeline_inputs"]["num_inference_steps"] = args.steps
        demo_config["pipeline"]["pipeline_inputs"]["cfg_scale"] = args.cfg_scale
        demo_config["pipeline"]["pipeline_inputs"]["animatediff_batch_size"] = args.animatediff_size
        demo_config["pipeline"]["pipeline_inputs"]["animatediff_stride"] = args.animatediff_stride
        demo_config["pipeline"]["pipeline_inputs"]["denoising_strength"] = args.denoise

        if args.use_rife and os.path.isfile(args.rife_model):
            demo_config["models"]["model_list"].append(args.rife_model)
            demo_config["smoother_configs"].append(
                {"processor_type": "RIFE", "config": {"interpolate": interpolate, "interpolate_times": args.interpolate}})

        if args.use_fastblend:
            demo_config["smoother_configs"].append({"processor_type": "FastBlend", "config": {}})

        if args.trans_first_frame:
            image_config = {}
            image_config["models"] = demo_config["models"]
            del image_config["models"]["model_list"][1]
            image_config["data"] = {}
            image_config["pipeline"] = {}
            image_config["pipeline"]["pipeline_inputs"] = {}
            image_config["data"]["input_frames"] = {
                "video_file": video,
                "image_folder": None,
                "height": height,
                "width": width,
            }
            image_config["data"]["output_folder"] = output_dir
            image_config["pipeline"]["seed"] = args.seed
            image_config["pipeline"]["pipeline_inputs"]["prompt"] = args.prompt
            image_config["pipeline"]["pipeline_inputs"]["negative_prompt"] = "verybadimagenegative_v1.3"
            image_config["pipeline"]["pipeline_inputs"]["num_inference_steps"] = args.steps
            image_config["pipeline"]["pipeline_inputs"]["cfg_scale"] = args.cfg_scale
            image_config["pipeline"]["pipeline_inputs"]["denoising_strength"] = args.denoise

            print(image_config)
            image_runner = SDImagePipelineRunner()
            image_gen = image_runner.run(image_config)

            if args.upscale_output:
                image_super = video_upscaler.process_image(image_gen)
                image_super.save(os.path.join(output_dir, "super_image.png"))
        else:
            print(demo_config)
            runner = SDVideoPipelineRunner()
            runner.run(demo_config)

            if args.upscale_output:
                video_name = "video.mp4"
                video_upscaler.start()
                video_upscaler(os.path.join(output_dir, video_name), output_dir)








