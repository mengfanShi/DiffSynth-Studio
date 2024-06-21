from examples.Diffutoon.diffutoon_toon_shading import config
from diffsynth import SDVideoPipelineRunner
from diffsynth.extensions.CUGAN.inference_video import VideoRealWaifuUpScaler
import argparse
import datetime
import os
import cv2


def parse_args():
    parser = argparse.ArgumentParser(description="Run SDVideoPipelineRunner with specified parameters.")

    parser.add_argument("--video_path", type=str, required=True, help="Path to the input video file.")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt for the pipeline.")
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
    parser.add_argument("--fps", type=int, default=30, help="fps of generate video.")
    parser.add_argument("--model_path", type=str, default="models/stable_diffusion/aingdiffusion_v17.safetensors",
                        help="Stable Diffusion model path.")
    parser.add_argument("--animatediff", type=str, default="models/AnimateDiff/mm_sd_v15_v3.ckpt",
                        help="Animatediff model path.")
    parser.add_argument("--animatediff_size", type=int, default=16, help="Animatediff batch size.")
    parser.add_argument("--animatediff_stride", type=int, default=8, help="Animatediff stride.")
    parser.add_argument("--denoise", type=float, default=1, help="Denoising strength.")
    parser.add_argument("--super_model", type=str, default=None, help="Super Resolution model path.")
    parser.add_argument("--upscaler_scale", type=int, default=2, help="Super Resolution scalser scale.")
    parser.add_argument("--upscale_input", action="store_true", help="whether upscale input video.")
    parser.add_argument("--upscale_output", action="store_true", help="whether upscale output video.")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    time_str = datetime.datetime.now().strftime("-%Y-%m-%dT%H-%M-%S")
    demo_config = config.copy()
    args = parse_args()

    if os.path.isfile(args.video_path):
        basename, ext = os.path.splitext(os.path.basename(args.video_path))
        output_dir = os.path.join(args.output_folder, basename + time_str)

        if args.super_model is not None and os.path.exists(args.super_model):
            video_upscaler = VideoRealWaifuUpScaler(
                scale=args.upscaler_scale, weight_path=args.super_model, device=args.device)

        if args.upscale_input:
            video_upscaler.start()
            video_upscaler(args.video_path, output_dir, "upscaled_input")
            args.video_path = os.path.join(output_dir, "upscaled_input" + ext)

        if os.path.isfile(args.model_path):
            demo_config["models"]["model_list"][0] = args.model_path

        if os.path.isfile(args.animatediff):
            demo_config["models"]["model_list"][1] = args.animatediff

        start_frame_id = args.start_frame_id
        end_frame_id = args.end_frame_id
        fps = args.fps
        if args.total_video:
            cap = cv2.VideoCapture(args.video_path)
            start_frame_id = 0
            end_frame_id = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            cap.release()

        width = (args.width // 64) * 64
        height = (args.height // 64) * 64
        if args.origin_size:
            width = (frame_width // 64) * 64
            height = (frame_height // 64) * 64

        demo_config["models"]["device"] = args.device
        demo_config["data"]["input_frames"] = {
            "video_file": args.video_path,
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

        print(demo_config)

        runner = SDVideoPipelineRunner()
        runner.run(demo_config)

        if args.upscale_output:
            video_upscaler.start()
            video_upscaler(os.path.join(output_dir, "video.mp4"), output_dir)





