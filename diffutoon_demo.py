from examples.Diffutoon.diffutoon_toon_shading import config
from diffsynth import SDVideoPipelineRunner
import argparse
import datetime
import os
import cv2


def parse_args():
    parser = argparse.ArgumentParser(description="Run SDVideoPipelineRunner with specified parameters.")

    parser.add_argument("--video_path", type=str, required=True, help="Path to the input video file.")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt for the pipeline.")
    parser.add_argument("--height", type=int, default=1024, help="Height of the input video frames.")
    parser.add_argument("--width", type=int, default=1024, help="Width of the input video frames.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the pipeline on (e.g., 'cuda', 'cpu').")
    parser.add_argument("--start_frame_id", type=int, default=0, help="Starting frame ID for processing.")
    parser.add_argument("--end_frame_id", type=int, default=30, help="Ending frame ID for processing.")
    parser.add_argument("--total_video", action="store_true", help="whether process total video.")
    parser.add_argument("--output_folder", type=str, default="data/toon_video", help="Folder to save the output frames.")
    parser.add_argument("--steps", type=int, default=10, help="Number of inference steps.")
    parser.add_argument("--seed", type=int, default=0, help="seed of generate video.")
    parser.add_argument("--fps", type=int, default=30, help="fps of generate video.")
    parser.add_argument("--model_path", type=str, default="models/stable_diffusion/aingdiffusion_v12.safetensors",
                        help="Stable Diffusion model path.")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    time_str = datetime.datetime.now().strftime("-%Y-%m-%dT%H-%M-%S")
    demo_config = config.copy()
    args = parse_args()

    if os.path.isfile(args.video_path):
        basename = os.path.splitext(os.path.basename(args.video_path))[0]
        output_dir = os.path.join(args.output_folder, basename + time_str)

        if os.path.isfile(args.model_path):
            demo_config["models"]["model_list"][0] = args.model_path

        end_frame_id = args.end_frame_id
        fps = args.fps
        if args.total_video:
            cap = cv2.VideoCapture(args.video_path)
            end_frame_id = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            cap.release()

        demo_config["models"]["device"] = args.device
        demo_config["data"]["input_frames"] = {
            "video_file": args.video_path,
            "image_folder": None,
            "height": args.height,
            "width": args.width,
            "start_frame_id": args.start_frame_id,
            "end_frame_id": end_frame_id
        }
        demo_config["data"]["controlnet_frames"] = [demo_config["data"]["input_frames"], demo_config["data"]["input_frames"]]
        demo_config["data"]["output_folder"] = output_dir
        demo_config["data"]["fps"] = fps
        demo_config["pipeline"]["seed"] = args.seed
        demo_config["pipeline"]["pipeline_inputs"]["prompt"] = args.prompt
        demo_config["pipeline"]["pipeline_inputs"]["num_inference_steps"] = args.steps

        print(demo_config)

        runner = SDVideoPipelineRunner()
        runner.run(demo_config)





