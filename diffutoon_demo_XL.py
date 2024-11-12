from examples.Diffutoon.diffutoon_toon_shading import config_XL as config
from diffsynth import SDXLVideoPipelineRunner,SDXLImagePipelineRunner
from diffsynth.extensions.CUGAN.inference_video import VideoRealWaifuUpScaler
from diffsynth.extensions.Tagger.tag_images_by_wd14_tagger import get_tagger_tag
from diffsynth.extensions.SceneDet.scenedet import save_scenes
from diffsynth.data.video import save_video
from moviepy.editor import VideoFileClip, concatenate_videoclips
import argparse
import datetime
import copy
import os
import cv2

video_model_manager = None
video_pipe = None
image_model_manager = None
image_pipe = None

def parse_args():
    parser = argparse.ArgumentParser(description="Run Video Style Transfer with specified parameters.")

    parser.add_argument("--video", type=str, required=True, help="Path to the input video file.")
    parser.add_argument("--prompt", type=str, default="masterpiece", help="Prompt for the pipeline.")
    parser.add_argument("--height", type=int, default=1024, help="Height of the input video frames, multiple of 64.")
    parser.add_argument("--width", type=int, default=1024, help="Width of the input video frames, multiple of 64.")
    parser.add_argument("--crop_method", type=str, default="padding", help="Crop method for the pipeline.")

    parser.add_argument("--device", type=str, default="cuda", help="Device to run the pipeline on (e.g., 'cuda', 'cpu').")
    parser.add_argument("--start_frame_id", type=int, default=0, help="Starting frame ID for processing.")
    parser.add_argument("--end_frame_id", type=int, default=30, help="Ending frame ID for processing.")
    parser.add_argument("--total_video", action="store_true", help="whether process total video.")
    parser.add_argument("--origin_size", action="store_true", help="whether output origin size of input video.")
    parser.add_argument("--output_folder", type=str, default="data/toon_video", help="Folder to save the output frames.")
    parser.add_argument("--steps", type=int, default=10, help="Number of inference steps.")
    parser.add_argument("--seed", type=int, default=42, help="seed of generate video.")
    parser.add_argument("--cfg_scale", type=float, default=7.0, help="cfg_scale.")
    parser.add_argument("--fps", type=float, default=None, help="fps of generate video.")
    parser.add_argument("--model_path", type=str, default="models/stable_diffusion_xl/IDillustration互联网插画风模型_v1.0.safetensors",
                        help="Stable Diffusion model path.")
    parser.add_argument("--lora", type=str, nargs='+', help="Lora model path.")
    parser.add_argument("--lora_alpha", type=float, nargs='+', help="Lora alpha.")
    parser.add_argument("--animatediff", type=str, default="models/AnimateDiff/mm_sdxl_v10.ckpt",
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
    parser.add_argument("--rife_model", type=str, default="models/RIFE/flownet_v3.pkl")
    parser.add_argument("--use_fastblend", action="store_true", help="whether use FastBlend smooth.")
    parser.add_argument("--interpolate", type=int, default=0, help="interpolate times param.")
    parser.add_argument("--trans_first_frame", action="store_true", help="whether transform input video's first frame.")

    parser.add_argument("--controlnet_tile_scale", type=float, default=0.3, help="Controlnet alpha.")
    parser.add_argument("--controlnet_lineart_scale", type=float, default=0.3, help="Controlnet alpha.")
    parser.add_argument("--controlnet_softedge_scale", type=float, default=0.3, help="Controlnet alpha.")
    parser.add_argument("--controlnet_depth_scale", type=float, default=0.3, help="Controlnet alpha.")

    parser.add_argument("--use_tagger", action="store_true", help="whether use tagger to tag image.")
    parser.add_argument("--tagger_dir", type=str, default="models/Tagger")

    args = parser.parse_args()
    return args

def update_config(demo_config, args):
    if os.path.isfile(args.model_path):
        demo_config["models"]["model_list"][0] = args.model_path

    if os.path.isfile(args.animatediff):
        demo_config["models"]["model_list"][1] = args.animatediff

    if os.path.isfile(args.rife_model):
        demo_config["models"]["model_list"][-1] = args.rife_model

    if args.lora and args.lora_alpha:
        for lora, alpha in zip(args.lora, args.lora_alpha):
            demo_config["models"]["loras"].append(lora)
            demo_config["models"]["lora_alphas"].append(alpha)

    if args.use_rife:
        interpolate = max(args.interpolate + 1, 1) > 1
        demo_config["smoother_configs"].append(
            {"processor_type": "RIFE",
             "config": {"interpolate": interpolate, "interpolate_times": args.interpolate}}
        )

    if args.use_fastblend:
        demo_config["smoother_configs"].append({"processor_type": "FastBlend", "config": {}})

    demo_config["models"]["device"] = args.device
    demo_config["models"]["controlnet_units"][0]["scale"] = args.controlnet_tile_scale
    demo_config["models"]["controlnet_units"][1]["scale"] = args.controlnet_lineart_scale
    demo_config["models"]["controlnet_units"][2]["scale"] = args.controlnet_softedge_scale
    demo_config["models"]["controlnet_units"][3]["scale"] = args.controlnet_depth_scale

    demo_config["pipeline"]["seed"] = args.seed
    demo_config["pipeline"]["pipeline_inputs"]["prompt"] = args.prompt
    demo_config["pipeline"]["pipeline_inputs"]["num_inference_steps"] = args.steps
    demo_config["pipeline"]["pipeline_inputs"]["cfg_scale"] = args.cfg_scale
    demo_config["pipeline"]["pipeline_inputs"]["animatediff_batch_size"] = args.animatediff_size
    demo_config["pipeline"]["pipeline_inputs"]["animatediff_stride"] = args.animatediff_stride
    demo_config["pipeline"]["pipeline_inputs"]["denoising_strength"] = args.denoise

def interpolate_video(video, output_dir, interpolate_times=1):
    _, ext = os.path.splitext(os.path.basename(video))
    new_video = os.path.join(output_dir, "video_sampling" + ext)
    cap = cv2.VideoCapture(video)
    out = cv2.VideoWriter(
        new_video,
        cv2.VideoWriter_fourcc(*'mp4v'), cap.get(cv2.CAP_PROP_FPS) / interpolate_times,
        (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    )

    frame_count = 0
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % interpolate_times == 0:
            out.write(frame)
            frame_count += 1
        count += 1
    cap.release()
    out.release()
    return new_video, frame_count


if __name__ == "__main__":
    time_str = datetime.datetime.now().strftime("-%Y-%m-%dT%H-%M-%S")
    demo_config = config.copy()
    args = parse_args()

    videos = []
    if os.path.isfile(args.video):
        basename_ori, _ = os.path.splitext(os.path.basename(args.video))
        output_dir_ori = os.path.join(args.output_folder, basename_ori + time_str + f"-{args.steps}steps")
        os.makedirs(output_dir_ori, exist_ok=True)
        videos = save_scenes(args.video, output_dir_ori)
        update_config(demo_config, args)

    videos_out = []
    for video in videos:
        basename, ext = os.path.splitext(os.path.basename(video))
        output_dir = os.path.join(output_dir_ori, basename + time_str + f"-{args.steps}steps")
        os.makedirs(output_dir, exist_ok=True)

        video_upscaler = None
        if os.path.exists(args.super_model):
            video_upscaler = VideoRealWaifuUpScaler(
                scale=args.upscaler_scale, weight_path=args.super_model, device=args.device)

        # Get video info
        cap = cv2.VideoCapture(video)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) if args.fps is None else args.fps

        ret, frame = cap.read()
        if ret:
            cv2.imwrite(os.path.join(output_dir, "frame_0.jpg"), frame)
        cap.release()

        # Get prompt by Tagger
        tag = args.prompt
        if args.use_tagger and os.path.exists(os.path.join(output_dir, "frame_0.jpg")):
            try:
                tag = get_tagger_tag(args.tagger_dir, [os.path.join(output_dir, "frame_0.jpg")], 
                                    prompt=args.prompt, general_threshold=0.5, character_threshold=0.5)
            except Exception as e:
                print(f"Tagger Error: {e}")

        # Upscale input if input res is small
        if args.upscale_input and video_upscaler is not None:
            video_upscaler.process_video(video, output_dir, "upscaled_input")
            video = os.path.join(output_dir, "upscaled_input" + ext)
            frame_width *= args.upscaler_scale
            frame_height *= args.upscaler_scale

        width = ((frame_width + 63) // 64) * 64 if args.origin_size else ((args.width + 63) // 64) * 64
        height = ((frame_height + 63) // 64) * 64 if args.origin_size else ((args.height + 63) // 64) * 64

        # If short video, do image generation
        interpolate_times = max(args.interpolate + 1, 1)
        interpolate = args.use_rife and interpolate_times > 1
        short_video = frame_count < args.animatediff_size

        if not short_video and frame_count < args.animatediff_size * interpolate_times:
            interpolate = False

        if short_video or args.trans_first_frame:
            image_config = {}
            image_config["models"] = copy.deepcopy(demo_config["models"])
            del image_config["models"]["model_list"][-1]
            del image_config["models"]["model_list"][1]
            image_config["data"] = {}
            image_config["pipeline"] = {}
            image_config["pipeline"]["pipeline_inputs"] = {}
            image_config["data"]["input_frames"] = {
                "video_file": video,
                "image_folder": None,
                "height": height,
                "width": width,
                "crop_method": args.crop_method,
                "original_width": frame_width if args.origin_size else args.width,
                "original_height": frame_height if args.origin_size else args.height,
            }
            image_config["data"]["output_folder"] = output_dir
            image_config["pipeline"]["seed"] = args.seed
            image_config["pipeline"]["pipeline_inputs"]["prompt"] = tag
            image_config["pipeline"]["pipeline_inputs"]["negative_prompt"] = "verybadimagenegative_v1.3"
            image_config["pipeline"]["pipeline_inputs"]["num_inference_steps"] = args.steps
            image_config["pipeline"]["pipeline_inputs"]["cfg_scale"] = args.cfg_scale
            image_config["pipeline"]["pipeline_inputs"]["denoising_strength"] = args.denoise

            print(image_config)
            image_runner = SDXLImagePipelineRunner()
            if image_model_manager is None or image_pipe is None:
                image_model_manager, image_pipe = image_runner.load_pipeline(**image_config["models"])
            image_gen = image_runner.run_with_pipe(image_model_manager, image_pipe, image_config, out_name=basename)

            if args.upscale_output:
                image_gen = video_upscaler.process_image(image_gen)
                image_gen.save(os.path.join(output_dir, "super_image.png"))

            if short_video:
                merge_video = os.path.join(output_dir, basename + ext)
                frames = [image_gen for _ in range(frame_count)]
                save_video(frames, merge_video, fps)

                videos_out.append(merge_video)

        # Do video transfer
        else:
            if args.use_rife:
                if not demo_config["smoother_configs"]:
                    demo_config["smoother_configs"].append(
                        {"processor_type": "RIFE", "config": {"interpolate": interpolate, "interpolate_times": args.interpolate}})
                else:
                    demo_config["smoother_configs"][0] = {
                        "processor_type": "RIFE", "config": {"interpolate": interpolate, "interpolate_times": args.interpolate}}

            if interpolate:
                video, frame_count = interpolate_video(video, output_dir, interpolate_times)

            start_frame_id = 0 if args.total_video else args.start_frame_id
            start_frame_id = min(max(0, start_frame_id), frame_count)
            end_frame_id = frame_count if args.total_video else args.end_frame_id
            end_frame_id = min(max(0, end_frame_id), frame_count)

            demo_config["data"]["input_frames"] = {
                "video_file": video,
                "image_folder": None,
                "height": height,
                "width": width,
                "start_frame_id": start_frame_id,
                "end_frame_id": end_frame_id,
                "crop_method": args.crop_method,
                "original_width": frame_width if args.origin_size else args.width,
                "original_height": frame_height if args.origin_size else args.height,
            }
            demo_config["data"]["controlnet_frames"] = [
                demo_config["data"]["input_frames"], demo_config["data"]["input_frames"],
                demo_config["data"]["input_frames"], demo_config["data"]["input_frames"]]
            demo_config["data"]["output_folder"] = output_dir
            demo_config["data"]["fps"] = fps
            demo_config["pipeline"]["pipeline_inputs"]["prompt"] = tag

            print(demo_config)
            video_runner = SDXLVideoPipelineRunner()
            if video_model_manager is None or video_pipe is None:
                video_model_manager, video_pipe = video_runner.load_pipeline(**demo_config["models"])
            video_runner.run_with_pipe(video_model_manager, video_pipe, demo_config, out_name=basename, ext=ext)
            out_video = os.path.join(output_dir, basename + ext)

            if interpolate:
                os.remove(video)

            if args.upscale_output:
                video_upscaler.process_video(out_video, output_dir, basename + "_sup")
                out_video = os.path.join(output_dir, basename + "_sup" + ext)

            videos_out.append(out_video)

    if videos_out:
        if len(videos_out) > 1:
            clips = []
            for video_path in videos_out:
                clip = VideoFileClip(video_path)
                if clip.fps is None:
                    print(f"Warning: {video_path} has no defined frame rate. Setting default fps.")
                    clip = clip.set_fps(fps)
                clips.append(clip)

            final_clip = concatenate_videoclips(clips)
            final_clip.write_videofile(os.path.join(output_dir_ori, basename_ori + ".mp4"), fps=fps)

            for clip in clips:
                clip.close()
        else:
            print("Only one video provided, writing directly...")
            single_clip = VideoFileClip(videos_out[0])
            single_clip.write_videofile(os.path.join(output_dir_ori, basename_ori + ".mp4"), fps=fps)
            single_clip.close()







