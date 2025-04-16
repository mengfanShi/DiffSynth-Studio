from CUGAN import VideoRealWaifuUpScaler
from PIL import Image
import argparse
import numpy as np
import os
import cv2


def parse_args():
    parser = argparse.ArgumentParser(description="Super Resolution")
    parser.add_argument(
        "--input", type=str, required=True, help="Path to input file or directory"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output directory path"
    )
    parser.add_argument(
        "--out_width", type=int, default=None, help="Width of the output"
    )
    parser.add_argument(
        "--out_height", type=int, default=None, help="Height of the output"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device (e.g., 'cuda', 'cpu')"
    )
    parser.add_argument(
        "--super_model", type=str, default="Scripts/CUGAN/pro-conservative-up2x.pth"
    )
    parser.add_argument(
        "--upscaler_scale", type=int, default=2, help="Super Resolution scaler scale"
    )
    parser.add_argument(
        "--keep_ratio", action="store_true", help="Keep the aspect ratio of the input"
    )
    return parser.parse_args()


def calculate_output_size(
    input_width, input_height, target_width, target_height, keep_ratio
):
    if not (target_width and target_height):
        return None, None

    if keep_ratio:
        area = target_width * target_height
        ratio = input_height / input_width
        out_height = round(np.sqrt(area * ratio))
        out_width = round(np.sqrt(area / ratio))
        return out_width, out_height

    return target_width, target_height


def process_image(
    model, image_path, output_dir, out_width=None, out_height=None, keep_ratio=False
):
    img = Image.open(image_path)
    res = model.process_image(img)

    if out_width and out_height:
        out_width, out_height = calculate_output_size(
            img.width, img.height, out_width, out_height, keep_ratio
        )
        res = res.resize((out_width, out_height))

    output_path = os.path.join(output_dir, os.path.basename(image_path))
    res.save(output_path)
    return output_path


def process_video(
    model, video_path, output_dir, out_width=None, out_height=None, keep_ratio=False
):
    basename = os.path.splitext(os.path.basename(video_path))[0] + "_sup"
    ext = os.path.splitext(video_path)[-1]

    if out_width and out_height and keep_ratio:
        cap = cv2.VideoCapture(video_path)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        cap.release()
        out_width, out_height = calculate_output_size(
            width, height, out_width, out_height, keep_ratio
        )

    output_path = os.path.join(output_dir, basename + ext)
    model.process_video(
        video_path, output_dir, basename, out_width=out_width, out_height=out_height
    )
    return output_path


def main():
    args = parse_args()

    if not os.path.exists(args.input):
        raise ValueError(f"Input path does not exist: {args.input}")

    os.makedirs(args.output, exist_ok=True)

    try:
        model = VideoRealWaifuUpScaler(
            scale=args.upscaler_scale, weight_path=args.super_model, device=args.device
        )
    except Exception as e:
        print(f"Error initializing super resolution model: {e}")
        return

    if os.path.isfile(args.input):
        if args.input.lower().endswith((".png", ".jpg", ".jpeg")):
            output_path = process_image(
                model,
                args.input,
                args.output,
                args.out_width,
                args.out_height,
                args.keep_ratio,
            )
            print(f"Image processed and saved to: {output_path}")
        elif args.input.lower().endswith((".mp4", ".avi")):
            output_path = process_video(
                model,
                args.input,
                args.output,
                args.out_width,
                args.out_height,
                args.keep_ratio,
            )
            print(f"Video processed and saved to: {output_path}")
    else:
        for root, _, files in os.walk(args.input):
            for file in files:
                file_path = os.path.join(root, file)
                if file.lower().endswith((".png", ".jpg", ".jpeg")):
                    process_image(
                        model,
                        file_path,
                        args.output,
                        args.out_width,
                        args.out_height,
                        args.keep_ratio,
                    )
                elif file.lower().endswith((".mp4", ".avi")):
                    process_video(
                        model,
                        file_path,
                        args.output,
                        args.out_width,
                        args.out_height,
                        args.keep_ratio,
                    )


if __name__ == "__main__":
    main()
