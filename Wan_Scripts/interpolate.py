import os
import cv2
import argparse
from PIL import Image
from RIFE import RIFESmoother
from diffsynth import save_video


def parse_args():
    parser = argparse.ArgumentParser(description="Interpolate video frames using RIFE")
    parser.add_argument(
        "--input", type=str, required=True, help="Path to the input video file"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to save the interpolated video"
    )
    parser.add_argument(
        "--interpolate", type=int, default=1, help="Number of times to interpolate"
    )
    parser.add_argument("--fps", type=int, default=30, help="FPS of the output video")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for RIFE (cpu or cuda)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Scripts/RIFE/flownet.pkl",
        help="RIFE model to use",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    videos = []

    if os.path.isdir(args.input):
        for file in os.listdir(args.input):
            if file.endswith((".mp4", ".avi", ".mkv")):
                videos.append(os.path.join(args.input, file))
    else:
        videos.append(args.input)

    try:
        smoother = RIFESmoother(
            device=args.device,
            model=args.model,
            interpolate=True,
            interpolate_times=args.interpolate,
        )
    except Exception as e:
        print(f"Error initializing RIFE: {e}")
        return

    for video in videos:
        print(f"Processing video: {video}")
        frames = []
        cap = cv2.VideoCapture(video)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
        cap.release()

        os.makedirs(args.output, exist_ok=True)
        output = os.path.join(args.output, os.path.basename(video))
        try:
            frames = smoother(frames)
            save_video(frames, output, fps=args.fps)
        except Exception as e:
            print(f"Error processing video: {e}")
            continue
        print(f"Video processed: {output}")
    print("All videos processed.")
    return


if __name__ == "__main__":
    main()
