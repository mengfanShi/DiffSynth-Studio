from Prompt_extend import QwenPromptExpander, VL_EN_DIGITAL_HUMAN_PROMPT
from PIL import Image
import argparse
import os
import cv2


def parse_args():
    parser = argparse.ArgumentParser(description="Prompt Extender")
    parser.add_argument(
        "--input", type=str, required=True, help="Path dir of the input"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path dir of the output"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Prompt to extend (can be text or txt file path)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--language", type=str, default="en", help="Language for the prompt"
    )
    parser.add_argument("--task", type=str, default="i2v", help="Task for the prompt")
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device (e.g., 'cuda', 'cpu')."
    )
    parser.add_argument(
        "--prompt_extender_model",
        type=str,
        required=True,
        help="Prompt extender model path.",
    )
    return parser.parse_args()


def process_prompts(prompt_source):
    """处理prompt输入，返回prompt列表"""
    if os.path.isfile(prompt_source) and prompt_source.endswith(".txt"):
        with open(prompt_source, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    return [prompt_source]


def process_media_file(media_path, model, output_dir, expander_kwargs, is_video=False):
    """处理单个图片或视频文件"""
    basename = os.path.splitext(os.path.basename(media_path))[0]
    output_file = os.path.join(output_dir, f"{basename}.txt")

    # 确保每次运行时覆盖旧文件而不是追加
    if os.path.exists(output_file):
        os.remove(output_file)

    if is_video:
        capture = cv2.VideoCapture(media_path)
        ret, frame = capture.read()
        if not ret:
            capture.release()
            return
        image = Image.fromarray(frame)
        capture.release()
    else:
        image = Image.open(media_path)

    image = image.convert("RGB")

    with open(output_file, "a", encoding="utf-8") as f:
        for prompt in expander_kwargs["prompts"]:
            expander_kwargs["prompt"] = prompt
            prompt_output = model(
                system_prompt=VL_EN_DIGITAL_HUMAN_PROMPT,
                image=image,
                **expander_kwargs,
            )
            if prompt_output.status:
                prompt = prompt_output.prompt
                prompt = prompt.strip().replace("\n", " ")  # 去除换行符
                f.write(f"{prompt}\n")
                print(
                    f"{'Video' if is_video else 'Image'} {basename} Prompt extended: {prompt}"
                )


def main():
    args = parse_args()

    if not os.path.exists(args.input):
        raise ValueError(f"Input path does not exist: {args.input}")

    os.makedirs(args.output, exist_ok=True)

    try:
        model = QwenPromptExpander(
            model_name=args.prompt_extender_model,
            device=args.device,
            is_vl="i2v" in args.task,
        )
    except Exception as e:
        print(f"Error initializing prompt extender: {e}")
        return

    # 处理prompt输入
    prompts = process_prompts(args.prompt)
    if not prompts:
        raise ValueError("No valid prompts provided")

    expander_kwargs = {
        "prompts": prompts,  # 存储所有prompt
        "tar_lang": args.language,
        "seed": args.seed,
        "prompt": "",  # 会被process_media_file覆盖
    }

    if os.path.isfile(args.input):
        # 处理单个文件
        if args.input.lower().endswith((".png", ".jpg", ".jpeg")):
            process_media_file(args.input, model, args.output, expander_kwargs)
        elif args.input.lower().endswith((".mp4", ".avi")):
            process_media_file(
                args.input, model, args.output, expander_kwargs, is_video=True
            )
        else:
            print(f"Unsupported file format: {args.input}")
    else:
        # 处理目录
        for root, dirs, files in os.walk(args.input):
            for file in files:
                file_path = os.path.join(root, file)
                if file.lower().endswith((".png", ".jpg", ".jpeg")):
                    process_media_file(file_path, model, args.output, expander_kwargs)
                elif file.lower().endswith((".mp4", ".avi")):
                    process_media_file(
                        file_path, model, args.output, expander_kwargs, is_video=True
                    )


if __name__ == "__main__":
    main()
