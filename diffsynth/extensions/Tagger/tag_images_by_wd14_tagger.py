import argparse
import csv
import os
import cv2
import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image
from tqdm import tqdm

import onnx
import onnxruntime as ort

import logging
logger = logging.getLogger(__name__)

# from wd14 tagger
IMAGE_SIZE = 448

# wd-v1-4-swinv2-tagger-v2 / wd-v1-4-vit-tagger / wd-v1-4-vit-tagger-v2/ wd-v1-4-convnext-tagger / wd-v1-4-convnext-tagger-v2
DEFAULT_WD14_TAGGER_REPO = "SmilingWolf/wd-v1-4-convnext-tagger-v2"


def preprocess_image(image):
    image = np.array(image)
    image = image[:, :, ::-1]  # RGB->BGR

    # pad to square
    size = max(image.shape[0:2])
    pad_x = size - image.shape[1]
    pad_y = size - image.shape[0]
    pad_l = pad_x // 2
    pad_t = pad_y // 2
    image = np.pad(image, ((pad_t, pad_y - pad_t), (pad_l, pad_x - pad_l), (0, 0)), mode="constant", constant_values=255)

    interp = cv2.INTER_AREA if size > IMAGE_SIZE else cv2.INTER_LANCZOS4
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=interp)

    image = image.astype(np.float32)
    return image


def get_tagger_tag(model_location, image_paths, remove_underscore=True, undesired_tags="", general_threshold=0.35,\
    character_threshold=0.1, frequency_tags=False, prompt="masterpiece"):
    if not os.path.exists(model_location):
        os.makedirs(model_location, exist_ok=True)
        logger.info(f"downloading wd14 tagger model from hf_hub. id: {DEFAULT_WD14_TAGGER_REPO}")
        files = ["selected_tags.csv", "model.onnx"]
        for file in files:
            hf_hub_download(DEFAULT_WD14_TAGGER_REPO, file, local_dir=model_location, force_download=True, force_filename=file)
    else:
        logger.info("using existing wd14 tagger model")

    onnx_path = f"{model_location}/model.onnx"
    logger.info(f"loading onnx model: {onnx_path}")

    if not os.path.exists(onnx_path):
        raise Exception(
            f"onnx model not found: {onnx_path}, please redownload the model"
        )

    if "OpenVINOExecutionProvider" in ort.get_available_providers():
        # requires provider options for gpu support
        # fp16 causes nonsense outputs
        ort_sess = ort.InferenceSession(
            onnx_path,
            providers=(["OpenVINOExecutionProvider"]),
            provider_options=[{'device_type' : "GPU_FP32"}],
        )
    else:
        ort_sess = ort.InferenceSession(
            onnx_path,
            providers=(
                ["CUDAExecutionProvider"] if "CUDAExecutionProvider" in ort.get_available_providers() else
                ["ROCMExecutionProvider"] if "ROCMExecutionProvider" in ort.get_available_providers() else
                ["CPUExecutionProvider"]
            ),
        )

    with open(os.path.join(model_location, "selected_tags.csv"), "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        line = [row for row in reader]
        header = line[0]
        rows = line[1:]
    assert header[0] == "tag_id" and header[1] == "name" and header[2] == "category", f"unexpected csv format: {header}"

    rating_tags = [row[1] for row in rows[0:] if row[2] == "9"]
    general_tags = [row[1] for row in rows[0:] if row[2] == "0"]
    character_tags = [row[1] for row in rows[0:] if row[2] == "4"]

    if remove_underscore:
        rating_tags = [tag.replace("_", " ") if len(tag) > 3 else tag for tag in rating_tags]
        general_tags = [tag.replace("_", " ") if len(tag) > 3 else tag for tag in general_tags]
        character_tags = [tag.replace("_", " ") if len(tag) > 3 else tag for tag in character_tags]

    tag_freq = {}

    caption_separator = ", "
    stripped_caption_separator = caption_separator.strip()
    undesired_tags = set([tag.strip() for tag in undesired_tags.split(stripped_caption_separator) if tag.strip() != ""])

    model = onnx.load(onnx_path)
    model_input = model.graph.input[0].name
    del model

    def run_batch(path_imgs, save=False):
        imgs = np.array([im for _, im in path_imgs])

        probs = ort_sess.run(None, {model_input: imgs})[0]
        probs = probs[: len(path_imgs)]

        general_tag = []

        for (image_path, _), prob in zip(path_imgs, probs):
            combined_tags = []
            rating_tag_text = ""
            character_tag_text = ""
            general_tag_text = ""

            for i, p in enumerate(prob[4:]):
                if i < len(general_tags) and p >= general_threshold:
                    tag_name = general_tags[i]

                    if tag_name not in undesired_tags:
                        tag_freq[tag_name] = tag_freq.get(tag_name, 0) + 1
                        general_tag_text += caption_separator + tag_name
                        combined_tags.append(tag_name)
                elif i >= len(general_tags) and p >= character_threshold:
                    tag_name = character_tags[i - len(general_tags)]

                    if tag_name not in undesired_tags:
                        tag_freq[tag_name] = tag_freq.get(tag_name, 0) + 1
                        character_tag_text += caption_separator + tag_name

            ratings_probs = prob[:4]
            rating_index = ratings_probs.argmax()
            found_rating = rating_tags[rating_index]

            if found_rating not in undesired_tags:
                tag_freq[found_rating] = tag_freq.get(found_rating, 0) + 1
                rating_tag_text = found_rating
                combined_tags.append(found_rating)

            if len(general_tag_text) > 0:
                general_tag_text = general_tag_text[len(caption_separator) :]
            if len(character_tag_text) > 0:
                character_tag_text = character_tag_text[len(caption_separator) :]

            if save:
                caption_file = os.path.splitext(image_path)[0] + ".txt"
                tag_text = caption_separator.join(combined_tags)

                with open(caption_file, "wt", encoding="utf-8") as f:
                    f.write(tag_text + "\n")
                    logger.info("")
                    logger.info(f"{image_path}:")
                    logger.info(f"\tRating tags: {rating_tag_text}")
                    logger.info(f"\tCharacter tags: {character_tag_text}")
                    logger.info(f"\tGeneral tags: {general_tag_text}")

            general_tag.append(general_tag_text)

        return general_tag

    data = [[(None, ip)] for ip in image_paths]

    b_imgs = []
    for data_entry in tqdm(data, smoothing=0.0):
        for data in data_entry:
            if data is None:
                continue

            image, image_path = data
            if image is None:
                try:
                    image = Image.open(image_path)
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                    image = preprocess_image(image)
                except Exception as e:
                    logger.error(f"Could not load image path: {image_path}, error: {e}")
                    continue
            b_imgs.append((image_path, image))

    tags = []
    if len(b_imgs) > 0:
        b_imgs = [(str(image_path), image) for image_path, image in b_imgs]
        tags = run_batch(b_imgs)

    if frequency_tags:
        sorted_tags = sorted(tag_freq.items(), key=lambda x: x[1], reverse=True)
        print("Tag frequencies:")
        for tag, freq in sorted_tags:
            print(f"{tag}: {freq}")

    tags.insert(0, prompt + caption_separator)
    tag_str = "".join(tags)
    logger.info("done!")

    return tag_str


if __name__ == "__main__":
    image_paths = ["/home/mi/yuhan/data/llm/videotransfer/output/control_0/0.png"]
    model_dir = "/home/mi/yuhan/data/llm/videotransfer/models/Tagger"
    tag_str = get_tagger_tag(model_dir, image_paths)
    print(tag_str)
