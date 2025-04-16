import torch, os, imageio, math, json
from torchvision.transforms import v2
from einops import rearrange
import lightning as pl
import pandas as pd
from diffsynth import WanVideoPipeline, ModelManager
import torchvision
from PIL import Image
import numpy as np
from torch.utils.data import BatchSampler
from scipy.io import wavfile
from Audio_process import AudioProcessor


ASPECT_RATIO_512 = {
    "0.25": [256.0, 1024.0],
    "0.26": [256.0, 992.0],
    "0.27": [256.0, 960.0],
    "0.28": [256.0, 928.0],
    "0.32": [288.0, 896.0],
    "0.33": [288.0, 864.0],
    "0.35": [288.0, 832.0],
    "0.4": [320.0, 800.0],
    "0.42": [320.0, 768.0],
    "0.48": [352.0, 736.0],
    "0.5": [352.0, 704.0],
    "0.52": [352.0, 672.0],
    "0.57": [384.0, 672.0],
    "0.6": [384.0, 640.0],
    "0.68": [416.0, 608.0],
    "0.72": [416.0, 576.0],
    "0.78": [448.0, 576.0],
    "0.82": [448.0, 544.0],
    "0.88": [480.0, 544.0],
    "0.94": [480.0, 512.0],
    "1.0": [512.0, 512.0],
    "1.07": [512.0, 480.0],
    "1.13": [544.0, 480.0],
    "1.21": [544.0, 448.0],
    "1.29": [576.0, 448.0],
    "1.38": [576.0, 416.0],
    "1.46": [608.0, 416.0],
    "1.67": [640.0, 384.0],
    "1.75": [672.0, 384.0],
    "2.0": [704.0, 352.0],
    "2.09": [736.0, 352.0],
    "2.4": [768.0, 320.0],
    "2.5": [800.0, 320.0],
    "2.89": [832.0, 288.0],
    "3.0": [864.0, 288.0],
    "3.11": [896.0, 288.0],
    "3.62": [928.0, 256.0],
    "3.75": [960.0, 256.0],
    "3.88": [992.0, 256.0],
    "4.0": [1024.0, 256.0],
}


def get_closest_ratio(height, width, ratios=ASPECT_RATIO_512):
    aspect_ratio = height / width
    closest_ratio = min(
        ratios.keys(), key=lambda ratio: abs(float(ratio) - aspect_ratio)
    )
    return ratios[closest_ratio], float(closest_ratio)


def get_image_size_without_loading(path):
    with Image.open(path) as img:
        return img.size  # (width, height)


class TextVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path,
        metadata_path,
        frame_interval=1,
        num_frames=81,
        height=480,
        width=832,
        enable_bucket=False,
        caption_model=["qwen2-VL-72B-detail"],
    ):
        if metadata_path.endswith(".csv"):
            metadata = pd.read_csv(metadata_path)
            self.path = [
                {"video_path": os.path.join(base_path, "train", file_name)}
                for file_name in metadata["file_name"]
            ]
            self.text = [{caption_model[0]: text} for text in metadata["text"]]
        elif metadata_path.endswith(".json"):
            metadata = json.load(open(metadata_path))
            path, text = [], []
            for data in metadata:
                if "video_path" not in data:
                    continue

                path_dict = {}
                path_dict["video_path"] = (
                    data["video_path"]
                    if os.path.isabs(data["video_path"])
                    else os.path.join(base_path, data["video_path"])
                )
                if (
                    "control_file_path" in data
                    and data["control_file_path"] is not None
                ):
                    path_dict["control_path"] = (
                        data["control_file_path"]
                        if os.path.isabs(data["control_file_path"])
                        else os.path.join(base_path, data["control_file_path"])
                    )
                if "audio_path" in data and data["audio_path"] is not None:
                    path_dict["audio_path"] = (
                        data["audio_path"]
                        if os.path.isabs(data["audio_path"])
                        else os.path.join(base_path, data["audio_path"])
                    )

                txt_dict = {}
                caption_found = False
                for caption in caption_model:
                    if (
                        caption in data["description"]
                        and data["description"][caption] != ""
                    ):
                        txt_dict[caption] = data["description"][caption]
                        caption_found = True

                if not caption_found:
                    print("No caption found for video:", path_dict["video_path"])
                    continue

                path.append(path_dict)
                text.append(txt_dict)
            self.path = path
            self.text = text
        else:
            raise ValueError("Metadata file format not supported.")

        self.frame_interval = frame_interval
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.larger_side = min(height, width)
        self.enable_bucket = enable_bucket

        self.frame_process = v2.Compose(
            [
                v2.CenterCrop(size=(height, width)),
                v2.Resize(size=(height, width), antialias=True),
                v2.ToTensor(),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        self.frame_process_bucket = v2.Compose(
            [
                v2.ToTensor(),
            ]
        )

        if self.enable_bucket:
            area = height * width
            res = round(math.sqrt(area), 6)
            self.larger_side = (int(res / 16) + 1) * 16
            self.aspect_ratio_sample_size = {
                key: [x / 512 * res for x in ASPECT_RATIO_512[key]]
                for key in ASPECT_RATIO_512.keys()
            }

    def crop_and_resize(self, image, target_short_side):
        w, h = image.size
        if h < w:
            if target_short_side > h:
                return image
            new_h = target_short_side
            new_w = int(target_short_side * w / h)
        else:
            if target_short_side > w:
                return image
            new_w = target_short_side
            new_h = int(target_short_side * h / w)

        image = torchvision.transforms.functional.resize(
            image,
            (new_h, new_w),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
        )
        return image

    def load_frames_using_imageio(
        self, file_path, interval, num_frames, start_frame_id=None
    ):
        try:
            reader = imageio.get_reader(file_path)
            total_frames = reader.count_frames()
            fps = reader.get_meta_data()["fps"]
            if total_frames < num_frames * interval:
                reader.close()
                return None, None, None

            if start_frame_id is None:
                start_frame_id = torch.randint(
                    0, total_frames - (num_frames - 1) * interval, (1,)
                )[0]

            frames = []
            for frame_id in range(num_frames):
                frame = reader.get_data(start_frame_id + frame_id * interval)
                frame = Image.fromarray(frame)
                frame = self.crop_and_resize(frame, self.larger_side)

                if not self.enable_bucket:
                    frame = self.frame_process(frame)
                else:
                    frame = self.frame_process_bucket(frame)
                frames.append(frame)
            reader.close()

            frames = torch.stack(frames, dim=0)
            frames = rearrange(frames, "T C H W -> C T H W")
            return frames, start_frame_id, fps
        except Exception as e:
            print(f"Error loading frames from {file_path}: {e}")
            return None, None, None

    def load_video(self, file_path, start_frame_id=None):
        frames, start_frame_id, fps = self.load_frames_using_imageio(
            file_path, self.frame_interval, self.num_frames, start_frame_id
        )
        return frames, start_frame_id, fps

    def is_image(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        if file_ext_name.lower() in ["jpg", "png", "webp", "jpeg", "bmp"]:
            return True
        return False

    def load_image(self, file_path):
        frame = Image.open(file_path).convert("RGB")
        frame = self.crop_and_resize(frame, self.larger_side)
        if not self.enable_bucket:
            frame = self.frame_process(frame)
        else:
            frame = self.frame_process_bucket(frame)
        frame = rearrange(frame, "C H W -> C 1 H W")
        return frame

    def load_audio(self, file_path, start_frame_id=0, fps=30):
        try:
            assert self.frame_interval == 1
            sample_rate, audio = wavfile.read(file_path)
            audio_start = int(start_frame_id / fps * sample_rate)
            audio_length = int(round(self.num_frames / fps * sample_rate))
            audio_end = min(audio_start + audio_length, len(audio))
            audio = audio[audio_start:audio_end]
            return audio, sample_rate
        except Exception as e:
            print(f"Error loading audio from {file_path}: {e}")
            return None, None

    def __getitem__(self, data_id):
        text = self.text[data_id]
        path = self.path[data_id]
        data = {"text": text, "path": path}

        if self.is_image(path["video_path"]):
            video = self.load_image(path["video_path"])
            data["type"] = "image"
        else:
            video, start_frame_id, fps = self.load_video(path["video_path"])
            data["type"] = "video"

            if "control_path" in path:
                control, _, _ = self.load_video(path["control_path"], start_frame_id)
                if control is not None:
                    data["control"] = control

            if "audio_path" in path:
                audio, sample_rate = self.load_audio(
                    path["audio_path"], start_frame_id, fps
                )
                if audio is not None:
                    data["audio"] = audio
                    data["fps"] = fps
                    data["sample_rate"] = sample_rate

        if video is not None:
            data["video"] = video
            if self.enable_bucket:
                h, w = video.shape[-2:]
                ar_bucket = get_closest_ratio(h, w, self.aspect_ratio_sample_size)
                data["aspect_ratio"] = ar_bucket
        return data

    def __len__(self):
        return len(self.path)


class LightningModelForDataProcess(pl.LightningModule):
    def __init__(
        self,
        text_encoder_path,
        vae_path,
        image_encoder_path,
        tiled=False,
        tile_size=(34, 34),
        tile_stride=(18, 16),
        num_frames=81,
        cache_path=None,
        enable_bucket=False,
        wav2vec_model_path=None,
        audio_separator_model_path=None,
        audio_only_last_features=False,
    ):
        super().__init__()
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        model_manager.load_models([text_encoder_path, image_encoder_path, vae_path])
        self.pipe = WanVideoPipeline.from_model_manager(model_manager)

        self.tiler_kwargs = {
            "tiled": tiled,
            "tile_size": tile_size,
            "tile_stride": tile_stride,
        }
        self.enable_bucket = enable_bucket
        self.cache_path = (
            cache_path if not isinstance(cache_path, list) else cache_path[0]
        )
        self.wav2vec_model_path = wav2vec_model_path
        self.audio_separator_model_path = audio_separator_model_path
        self.audio_only_last_features = audio_only_last_features
        self.num_frames = num_frames

        self.build_audio_processor(only_last_features=audio_only_last_features)

    def build_audio_processor(self, only_last_features=False):
        self.audio_processor = None
        if self.wav2vec_model_path is not None:
            try:
                self.audio_processor = AudioProcessor(
                    wav2vec_model_path=self.wav2vec_model_path,
                    audio_separator_model_path=self.audio_separator_model_path,
                    sample_rate=16000,
                    device=self.pipe.device,
                    only_last_features=only_last_features,
                    cache_dir=os.path.join(self.cache_path, "vocals"),
                )
            except Exception as e:
                print(f"Error initializing AudioProcessor: {e}")

    def test_step(self, batch, batch_idx):
        text, path = batch["text"], batch["path"]
        self.pipe.device = self.device

        video = batch.get("video", None)
        control = batch.get("control", None)
        audio = batch.get("audio", None)
        if video is not None:
            video = video.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
            if control is not None:
                control = control.to(
                    dtype=self.pipe.torch_dtype, device=self.pipe.device
                )

            if self.audio_only_last_features:
                audio_feature = torch.zeros(
                    self.num_frames,
                    768,
                    device=self.pipe.device,
                    dtype=self.pipe.torch_dtype,
                )
            else:
                audio_feature = torch.zeros(
                    self.num_frames,
                    12,
                    768,
                    device=self.pipe.device,
                    dtype=self.pipe.torch_dtype,
                )

            if audio is not None and self.audio_processor is not None:
                try:
                    audio_name = os.path.splitext(path["audio_path"][0])[0]
                    audio_name = f"{audio_name}_clip.wav"
                    audio = audio[0].cpu().numpy()
                    # 1. 确保音频数据范围正确
                    if audio.dtype == np.float32 or audio.dtype == np.float64:
                        audio = (audio * 32767).clip(-32768, 32767).astype(np.int16)
                    else:
                        audio = audio.astype(np.int16)

                    # 2. 标准化音频形状
                    if len(audio.shape) == 1:
                        audio = audio.reshape(-1, 1)  # 单声道转 (n_samples, 1)
                    elif audio.shape[0] < audio.shape[1]:
                        audio = (
                            audio.T
                        )  # 转置 (n_channels, n_samples) → (n_samples, n_channels)

                    wavfile.write(
                        audio_name, int(batch["sample_rate"].cpu().item()), audio
                    )
                    audio_feature, _ = self.audio_processor.preprocess(
                        audio_name,
                        fps=batch["fps"].cpu().item(),
                        clip_length=self.num_frames,
                    )
                    audio_feature = audio_feature.to(
                        dtype=self.pipe.torch_dtype, device=self.pipe.device
                    )
                except Exception as e:
                    print(f"Error processing audio: {e}")

            if self.enable_bucket:
                h, w = video.shape[-2:]
                closest_size, closest_ratio = batch.get("aspect_ratio", (None, None))

                if closest_ratio is not None and closest_size is not None:
                    closest_ratio = closest_ratio.cpu().numpy()[0]
                    closest_size = [
                        int(x.cpu().numpy() / 16) * 16 for x in closest_size
                    ]
                    closest_size = list(map(lambda x: int(x), closest_size))

                    cache_path = os.path.join(
                        self.cache_path,
                        "cache",
                        f"ar_frames_{closest_size[0]}_{closest_size[1]}_{closest_ratio:.3f}",
                    )
                    os.makedirs(cache_path, exist_ok=True)
                    data_path = os.path.join(
                        cache_path,
                        f"{os.path.basename(path['video_path'][0])}.tensors.pth",
                    )
                    if os.path.exists(data_path):
                        print("Cache exists:", os.path.basename(path["video_path"][0]))
                        return

                    if closest_size[0] / h > closest_size[1] / w:
                        resize_size = closest_size[0], int(w * closest_size[0] / h)
                    else:
                        resize_size = int(h * closest_size[1] / w), closest_size[1]

                    transform = v2.Compose(
                        [
                            v2.Resize(
                                resize_size, interpolation=v2.InterpolationMode.BILINEAR
                            ),
                            v2.CenterCrop(closest_size),
                            v2.Normalize(
                                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
                            ),
                        ]
                    )
                    video = rearrange(video, "1 C T H W -> 1 T C H W")
                    video = transform(video)
                    video = rearrange(video, "1 T C H W -> 1 C T H W")

                    if control is not None:
                        control = rearrange(control, "1 C T H W -> 1 T C H W")
                        control = transform(control)
                        control = rearrange(control, "1 T C H W -> 1 C T H W")
                else:
                    print(
                        "No aspect ratio information found for:", path["video_path"][0]
                    )
                    return
            else:
                closest_size, closest_ratio = ["not", "bucket"], 0.0

                cache_path = os.path.join(
                    self.cache_path,
                    "cache",
                    f"ar_frames_{closest_size[0]}_{closest_size[1]}_{closest_ratio:.3f}",
                )
                os.makedirs(cache_path, exist_ok=True)
                data_path = os.path.join(
                    cache_path, f"{os.path.basename(path['video_path'][0])}.tensors.pth"
                )
                if os.path.exists(data_path):
                    print("Cache exists:", os.path.basename(path["video_path"][0]))
                    return

            frame = video[0, :, 0]
            frame = rearrange(frame, "C H W -> H W C").to(dtype=torch.float32)
            frame = Image.fromarray(
                ((frame * 0.5 + 0.5) * 255).cpu().numpy().astype(np.uint8)
            )

            tail_frame = video[0, :, -1]
            tail_frame = rearrange(tail_frame, "C H W -> H W C").to(dtype=torch.float32)
            tail_frame = Image.fromarray(
                ((tail_frame * 0.5 + 0.5) * 255).cpu().numpy().astype(np.uint8)
            )

            prompt_emb = {}
            for caption in text:
                prompt_emb[caption] = self.pipe.encode_prompt(text[caption][0])

            latents = self.pipe.encode_video(video, **self.tiler_kwargs)[0]
            if control is not None:
                control = self.pipe.encode_video(control, **self.tiler_kwargs)[0]
            else:
                control = torch.zeros_like(latents)

            num_frames, height, width = (
                video.shape[-3],
                video.shape[-2],
                video.shape[-1],
            )
            cond_data_dict_front = self.pipe.encode_image(
                image=frame,
                end_image=None,
                num_frames=num_frames,
                height=height,
                width=width,
            )
            cond_data_dict_tail = self.pipe.encode_image(
                image=frame,
                end_image=tail_frame,
                num_frames=num_frames,
                height=height,
                width=width,
            )

            data = {
                "latents": latents,
                "control": control,
                "prompt_emb": prompt_emb,
                "audio_feature": audio_feature,
                "clip_feature": cond_data_dict_front["clip_feature"][0],
                "y": cond_data_dict_front["y"][0],
                "y_tail": cond_data_dict_tail["y"][0],
            }

            torch.save(data, data_path)


class BucketAwareDataset(torch.utils.data.Dataset):
    def __init__(self, cache_root, steps_per_epoch):
        self.steps_per_epoch = steps_per_epoch
        self.samples = []
        self.bucket_info = []
        self.bucket_names = set()

        if isinstance(cache_root, list):
            cache_roots = cache_root
        else:
            cache_roots = [cache_root]

        for cache in cache_roots:
            for root, dirs, _ in os.walk(cache):
                for bucket_dir in dirs:
                    if bucket_dir.startswith("ar_frames_"):
                        self.bucket_names.add(bucket_dir)
                        bucket_path = os.path.join(root, bucket_dir)
                        if os.path.isdir(bucket_path):
                            files = [
                                os.path.join(bucket_path, f)
                                for f in os.listdir(bucket_path)
                                if f.endswith(".tensors.pth")
                            ]
                            self.samples.extend(files)
                            self.bucket_info.extend([bucket_dir] * len(files))

    def __getitem__(self, index):
        try:
            data = torch.load(
                self.samples[index], map_location="cpu", weights_only=True
            )
            data["name"] = [self.samples[index]]
            data["bucket_name"] = [self.bucket_info[index]]
        except Exception as e:
            print(f"Error loading file {self.samples[index]}: {e}")
            idx = torch.randint(0, len(self.samples), (1,))[0]
            data = torch.load(self.samples[idx], map_location="cpu", weights_only=True)
            data["name"] = [self.samples[index]]
            data["bucket_name"] = [self.bucket_info[index]]

        return data

    def get_bucket_names(self):
        return list(self.bucket_names)

    def __len__(self):
        return min(self.steps_per_epoch, len(self.samples))


class EnhancedBucketBatchSampler(BatchSampler):
    def __init__(
        self, sampler, dataset, batch_size, bucket_names, drop_last=False, seed=None
    ):
        self.sampler = sampler
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.buckets = {bucket_name: [] for bucket_name in bucket_names}
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        if self.seed is not None:
            generator = torch.Generator().manual_seed(self.seed + self.epoch)
            self.epoch += 1
        else:
            generator = None

        if self.dataset.steps_per_epoch < len(self.dataset.bucket_info):
            base_id = torch.randint(
                0, len(self.dataset.bucket_info), (1,), generator=generator
            )[0]
        else:
            base_id = 0

        for idx in self.sampler:
            idx = (idx + base_id) % len(self.dataset.bucket_info)
            bucket_name = self.dataset.bucket_info[idx]
            self.buckets[bucket_name].append(idx)
            if len(self.buckets[bucket_name]) == self.batch_size:
                yield self.buckets[bucket_name]
                self.buckets[bucket_name] = []


def data_process(args):
    dataset = TextVideoDataset(
        args.dataset_path,
        args.dataset_file,
        frame_interval=args.frame_interval,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        enable_bucket=args.enable_bucket,
        caption_model=args.caption_model,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, shuffle=False, batch_size=1, num_workers=args.dataloader_num_workers
    )
    model = LightningModelForDataProcess(
        text_encoder_path=args.text_encoder_path,
        image_encoder_path=args.image_encoder_path,
        vae_path=args.vae_path,
        tiled=args.tiled,
        tile_size=(args.tile_size_height, args.tile_size_width),
        tile_stride=(args.tile_stride_height, args.tile_stride_width),
        cache_path=args.cache_path,
        enable_bucket=args.enable_bucket,
        num_frames=args.num_frames,
        wav2vec_model_path=args.wav2vec_model_path,
        audio_separator_model_path=args.audio_separator_model_path,
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices="auto",
        default_root_dir=args.output_path,
        precision=args.precision,
        num_nodes=args.nnodes,
    )
    trainer.test(model, dataloader)
