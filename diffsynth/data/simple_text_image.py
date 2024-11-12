import torch, os, csv, random
from torchvision import transforms
import pandas as pd
import numpy as np
from PIL import Image
import imageio
from decord import VideoReader
from tqdm import tqdm


class TextImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, steps_per_epoch=10000, height=1024, width=1024, center_crop=True, random_flip=False):
        self.steps_per_epoch = steps_per_epoch
        metadata = pd.read_csv(os.path.join(dataset_path, "train/metadata.csv"))
        self.path = [os.path.join(dataset_path, "train", file_name) for file_name in metadata["file_name"]]
        self.text = metadata["text"].to_list()
        self.image_processor = transforms.Compose(
            [
                transforms.Resize(max(height, width), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop((height, width)) if center_crop else transforms.RandomCrop((height, width)),
                transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )


    def __getitem__(self, index):
        data_id = torch.randint(0, len(self.path), (1,))[0]
        data_id = (data_id + index) % len(self.path) # For fixed seed.
        text = self.text[data_id]
        image = Image.open(self.path[data_id]).convert("RGB")
        image = self.image_processor(image)
        return {"text": text, "image": image}


    def __len__(self):
        return self.steps_per_epoch


class TextVideoDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            csv_path, video_folder,
            height=1024, width=1024, sample_stride=4, sample_n_frames=16,
            is_image=False, control_video_folder=None, process_image=None, 
            processors_id=None, extra_prompts=None
        ):

        with open(csv_path, 'r') as csvfile:
            self.dataset = list(csv.DictReader(csvfile))
        self.length = len(self.dataset)

        self.video_folder    = video_folder
        self.sample_stride   = sample_stride
        self.sample_n_frames = sample_n_frames
        self.is_image        = is_image
        self.control_video_folder = control_video_folder
        self.process_image = process_image
        self.shape = (width, height)

        self.pixel_transforms = transforms.Compose([

            transforms.Resize(max(height, width), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop((height, width)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])

        self.processors_id = processors_id
        self.extra_promts = extra_prompts

    def get_batch(self, idx):
        video_dict = self.dataset[idx]
        videoid, name = video_dict['videoid'], video_dict['name']
        if self.extra_promts is not None:
            name += self.extra_promts

        video_dir    = os.path.join(self.video_folder, f"{videoid}.mp4")
        video_reader = VideoReader(video_dir)
        video_length = len(video_reader)

        if not self.is_image:
            clip_length = min(video_length, (self.sample_n_frames - 1) * self.sample_stride + 1)
            start_idx   = random.randint(0, video_length - clip_length)
            batch_index = np.linspace(start_idx, start_idx + clip_length - 1, self.sample_n_frames, dtype=int)
        else:
            batch_index = [random.randint(0, video_length - 1)]

        pixel_values = torch.from_numpy(video_reader.get_batch(batch_index).asnumpy()).permute(0, 3, 1, 2).contiguous()
        pixel_values = pixel_values * (2 / 255) - 1
        del video_reader

        if self.control_video_folder is not None and self.processors_id is not None:
            controlnet_frames_ = []
            for processor_id in self.processors_id:
                videoid = video_dict.get('control_videoid', videoid)
                video_dir = os.path.join(self.control_video_folder, processor_id, f"{videoid}.mp4")
                video_reader = VideoReader(video_dir)
                control_pixel_values = video_reader.get_batch(batch_index).asnumpy()
                controlnet_frame = [Image.fromarray(img).resize(self.shape) for img in control_pixel_values]
                controlnet_frame = torch.concat([torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0) for img in controlnet_frame], dim=0)
                controlnet_frames_.append(controlnet_frame)
            controlnet_frames = torch.stack(controlnet_frames_, dim=0)
        else:
            controlnet_frames = None

        if self.is_image:
            pixel_values = pixel_values[0]

        return pixel_values, name, controlnet_frames

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        while True:
            try:
                pixel_values, name, controlnet_frames= self.get_batch(idx)
                break

            except Exception as e:
                print(e)
                idx = random.randint(0, self.length-1)

        pixel_values = self.pixel_transforms(pixel_values)
        if self.control_video_folder is not None:
            sample = dict(pixel_values=pixel_values, text=name, controlnet_frames=controlnet_frames)
        else:
            sample = dict(pixel_values=pixel_values, text=name)
        return sample
    

class PreprocessDataset:
    def __init__(self, csv_path, control_video_folder, process_image, shape=(1024, 1024)):
        with open(csv_path, 'r') as csvfile:
            self.dataset = list(csv.DictReader(csvfile))
        self.control_video_folder = control_video_folder
        self.process_image = process_image
        self.shape = shape

    def preprocess_and_save(self, output_folder):
        os.makedirs(output_folder, exist_ok=True)
        for video_dict in tqdm(self.dataset):
            videoid = video_dict['videoid']
            
            control_videoid = video_dict.get('control_videoid', videoid)
            control_video_dir = os.path.join(self.control_video_folder, f"{control_videoid}.mp4")
            control_video_reader = VideoReader(control_video_dir)
            video_length = len(control_video_reader)

            control_pixel_values = control_video_reader.get_batch(range(video_length)).asnumpy()
            controlnet_frames = [Image.fromarray(img) for img in control_pixel_values]

            for processor in self.process_image:
                processor_id = processor.processor_id
                processed_frames = [processor(controlnet_frame) for controlnet_frame in controlnet_frames]
                print(f"Processed frames for processor {processor_id}: {len(processed_frames)}")

                processor_folder = os.path.join(output_folder, processor_id)
                os.makedirs(processor_folder, exist_ok=True)
                video_path = os.path.join(processor_folder, f"{videoid}.mp4")
                
                with imageio.get_writer(video_path,fps=30) as video:
                    for frame in processed_frames:
                        frame = np.array(frame.convert('RGB'))
                        video.append_data(frame)

            del control_video_reader