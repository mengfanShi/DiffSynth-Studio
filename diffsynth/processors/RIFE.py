import torch
import numpy as np
from PIL import Image
from .base import VideoProcessor


class RIFESmoother(VideoProcessor):
    def __init__(self, model, device="cuda", scale=1.0, batch_size=4, interpolate=False, blend=True, interpolate_times=0):
        self.model = model
        self.device = device

        # IFNet only does not support float16
        self.torch_dtype = torch.float32

        # Other parameters
        self.scale = scale
        self.batch_size = batch_size
        self.interpolate = interpolate
        self.interpolate_times = interpolate_times
        self.blend = blend

    @staticmethod
    def from_model_manager(model_manager, **kwargs):
        return RIFESmoother(model_manager.fetch_model("rife"), device=model_manager.device, **kwargs)

    def process_image(self, image):
        width, height = image.size
        if width % 32 != 0 or height % 32 != 0:
            width = (width + 31) // 32
            height = (height + 31) // 32
            image = image.resize((width, height))
        image = torch.Tensor(np.array(image, dtype=np.float32)[:, :, [2,1,0]] / 255).permute(2, 0, 1)
        return image

    def process_images(self, images):
        images = [self.process_image(image) for image in images]
        images = torch.stack(images)
        return images

    def decode_images(self, images):
        images = (images[:, [2,1,0]].permute(0, 2, 3, 1) * 255).clip(0, 255).numpy().astype(np.uint8)
        images = [Image.fromarray(image) for image in images]
        return images

    def add_interpolated_images(self, images, interpolated_images):
        output_images = []
        for image, interpolated_image in zip(images, interpolated_images):
            output_images.append(image)
            output_images.append(interpolated_image)
        output_images.append(images[-1])
        return output_images

    def add_interpolated_N_images(self, images_res):
        output_images = []
        for i in range(len(images_res[-1])):
            for j in range(len(images_res)):
                output_images.append(images_res[j][i])
        output_images.append(images_res[0][-1])
        return output_images

    def process_tensors(self, input_tensor, scale=1.0, batch_size=4):
        output_tensor = []
        for batch_id in range(0, input_tensor.shape[0], batch_size):
            batch_id_ = min(batch_id + batch_size, input_tensor.shape[0])
            batch_input_tensor = input_tensor[batch_id: batch_id_]
            batch_input_tensor = batch_input_tensor.to(device=self.device, dtype=self.torch_dtype)
            # flow, mask, merged = self.model(batch_input_tensor, [4/scale, 2/scale, 1/scale])
            # output_tensor.append(merged[2].cpu())
            flow, mask, merged = self.model(batch_input_tensor, scale=[8/scale, 4/scale, 2/scale, 1/scale])
            output_tensor.append(merged[3].cpu())
        output_tensor = torch.concat(output_tensor, dim=0)
        return output_tensor

    @torch.no_grad()
    def __call__(self, rendered_frames, **kwargs):
        # Preprocess
        processed_images = self.process_images(rendered_frames)

        if self.interpolate:
            input_tensor = torch.cat((processed_images[:-1], processed_images[1:]), dim=1)
            # Interpolate
            output_tensor = self.process_tensors(input_tensor, scale=self.scale, batch_size=self.batch_size)
            result = [processed_images, output_tensor]
            if self.interpolate_times > 1:
                output_1 = output_tensor
                output_2 = output_tensor
                for i in range(self.interpolate_times - 1):
                    if i % 2 == 0:
                        input_tensor = torch.cat((processed_images[:-1], output_1), dim=1)
                        output_1 = self.process_tensors(input_tensor, scale=self.scale, batch_size=self.batch_size)
                        result.insert(1, output_1)
                    else:
                        input_tensor = torch.cat((output_2, processed_images[1:]), dim=1)
                        output_2 = self.process_tensors(input_tensor, scale=self.scale, batch_size=self.batch_size)
                        result.append(output_2)
            processed_images = self.add_interpolated_N_images(result)
            processed_images = torch.stack(processed_images)
        else:
            input_tensor = torch.cat((processed_images[:-2], processed_images[2:]), dim=1)
            output_tensor = self.process_tensors(input_tensor, scale=self.scale, batch_size=self.batch_size)
            if self.blend:
                input_tensor = torch.cat((processed_images[1:-1], output_tensor), dim=1)
                output_tensor = self.process_tensors(input_tensor, scale=self.scale, batch_size=self.batch_size)
                processed_images[1:-1] = output_tensor
            else:
                processed_images[1:-1] = (processed_images[1:-1] + output_tensor) / 2

        # To images
        output_images = self.decode_images(processed_images)
        if output_images[0].size != rendered_frames[0].size:
            output_images = [image.resize(rendered_frames[0].size) for image in output_images]
        return output_images
