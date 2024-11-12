import torch
import numpy as np
from PIL import Image

from .flownet_v1 import IFNet as IFNet_v1
from .flownet_v2 import IFNet as IFNet_v2
from .flownet_v3 import IFNet as IFNet_v3


class RIFEInterpolater:
    def __init__(self, model, device="cuda"):
        self.model = model
        self.device = device
        # IFNet only does not support float16
        self.torch_dtype = torch.float32

    @staticmethod
    def from_model_manager(model_manager):
        return RIFEInterpolater(model_manager.fetch_model("rife"), device=model_manager.device)

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

    @torch.no_grad()
    def interpolate_(self, images, scale=1.0):
        input_tensor = self.process_images(images)
        input_tensor = torch.cat((input_tensor[:-1], input_tensor[1:]), dim=1)
        input_tensor = input_tensor.to(device=self.device, dtype=self.torch_dtype)
        flow, mask, merged = self.model(input_tensor, [4/scale, 2/scale, 1/scale])
        output_images = self.decode_images(merged[2].cpu())
        if output_images[0].size != images[0].size:
            output_images = [image.resize(images[0].size) for image in output_images]
        return output_images

    @torch.no_grad()
    def interpolate(self, images, scale=1.0, batch_size=4, num_iter=1, progress_bar=lambda x:x):
        # Preprocess
        processed_images = self.process_images(images)

        for iter in range(num_iter):
            # Input
            input_tensor = torch.cat((processed_images[:-1], processed_images[1:]), dim=1)

            # Interpolate
            output_tensor = []
            for batch_id in progress_bar(range(0, input_tensor.shape[0], batch_size)):
                batch_id_ = min(batch_id + batch_size, input_tensor.shape[0])
                batch_input_tensor = input_tensor[batch_id: batch_id_]
                batch_input_tensor = batch_input_tensor.to(device=self.device, dtype=self.torch_dtype)
                flow, mask, merged = self.model(batch_input_tensor, [4/scale, 2/scale, 1/scale])
                output_tensor.append(merged[2].cpu())
            # Output
            output_tensor = torch.concat(output_tensor, dim=0).clip(0, 1)
            processed_images = self.add_interpolated_images(processed_images, output_tensor)
            processed_images = torch.stack(processed_images)

        # To images
        output_images = self.decode_images(processed_images)
        if output_images[0].size != images[0].size:
            output_images = [image.resize(images[0].size) for image in output_images]
        return output_images


class RIFESmoother(RIFEInterpolater):
    def __init__(self, model, device="cuda"):
        super(RIFESmoother, self).__init__(model, device=device)

    @staticmethod
    def from_model_manager(model_manager):
        return RIFEInterpolater(model_manager.fetch_model("rife"), device=model_manager.device)
    
    def process_tensors(self, input_tensor, scale=1.0, batch_size=4):
        output_tensor = []
        for batch_id in range(0, input_tensor.shape[0], batch_size):
            batch_id_ = min(batch_id + batch_size, input_tensor.shape[0])
            batch_input_tensor = input_tensor[batch_id: batch_id_]
            batch_input_tensor = batch_input_tensor.to(device=self.device, dtype=self.torch_dtype)
            flow, mask, merged = self.model(batch_input_tensor, [4/scale, 2/scale, 1/scale])
            output_tensor.append(merged[2].cpu())
        output_tensor = torch.concat(output_tensor, dim=0)
        return output_tensor

    @torch.no_grad()
    def __call__(self, rendered_frames, scale=1.0, batch_size=4, num_iter=1, **kwargs):
        # Preprocess
        processed_images = self.process_images(rendered_frames)

        for iter in range(num_iter):
            # Input
            input_tensor = torch.cat((processed_images[:-2], processed_images[2:]), dim=1)

            # Interpolate
            output_tensor = self.process_tensors(input_tensor, scale=scale, batch_size=batch_size)
            # Blend
            input_tensor = torch.cat((processed_images[1:-1], output_tensor), dim=1)
            output_tensor = self.process_tensors(input_tensor, scale=scale, batch_size=batch_size)

            # Add to frames
            processed_images[1:-1] = output_tensor

        # To images
        output_images = self.decode_images(processed_images)
        if output_images[0].size != rendered_frames[0].size:
            output_images = [image.resize(rendered_frames[0].size) for image in output_images]
        return output_images
