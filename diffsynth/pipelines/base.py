import torch
import numpy as np
from PIL import Image
from ..models.utils import load_state_dict
from ..models.sd_lora import SDLoRA

def is_sd_lora(state_dict):
    return "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.lora_down.weight" in state_dict


class BasePipeline(torch.nn.Module):

    def __init__(self, device="cuda", torch_dtype=torch.float16):
        super().__init__()
        self.device = device
        self.torch_dtype = torch_dtype
        self.cpu_offload = False
        self.model_names = []

    def preprocess_image(self, image):
        image = torch.Tensor(np.array(image, dtype=np.float32) * (2 / 255) - 1).permute(2, 0, 1).unsqueeze(0)
        return image

    def preprocess_images(self, images):
        return [self.preprocess_image(image) for image in images]

    def vae_output_to_image(self, vae_output):
        image = vae_output[0].cpu().float().permute(1, 2, 0).numpy()
        image = Image.fromarray(((image / 2 + 0.5).clip(0, 1) * 255).astype("uint8"))
        return image

    def vae_output_to_video(self, vae_output):
        video = vae_output.cpu().permute(1, 2, 0).numpy()
        video = [Image.fromarray(((image / 2 + 0.5).clip(0, 1) * 255).astype("uint8")) for image in video]
        return video

    def merge_latents(self, value, latents, masks, scales):
        height, width = value.shape[-2:]
        weight = torch.ones_like(value)
        for latent, mask, scale in zip(latents, masks, scales):
            mask = self.preprocess_image(mask.resize((width, height))).mean(dim=1, keepdim=True) > 0
            mask = mask.repeat(1, latent.shape[1], 1, 1)
            value[mask] += latent[mask] * scale
            weight[mask] += scale
        value /= weight
        return value

    def control_noise_via_local_prompts(self, prompt_emb_global, prompt_emb_locals, masks, mask_scales, inference_callback, special_kwargs=None, special_local_kwargs_list=None):
        if special_kwargs is None:
            noise_pred_global = inference_callback(prompt_emb_global)
        else:
            noise_pred_global = inference_callback(prompt_emb_global, special_kwargs)
        if special_local_kwargs_list is None:
            noise_pred_locals = [inference_callback(prompt_emb_local) for prompt_emb_local in prompt_emb_locals]
        else:
            noise_pred_locals = [inference_callback(prompt_emb_local, special_kwargs) for prompt_emb_local, special_kwargs in zip(prompt_emb_locals, special_local_kwargs_list)]
        noise_pred = self.merge_latents(noise_pred_global, noise_pred_locals, masks, mask_scales)
        return noise_pred

    def extend_prompt(self, prompt, local_prompts, masks, mask_scales):
        local_prompts = local_prompts or []
        masks = masks or []
        mask_scales = mask_scales or []
        extended_prompt_dict = self.prompter.extend_prompt(prompt)
        prompt = extended_prompt_dict.get("prompt", prompt)
        local_prompts += extended_prompt_dict.get("prompts", [])
        masks += extended_prompt_dict.get("masks", [])
        mask_scales += [100.0] * len(extended_prompt_dict.get("masks", []))
        return prompt, local_prompts, masks, mask_scales

    def enable_cpu_offload(self):
        self.cpu_offload = True

    def load_models_to_device(self, loadmodel_names=[]):
        # only load models to device if cpu_offload is enabled
        if not self.cpu_offload:
            return
        # offload the unneeded models to cpu
        for model_name in self.model_names:
            if model_name not in loadmodel_names:
                model = getattr(self, model_name)
                if model is not None:
                    model.cpu()
        # load the needed models to device
        for model_name in loadmodel_names:
            model = getattr(self, model_name)
            if model is not None:
                model.to(self.device)
        # fresh the cuda cache
        torch.cuda.empty_cache()

    def load_sd_lora(self, loras, alphas):
        for lora,alpha in zip(loras, alphas):
            state_dict = load_state_dict(lora)

            convert_diffusers=is_sd_lora(state_dict)
            SDLoRA().add_lora_to_text_encoder(self.text_encoder, state_dict, alpha=alpha, device=self.device)
            SDLoRA().add_lora_to_unet(self.unet, state_dict, alpha=alpha, device=self.device, convert_diffusers=convert_diffusers)
            if hasattr(self, "motion_modules"):
                SDLoRA().add_lora_to_motion_module(self.motion_modules, state_dict, alpha=alpha, device=self.device, convert_diffusers=convert_diffusers)

    def generate_noise(self, shape, seed=None, device="cpu", dtype=torch.float16):
        generator = None if seed is None else torch.Generator(device).manual_seed(seed)
        noise = torch.randn(shape, generator=generator, device=device, dtype=dtype)
        return noise
