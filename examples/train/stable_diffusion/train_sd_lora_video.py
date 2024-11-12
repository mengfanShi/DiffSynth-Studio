from diffsynth import ModelManager, SDVideoPipeline
from diffsynth.trainers.text_to_video import LightningModelForT2VLoRA, add_general_parsers, launch_training_task
import torch, os, argparse
from examples.Diffutoon.diffutoon_toon_shading import config
from collections import OrderedDict
from diffsynth.controlnets import ControlNetConfigUnit

os.environ["TOKENIZERS_PARALLELISM"] = "True"


def prepare_lora_state_dict(unet_lora_state_dict, with_mm=True):
    lora_state_dict = OrderedDict()
    for key in unet_lora_state_dict:
        new_key = key.replace("base_model.model.", "").replace("lora_A", "lora_down").replace("lora_B", "lora_up")
        if "motion_modules" in key:
            if not with_mm:
                continue
            new_key = "lora_mm_" + convert_to_lora_format(new_key)
        else:
            new_key = "lora_unet_" + convert_to_lora_format(new_key)

        lora_state_dict[new_key] = unet_lora_state_dict[key]

    return lora_state_dict


def convert_to_lora_format(key):
    if ".lora" in key:
        parts = key.split(".lora")
        return parts[0].replace(".", "_") + ".lora" + parts[1]
    else:
        parts = key.split(".")
        return "_".join(parts[:-1]) + "." + parts[-1]


class LightningModel(LightningModelForT2VLoRA):
    def __init__(
        self,
        config,
        torch_dtype=torch.float16,
        learning_rate=1e-4, use_gradient_checkpointing=True,
        lora_rank=4, lora_alpha=4, lora_dropout=0.05, lora_target_modules="to_q,to_k,to_v,to_out",mm_lora=False,lr_type=None
    ):
        super().__init__(learning_rate=learning_rate, use_gradient_checkpointing=use_gradient_checkpointing, mm_lora=mm_lora, lr_type=lr_type)

        self.config = config
        model_manager = ModelManager(torch_dtype=torch_dtype, device=self.device)
        model_manager.load_models(config['model_list'])
        self.pipe = SDVideoPipeline.from_model_manager(
            model_manager,
            [
                ControlNetConfigUnit(
                    processor_id=unit["processor_id"],
                    model_path=unit["model_path"],
                    scale=unit["scale"]
                ) for unit in config['controlnet_units']
            ]
        )
        
        self.pipe.scheduler.set_timesteps(1000)

        self.freeze_parameters()
        self.add_lora_to_model(self.pipe.denoising_model(), lora_rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout, lora_target_modules=lora_target_modules)
        if self.mm_lora:
            self.add_lora_to_model(self.pipe.motion_modules, lora_rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout, lora_target_modules=lora_target_modules)

        self.state_dict_converter = prepare_lora_state_dict

        self.save_hyperparameters(config)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model. For example, `models/stable_diffusion/v1-5-pruned-emaonly.safetensors`.",
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="to_q,to_k,to_v",
        help="Layers with LoRA modules.",
    )
    parser = add_general_parsers(parser)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    train_config = {}
    train_config["models"] = config["models"].copy()
    train_config["models"]["model_list"][0] = args.pretrained_path

    model = LightningModel(
        train_config["models"],
        torch_dtype=torch.float32 if args.precision == "32" else torch.float16,
        learning_rate=args.learning_rate,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        mm_lora=args.mm_lora,
        lr_type=args.lr_type,
    )
    launch_training_task(model, args)