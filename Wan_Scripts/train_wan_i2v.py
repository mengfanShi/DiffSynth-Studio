import torch, os, argparse
import lightning as pl
from diffsynth import WanVideoPipeline, ModelManager, load_state_dict
from peft import LoraConfig, inject_adapter_in_model
from deepspeed.ops.adam import DeepSpeedCPUAdam
from dataset import EnhancedBucketBatchSampler, BucketAwareDataset, data_process


def expand_paths(input_paths):
    expanded = []
    if input_paths is None:
        return None
    for path in input_paths:
        if os.path.isdir(path):
            for f in os.listdir(path):
                if f.endswith(".safetensors"):
                    expanded.append(os.path.join(path, f))
        else:
            expanded.append(path)
    return expanded if len(expanded) > 0 else None


def load_state_dict_from_folder(file_path, torch_dtype=None):
    state_dict = {}
    for file_name in os.listdir(file_path):
        if "." in file_name and file_name.split(".")[-1] in [
            "safetensors",
            "bin",
            "ckpt",
            "pth",
            "pt",
        ]:
            state_dict.update(
                load_state_dict(
                    os.path.join(file_path, file_name), torch_dtype=torch_dtype
                )
            )
    return state_dict


class LightningModelForTrain(pl.LightningModule):
    def __init__(
        self,
        dit_path,
        learning_rate=1e-5,
        lora_rank=4,
        lora_alpha=4,
        train_architecture="lora",
        lora_target_modules="q,k,v,o,ffn.0,ffn.2",
        init_lora_weights="kaiming",
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        train_caption_model="qwen2-VL-72B-detail",
        lora_path=None,
        pretrained_lora_path=None,
        deepspeed_offload=False,
        cache_path=None,
        batch_size=1,
        steps_per_epoch=500,
        warmup_steps=1,
        dataloader_num_workers=1,
        use_tail_frame=False,
        use_control=False,
        use_audio=False,
        num_frames=81,
        seed=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        model_manager.load_models([dit_path])
        if lora_path is not None:
            model_manager.load_lora(lora_path, lora_alpha=1.0)

        self.pipe = WanVideoPipeline.from_model_manager(model_manager)
        self.pipe.scheduler.set_timesteps(1000, training=True)
        self.freeze_parameters()
        if train_architecture == "lora":
            self.add_lora_to_model(
                self.pipe.denoising_model(),
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_target_modules=lora_target_modules,
                init_lora_weights=init_lora_weights,
                pretrained_lora_path=pretrained_lora_path,
            )
        else:
            self.pipe.denoising_model().requires_grad_(True)

        self.learning_rate = learning_rate
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.train_caption_model = train_caption_model
        self.deepspeed_offload = deepspeed_offload
        self.cache_path = cache_path
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.warmup_steps = warmup_steps
        self.dataloader_num_workers = dataloader_num_workers
        self.use_tail_frame = use_tail_frame
        self.use_control = use_control
        self.use_audio = use_audio
        self.num_frames = num_frames
        self.seed = seed

    def freeze_parameters(self):
        # Freeze parameters
        self.pipe.requires_grad_(False)
        self.pipe.eval()
        self.pipe.denoising_model().train()

    def add_lora_to_model(
        self,
        model,
        lora_rank=4,
        lora_alpha=4,
        lora_target_modules="q,k,v,o,ffn.0,ffn.2",
        init_lora_weights="kaiming",
        pretrained_lora_path=None,
        state_dict_converter=None,
    ):
        # Add LoRA to UNet
        self.lora_alpha = lora_alpha
        if init_lora_weights == "kaiming":
            init_lora_weights = True

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            init_lora_weights=init_lora_weights,
            target_modules=lora_target_modules.split(","),
        )
        model = inject_adapter_in_model(lora_config, model)
        for param in model.parameters():
            # Upcast LoRA parameters into fp32
            if param.requires_grad:
                param.data = param.to(torch.float32)

        # Lora pretrained lora weights
        if pretrained_lora_path is not None:
            if os.path.isdir(pretrained_lora_path):
                state_dict = load_state_dict_from_folder(pretrained_lora_path)
            else:
                state_dict = load_state_dict(pretrained_lora_path)

            if state_dict_converter is not None:
                state_dict = state_dict_converter(state_dict)
            missing_keys, unexpected_keys = model.load_state_dict(
                state_dict, strict=False
            )
            all_keys = [i for i, _ in model.named_parameters()]
            num_updated_keys = len(all_keys) - len(missing_keys)
            num_unexpected_keys = len(unexpected_keys)
            print(
                f"{num_updated_keys} parameters are loaded from {pretrained_lora_path}. {num_unexpected_keys} parameters are unexpected."
            )

    def training_step(self, batch, batch_idx):
        # Data
        latents = batch["latents"].to(self.device)
        prompt_emb = batch["prompt_emb"][self.train_caption_model]
        prompt_emb["context"] = prompt_emb["context"][:, 0].to(self.device)
        clip_feature = batch["clip_feature"].to(self.device)
        audio_feature = batch["audio_feature"].to(self.device)
        y = (
            batch["y"].to(self.device)
            if not self.use_tail_frame
            else batch["y_tail"].to(self.device)
        )

        # TODO: add control and audio feature
        if self.use_control:
            control = batch["control"].to(self.device)
            y = torch.concat([control, y], dim=1)

        if not self.use_tail_frame:
            # Whether equals num_frames
            vae_frames = (self.num_frames + 3) // 4
            if latents.shape[2] != vae_frames:
                latents = latents[:, :, :vae_frames]
                y = y[:, :, :vae_frames]
                audio_feature = audio_feature[:, :vae_frames]

        if not self.use_audio:
            audio_feature = None

        # Loss
        self.pipe.device = self.device
        noise = torch.randn_like(latents)
        timestep_id = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (1,))
        timestep = self.pipe.scheduler.timesteps[timestep_id].to(
            dtype=self.pipe.torch_dtype, device=self.pipe.device
        )
        extra_input = self.pipe.prepare_extra_input(latents)
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timestep)
        training_target = self.pipe.scheduler.training_target(latents, noise, timestep)

        # Compute loss
        noise_pred = self.pipe.denoising_model()(
            noisy_latents,
            timestep=timestep,
            **prompt_emb,
            **extra_input,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
            use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload,
            clip_feature=clip_feature,
            y=y,
        )
        loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
        loss = loss * self.pipe.scheduler.training_weight(timestep)

        # Record log
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        trainable_modules = filter(
            lambda p: p.requires_grad, self.pipe.denoising_model().parameters()
        )
        if self.deepspeed_offload:
            optimizer = DeepSpeedCPUAdam(trainable_modules, lr=self.learning_rate)
        else:
            optimizer = torch.optim.AdamW(trainable_modules, lr=self.learning_rate)

        if self.warmup_steps > 1:
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lambda step: min(step / self.warmup_steps, 1.0)
            )
            return [optimizer], [scheduler]
        return optimizer

    def on_save_checkpoint(self, checkpoint):
        checkpoint.clear()
        trainable_param_names = list(
            filter(
                lambda named_param: named_param[1].requires_grad,
                self.pipe.denoising_model().named_parameters(),
            )
        )
        trainable_param_names = set(
            [named_param[0] for named_param in trainable_param_names]
        )
        state_dict = self.pipe.denoising_model().state_dict()
        lora_state_dict = {}
        for name, param in state_dict.items():
            if name in trainable_param_names:
                lora_state_dict[name] = param
        checkpoint.update(lora_state_dict)

    def train_dataloader(self):
        dataset = BucketAwareDataset(
            cache_root=self.cache_path, steps_per_epoch=self.steps_per_epoch
        )
        bucket_names = dataset.get_bucket_names()
        sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)

        batch_sampler = EnhancedBucketBatchSampler(
            sampler=sampler,
            dataset=dataset,
            batch_size=self.batch_size,
            bucket_names=bucket_names,
            drop_last=True,
            seed=self.seed,
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=self.dataloader_num_workers,
            pin_memory=args.pin_memory,
            shuffle=False,
        )

        return dataloader


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--task",
        type=str,
        default="data_process",
        required=True,
        choices=["data_process", "train"],
        help="Task. `data_process` or `train`.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        required=False,
        help="The path of the Dataset.",
    )
    parser.add_argument(
        "--dataset_file",
        type=str,
        default=None,
        required=True,
        help="The path of the Dataset File.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./",
        help="Path to save the model.",
    )
    parser.add_argument(
        "--text_encoder_path",
        type=str,
        default=None,
        help="Path of text encoder.",
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default=None,
        help="Path of image encoder.",
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default=None,
        help="Path of VAE.",
    )
    parser.add_argument(
        "--dit_path",
        type=str,
        default=None,
        nargs="+",
        help="Path of DiT.",
    )
    parser.add_argument(
        "--wav2vec_model_path",
        type=str,
        default=None,
        help="Path of wav2vec model.",
    )
    parser.add_argument(
        "--audio_separator_model_path",
        type=str,
        default=None,
        help="Path of audio separator model.",
    )
    parser.add_argument(
        "--tiled",
        default=False,
        action="store_true",
        help="Whether enable tile encode in VAE. This option can reduce VRAM required.",
    )
    parser.add_argument(
        "--tile_size_height",
        type=int,
        default=34,
        help="Tile size (height) in VAE.",
    )
    parser.add_argument(
        "--tile_size_width",
        type=int,
        default=34,
        help="Tile size (width) in VAE.",
    )
    parser.add_argument(
        "--tile_stride_height",
        type=int,
        default=18,
        help="Tile stride (height) in VAE.",
    )
    parser.add_argument(
        "--tile_stride_width",
        type=int,
        default=16,
        help="Tile stride (width) in VAE.",
    )
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=500,
        help="Number of steps per epoch.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=81,
        help="Number of frames.",
    )
    parser.add_argument(
        "--frame_interval",
        type=int,
        default=1,
        help="Frame interval.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Image height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=832,
        help="Image width.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=1,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate.",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="The number of batches in gradient accumulation.",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=1,
        help="Number of epochs.",
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q,k,v,o,ffn.0,ffn.2",
        help="Layers with LoRA modules.",
    )
    parser.add_argument(
        "--init_lora_weights",
        type=str,
        default="kaiming",
        choices=["gaussian", "kaiming"],
        help="The initializing method of LoRA weight.",
    )
    parser.add_argument(
        "--training_strategy",
        type=str,
        default="auto",
        choices=[
            "auto",
            "deepspeed_stage_1",
            "deepspeed_stage_2",
            "deepspeed_stage_2_offload",
            "deepspeed_stage_3",
            "deepspeed_stage_3_offload",
        ],
        help="Training strategy",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=4,
        help="The dimension of the LoRA update matrices.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=4.0,
        help="The weight of the LoRA update matrices.",
    )
    parser.add_argument(
        "--use_gradient_checkpointing",
        default=False,
        action="store_true",
        help="Whether to use gradient checkpointing.",
    )
    parser.add_argument(
        "--use_gradient_checkpointing_offload",
        default=False,
        action="store_true",
        help="Whether to use gradient checkpointing offload.",
    )
    parser.add_argument(
        "--pin_memory",
        default=False,
        action="store_true",
        help="Whether to pin memory.",
    )
    parser.add_argument(
        "--train_architecture",
        type=str,
        default="lora",
        choices=["lora", "full"],
        help="Model structure to train. LoRA training or full training.",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        nargs="+",
        help="Pretrained LoRA path. Required if add on base model.",
    )
    parser.add_argument(
        "--pretrained_lora_path",
        type=str,
        default=None,
        nargs="+",
        help="Pretrained LoRA path. Required if the training is resumed.",
    )
    parser.add_argument(
        "--use_swanlab",
        default=False,
        action="store_true",
        help="Whether to use SwanLab logger.",
    )
    parser.add_argument(
        "--swanlab_mode",
        default=None,
        help="SwanLab mode (cloud or local).",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16",
        help="Precision.",
    )
    parser.add_argument(
        "--enable_bucket",
        default=False,
        action="store_true",
        help="Whether to enable bucket.",
    )
    parser.add_argument(
        "--cache_path",
        type=str,
        nargs="+",
        required=True,
        help="The path of the Dataset.",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")
    parser.add_argument("--warmup_steps", type=int, default=1, help="Warmup steps.")
    parser.add_argument(
        "--use_tail_frame",
        default=False,
        action="store_true",
        help="Whether to use tail frame.",
    )
    parser.add_argument(
        "--use_control",
        default=False,
        action="store_true",
        help="Whether to use control.",
    )
    parser.add_argument(
        "--use_audio", default=False, action="store_true", help="Whether to use audio."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=-1,
        help="Save top k checkpoints. -1 means save all checkpoints.",
    )
    parser.add_argument(
        "--resume_path",
        type=str,
        default=None,
        help="Path to resume training.",
    )
    parser.add_argument(
        "--caption_model",
        type=str,
        nargs="+",
        default=[
            "qwen2-VL-72B-detail",
            "qwen2-VL-72B-detail-cn",
            "qwen2-VL-72B-short",
            "qwen2-VL-72B-short-cn",
        ],
        help="Caption model when data process.",
    )
    parser.add_argument(
        "--train_caption_model",
        type=str,
        default="qwen2-VL-72B-detail",
        choices=[
            "qwen2-VL-72B-detail",
            "qwen2-VL-72B-detail-cn",
            "qwen2-VL-72B-short",
            "qwen2-VL-72B-short-cn",
        ],
        help="Caption model when train.",
    )
    parser.add_argument(
        "--nnodes",
        type=int,
        default=1,
        help="num nodes for ddp training",
    )
    args = parser.parse_args()
    return args


def train(args):
    model = LightningModelForTrain(
        dit_path=args.dit_path,
        learning_rate=args.learning_rate,
        train_architecture=args.train_architecture,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_target_modules=args.lora_target_modules,
        init_lora_weights=args.init_lora_weights,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        train_caption_model=args.train_caption_model,
        lora_path=args.lora_path,
        pretrained_lora_path=args.pretrained_lora_path,
        deepspeed_offload="offload" in args.training_strategy,
        batch_size=args.batch_size,
        use_tail_frame=args.use_tail_frame,
        use_control=args.use_control,
        use_audio=args.use_audio,
        dataloader_num_workers=args.dataloader_num_workers,
        steps_per_epoch=args.steps_per_epoch,
        warmup_steps=args.warmup_steps,
        cache_path=args.cache_path,
        num_frames=args.num_frames,
        seed=args.seed,
    )

    if args.use_swanlab:
        from swanlab.integration.pytorch_lightning import SwanLabLogger

        swanlab_config = {"UPPERFRAMEWORK": "DiffSynth-Studio"}
        swanlab_config.update(vars(args))
        swanlab_logger = SwanLabLogger(
            project="wan",
            name="wan",
            config=swanlab_config,
            mode=args.swanlab_mode,
            logdir=os.path.join(args.output_path, "swanlog"),
        )
        logger = [swanlab_logger]
    else:
        logger = None

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices="auto",
        strategy=args.training_strategy,
        default_root_dir=args.output_path,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[
            pl.pytorch.callbacks.ModelCheckpoint(
                monitor="train_loss", save_top_k=args.top_k, mode="min"
            )
        ],
        logger=logger,
        precision=args.precision,
        num_nodes=args.nnodes,
        use_distributed_sampler=False,
    )
    trainer.fit(model, ckpt_path=args.resume_path)


if __name__ == "__main__":
    args = parse_args()
    args.dit_path = expand_paths(args.dit_path) or args.dit_path
    args.lora_path = expand_paths(args.lora_path) or args.lora_path

    if args.task == "data_process":
        data_process(args)
    elif args.task == "train":
        train(args)
