import lightning as pl
from peft import LoraConfig, inject_adapter_in_model
import torch
from ..data.simple_text_image import TextVideoDataset
from ..pipelines.dancer import lets_dance
from einops import rearrange


class LightningModelForT2VLoRA(pl.LightningModule):
    def __init__(
        self,
        learning_rate=1e-4,
        use_gradient_checkpointing=True,
        state_dict_converter=None,
        mm_lora = False,
        lr_type = None,
    ):
        super().__init__()
        # Set parameters
        self.learning_rate = learning_rate
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.state_dict_converter = state_dict_converter
        self.mm_lora = mm_lora
        self.lr_type = lr_type

    def load_models(self):
        # This function is implemented in other modules
        self.pipe = None

    def freeze_parameters(self):
        # Freeze parameters
        self.pipe.requires_grad_(False)
        self.pipe.eval()
        self.pipe.denoising_model().train()
    
    def add_lora_to_model(self, model, lora_rank=4, lora_alpha=4, lora_dropout=0.05, lora_target_modules="to_q,to_k,to_v,to_out"):
        # Add LoRA to UNet
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights="gaussian",
            target_modules=lora_target_modules.split(","),
        )
        model = inject_adapter_in_model(lora_config, model)
        for param in model.parameters():
            # Upcast LoRA parameters into fp32
            if param.requires_grad:
                param.data = param.to(torch.float32)

    def training_step(self, batch, batch_idx):
        # Data
        text, images = batch["text"], batch["pixel_values"]
        controlnet_frames = batch.get("controlnet_frames", None)

        if images.dim() == 5:
            # images = images.squeeze(0)
            batch_size = images.shape[0]
            images = rearrange(images, "b f c h w -> (b f) c h w")
            if controlnet_frames is not None:
                controlnet_frames = rearrange(controlnet_frames, "b d f c h w -> d (b f) c h w")
           
        # Prepare input parameters
        self.pipe.device = self.device
        with torch.no_grad():
            prompt_emb = self.pipe.encode_prompt_batch(text, positive=True)
            
            latents = self.pipe.vae_encoder(images.to(dtype=self.pipe.torch_dtype, device=self.device))
            noise = torch.randn_like(latents)
            timestep_id = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (1,))
            timestep = self.pipe.scheduler.timesteps[timestep_id].to(self.device)
            extra_input = self.pipe.prepare_extra_input(latents)

            noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timestep)
            training_target = self.pipe.scheduler.training_target(latents, noise, timestep)

        noise_pred = lets_dance(self.pipe.denoising_model(), self.pipe.motion_modules, self.pipe.controlnet,noisy_latents,timestep,**prompt_emb, controlnet_frames=controlnet_frames, batch_size=batch_size)
        loss = torch.nn.functional.mse_loss(noise_pred, training_target)

        # Record log
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        trainable_modules = list(filter(lambda p: p.requires_grad, self.pipe.denoising_model().parameters()))
        if self.mm_lora:
            trainable_modules += list(filter(lambda p: p.requires_grad, self.pipe.motion_modules.parameters()))
        
        optimizer = torch.optim.AdamW(trainable_modules, lr=self.learning_rate)
        if self.lr_type == "one_cycle":
            print("Using OneCycleLR")
            stepping_batches = self.trainer.estimated_stepping_batches
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, total_steps=stepping_batches)
            
            return {
                    "optimizer": optimizer,
                    "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
                    }
        
        if self.lr_type == "cosine":
            print("Using CosineAnnealingLR")
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
            return {
                    "optimizer": optimizer,
                    "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
                    }
        else:
            return optimizer

    def on_save_checkpoint(self, checkpoint):
        checkpoint.clear()
        trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.pipe.denoising_model().named_parameters()))
        trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
        state_dict = self.pipe.denoising_model().state_dict()
        lora_state_dict = {}
        for name, param in state_dict.items():
            if name in trainable_param_names:
                lora_state_dict[name] = param

        if self.mm_lora:
            motion_trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.pipe.motion_modules.named_parameters()))
            motion_trainable_param_names = set([named_param[0] for named_param in motion_trainable_param_names])
            motion_state_dict = self.pipe.motion_modules.state_dict()
            for name, param in motion_state_dict.items():
                if name in motion_trainable_param_names:
                    lora_state_dict[name] = param
        
        if self.state_dict_converter is not None:
            lora_state_dict = self.state_dict_converter(lora_state_dict)
        checkpoint.update(lora_state_dict)


def add_general_parsers(parser):
    parser.add_argument(
        "--csv_path",
        type=str,
        default=None,
        required=True,
        help="The path of the csv file containing the dataset."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        required=True,
        help="The path of the Dataset.",
    )
    parser.add_argument(
        "--control_video_folder",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./",
        help="Path to save the model.",
    )
    parser.add_argument(
        "--sample_stride",
        type=int,
        default=3,
        help="Number of frames to skip when sampling.",
    )
    parser.add_argument(
        "--sample_n_frames",
        type=int,
        default=16,
        help="Number of frames to sample.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=256,
        help="Image height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=384,
        help="Image width.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="16-mixed",
        choices=["32", "16", "16-mixed", "bf16"],
        help="Training precision",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=8e-5,
        help="Learning rate.",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=8,
        help="The dimension of the LoRA update matrices.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=16.0,
        help="The weight of the LoRA update matrices.",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="The weight of the LoRA update matrices.",
    )
    parser.add_argument(
        "--use_gradient_checkpointing",
        default=False,
        action="store_true",
        help="Whether to use gradient checkpointing.",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="The number of batches in gradient accumulation.",
    )
    parser.add_argument(
        "--training_strategy",
        type=str,
        default="auto",
        choices=["auto", "deepspeed_stage_1", "deepspeed_stage_2", "deepspeed_stage_3"],
        help="Training strategy",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=100,
        help="Number of epochs.",
    )
    parser.add_argument(
        "--mm_lora",
        default=False,
        action="store_true",
        help="Whether to use mm_lora.",
    )
    parser.add_argument(
        "--lr_type",
        default=None,
        type=str,
        help="which use cosine learning rate.",
    )
    parser.add_argument(
        "--extra_prompts",
        default=None,
        type=str,
        help="extra prompts.",
    )
    
    return parser


def launch_training_task(model, args):
    # dataset and data loader
    dataset = TextVideoDataset(
        args.csv_path,
        args.dataset_path,
        height=args.height,
        width=args.width,
        sample_stride=args.sample_stride, sample_n_frames=args.sample_n_frames,
        control_video_folder=args.control_video_folder,
        process_image = model.pipe.controlnet.process_image,
        processors_id = [processor.processor_id for processor in model.pipe.controlnet.processors],
        extra_prompts=args.extra_prompts,
    )
    train_loader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.dataloader_num_workers
    )

    model.save_hyperparameters(args)

    # train
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices="auto",
        precision=args.precision,
        strategy=args.training_strategy,
        default_root_dir=args.output_path,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[pl.pytorch.callbacks.ModelCheckpoint(save_top_k=-1)]
    )
    trainer.fit(model=model, train_dataloaders=train_loader)

    