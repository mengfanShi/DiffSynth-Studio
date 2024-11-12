#!/bin/bash

wget -c -O "models/stable_diffusion/aingdiffusion_v16.safetensors" "https://civitai.com/api/download/models/327677"
wget -c -O "models/AnimateDiff/mm_sd_v15_v3.ckpt" "https://huggingface.co/guoyww/animatediff/resolve/main/v3_sd15_mm.ckpt"
wget -c -O "models/ControlNet/control_v11p_sd15_lineart.pth" "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_lineart.pth"
wget -c -O "models/ControlNet/control_v11f1e_sd15_tile.pth" "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1e_sd15_tile.pth"
wget -c -O "models/ControlNet/control_v11f1p_sd15_depth.pth" "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1p_sd15_depth.pth"
wget -c -O "models/ControlNet/control_v11p_sd15_softedge.pth" "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_softedge.pth"
wget -c -O "models/Annotators/dpt_hybrid-midas-501f0c75.pt" "https://huggingface.co/lllyasviel/Annotators/resolve/main/dpt_hybrid-midas-501f0c75.pt"
wget -c -O "models/Annotators/ControlNetHED.pth" "https://huggingface.co/lllyasviel/Annotators/resolve/main/ControlNetHED.pth"
wget -c -O "models/Annotators/sk_model.pth" "https://huggingface.co/lllyasviel/Annotators/resolve/main/sk_model.pth"
wget -c -O "models/Annotators/sk_model2.pth" "https://huggingface.co/lllyasviel/Annotators/resolve/main/sk_model2.pth"
wget -c -O "models/textual_inversion/verybadimagenegative_v1.3.pt" "https://civitai.com/api/download/models/25820?type=Model&format=PickleTensor&size=full&fp=fp16"
