#!/bin/bash
export GPU_NUM=1
export MODEL_PATH="./Wan/Wan2.1-I2V-14B-720P"
export RIFE_PATH="./Scripts/RIFE/flownet.pkl"
export CUGAN_PATH="./Scripts/CUGAN/pro-conservative-up2x.pth"
export PROMPT_EXTEND=None
export IMAGE=$1
export LORA_PATH=$2
export OUTPUT=$3
export DIT_PATH=$4
export PROMPT=$5

export HEIGHT=720
export WIDTH=1280
export FPS=30
export NUM_FRAMES=81
export TASK="i2v"
export STEP=40
export SR_SIZE="1080P"

if [ ! -f "$LORA_PATH" ] && [ ! -d "$LORA_PATH" ]; then
    torchrun --standalone --nproc_per_node=$GPU_NUM Scripts/wan_test.py \
    --ckpt_path $DIT_PATH \
    --t5_model $MODEL_PATH"/models_t5_umt5-xxl-enc-bf16.pth" \
    --vae_model $MODEL_PATH"/Wan2.1_VAE.pth" \
    --clip_model $MODEL_PATH"/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
    --prompt "$PROMPT" \
    --output $OUTPUT \
    --steps $STEP \
    --task $TASK \
    --image $IMAGE \
    --height $HEIGHT \
    --width $WIDTH \
    --fps $FPS \
    --num_frames $NUM_FRAMES \
    --use_rife \
    --rife_model $RIFE_PATH \
    --use_cugan \
    --cugan_model $CUGAN_PATH \
    --sr_size $SR_SIZE \
    --prompt_extend_model $PROMPT_EXTEND
else
    torchrun --standalone --nproc_per_node=$GPU_NUM Scripts/wan_test.py \
    --ckpt_path $DIT_PATH \
    --lora_path $LORA_PATH \
    --t5_model $MODEL_PATH"/models_t5_umt5-xxl-enc-bf16.pth" \
    --vae_model $MODEL_PATH"/Wan2.1_VAE.pth" \
    --clip_model $MODEL_PATH"/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
    --prompt "$PROMPT" \
    --output $OUTPUT \
    --steps $STEP \
    --task $TASK \
    --image $IMAGE \
    --height $HEIGHT \
    --width $WIDTH \
    --fps $FPS \
    --num_frames $NUM_FRAMES \
    --use_rife \
    --rife_model $RIFE_PATH \
    --use_cugan \
    --cugan_model $CUGAN_PATH \
    --sr_size $SR_SIZE \
    --prompt_extend_model $PROMPT_EXTEND
fi




