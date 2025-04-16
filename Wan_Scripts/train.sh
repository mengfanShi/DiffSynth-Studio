#!/bin/bash
export NODE_RANK=$1
export NCCL_TIMEOUT=7200
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=9999

export GPU_NUM=8
export NUM_NODES=2
export OMP_NUM_THREADS=4
export MODEL_PATH="./Wan"
export DIT_PATH="./Wan"
export DATA_PATH="./data"
export TRAIN_TYPE="lora"
export TASK="train"
export OUTPUT="./output"
export CACHE_PATH="./bucket_cache"
export CAPTION_MODEL="qwen2-VL-72B-detail"
export WAV2VEC_MODEL=None
export AUDIO_SEPARATOR_MODEL=None

export HEIGHT=720
export WIDTH=1280
export NUM_FRAMES=81
export BATCH_SIZE=1
export FRAME_INTERVAL=1

torchrun --nproc_per_node=$GPU_NUM --nnodes=$NUM_NODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT Scripts/train_wan_i2v.py \
    --task $TASK \
    --dataset_file $DATA_PATH \
    --output_path $OUTPUT \
    --cache_path $CACHE_PATH \
    --dit_path $DIT_PATH \
    --text_encoder_path $MODEL_PATH"/models_t5_umt5-xxl-enc-bf16.pth" \
    --image_encoder_path $MODEL_PATH"/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
    --vae_path $MODEL_PATH"/Wan2.1_VAE.pth" \
    --train_caption_model $CAPTION_MODEL \
    --wav2vec_model_path $WAV2VEC_MODEL \
    --audio_separator_model_path $AUDIO_SEPARATOR_MODEL \
    --tiled \
    --num_frames $NUM_FRAMES \
    --frame_interval $FRAME_INTERVAL \
    --batch_size $BATCH_SIZE \
    --steps_per_epoch 10000 \
    --height $HEIGHT \
    --width $WIDTH \
    --dataloader_num_workers 8 \
    --accumulate_grad_batches 1 \
    --warmup_steps 10 \
    --learning_rate 1e-4 \
    --max_epochs 10 \
    --training_strategy "auto" \
    --lora_rank 16 \
    --lora_alpha 16 \
    --use_gradient_checkpointing \
    --train_architecture $TRAIN_TYPE \
    --precision "bf16" \
    --nnodes $NUM_NODES \
    --enable_bucket \
    --pin_memory \
    # --use_gradient_checkpointing_offload \





