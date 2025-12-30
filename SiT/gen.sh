#!/bin/bash

MODEL="SiT-XL/2"
PER_PROC_BATCH_SIZE=64
NUM_FID_SAMPLES=50000
PATH_TYPE="linear"
MODE="sde"
NUM_STEPS=250
# cfg_scale=1.0 (without CFG),by default we use cfg_scale=1.35 with guidance interval.
CFG_SCALE=1.35
# ig_scale=1.0 (without IG),by default we use ig_scale=1.4 with guidance interval.
IG_SCALE=1.4
GUIDANCE_HIGH=0.7
IG_GUIDANCE_LOW=0
RESOLUTION=256
VAE="ema"
GLOBAL_SEED=0
SAMPLE_DIR=[Base directory to save images]
CKPT=[Your checkpoint path]




python -m torch.distributed.launch \
    --nproc_per_node=4 \
    generate.py \
    --num-fid-samples $NUM_FID_SAMPLES \
    --path-type $PATH_TYPE \
    --per-proc-batch-size $PER_PROC_BATCH_SIZE \
    --mode $MODE \
    --num-steps $NUM_STEPS \
    --cfg-scale $CFG_SCALE \
    --ig-scale $sg_val \
    --guidance-high $GUIDANCE_HIGH \
    --ig-guidance-low $sg_guidance_low \
    --sample-dir $SAMPLE_DIR \
    --model $MODEL \
    --ckpt $CKPT \
    --vae $VAE \
    --resolution $RESOLUTION \
    --global-seed $GLOBAL_SEED \


python npz_convert.py \
    --model $MODEL \
    --ckpt $CKPT \
    --sample-dir $SAMPLE_DIR \
    --num-fid-samples $NUM_FID_SAMPLES \
    --resolution $RESOLUTION \
    --guidance-high $GUIDANCE_HIGH \
    --ig-guidance-low $sg_guidance_low \
    --vae $VAE \
    --cfg-scale $CFG_SCALE \
    --ig-scale $sg_val \
    --global-seed $GLOBAL_SEED \
    --mode $MODE





