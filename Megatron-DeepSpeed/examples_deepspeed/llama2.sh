#!/bin/bash
######################################
# Change the below configurations here

# model parallel
TP=4
ZERO_STAGE=0

GPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6006
NNODES=1
NODE_RANK=0


HIDDEN_SIZE=6656
FFN_HIDDEN_SIZE=11008
NUM_LAYERS=2
NUM_HEADS=32
SEQ_LENGTH=128
NUM_KV_HEADS=32

MICRO_BATCH_SIZE=16
NUM_GPUS=8
GLOBAL_BATCH_SIZE=$((MICRO_BATCH_SIZE * NUM_GPUS))



TRAIN_STEPS=80
LR=3e-4
MIN_LR=3e-5
WEIGHT_DECAY=0.1
GRAD_CLIP=1
DS_CONFIG=./deepspeed.json
cat <<EOT > $DS_CONFIG
{
  "train_batch_size" : $GLOBAL_BATCH_SIZE,
  "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE,
  "zero_optimization": {
    "stage": $ZERO_STAGE
  }
}
EOT

ds_args=""
ds_args=" --deepspeed ${ds_args}"
ds_args=" --deepspeed_config=$DS_CONFIG ${ds_args}"
ds_args=" --zero-stage=$ZERO_STAGE ${ds_args}"


DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

torchrun $DISTRIBUTED_ARGS \
       ../pretrain_gpt.py \
       --vocab-size 51200 \
       --no-pipeline-parallel \
       --tensor-model-parallel-size $TP \
       --num-layers $NUM_LAYERS \
       --hidden-size $HIDDEN_SIZE \
       --ffn-hidden-size $FFN_HIDDEN_SIZE \
       --num-attention-heads $NUM_HEADS \
       --micro-batch-size $MICRO_BATCH_SIZE \
       --global-batch-size $GLOBAL_BATCH_SIZE \
       --seq-length $SEQ_LENGTH \
       --max-position-embeddings $SEQ_LENGTH \
       --train-iters $TRAIN_STEPS \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr $LR \
       --lr-decay-style cosine \
       --min-lr $MIN_LR \
       --weight-decay $WEIGHT_DECAY \
       --clip-grad $GRAD_CLIP \
       --optimizer adam \
       --adam-beta1 0.9 \
       --adam-beta2 0.95 \
       --no-query-key-layer-scaling \
       --attention-dropout 0 \
       --hidden-dropout 0 \
       --use-rotary-position-embeddings \
       --untie-embeddings-and-output-weights \
       --swiglu \
       --normalization rmsnorm \
       --disable-bias-linear \
       --num-key-value-heads $NUM_KV_HEADS \
       $ds_args
#    $WANDB_ARGS \