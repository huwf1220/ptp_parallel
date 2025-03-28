#!/bin/bash
dir=`pwd`
lr=1e-4
min_lr=1e-5
# num_gpus
num_gpus=4
# batch_size
batch_size=1
global_batch_size=$((batch_size * num_gpus))

# model config
seq_len=1024
num_layers=2
hidden_size=1536
num_attn_heads=24
init_std=0.02

# mp&pp size
mp_size=2
no_pp="true"

#train_iter
train_iters=50

lr_decay_style="linear"
zero_stage=0
log_interval=10
eval_iters=1
eval_interval=1000

megatron_options=" \
    --bert-no-binary-head \
    --vocab-size 51200 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --init-method-std ${init_std} \
    --tensor-model-parallel-size ${mp_size} \
    --micro-batch-size ${batch_size} \
    --global-batch-size ${global_batch_size} \
    --num-layers ${num_layers} \
    --hidden-size ${hidden_size} \
    --num-attention-heads ${num_attn_heads} \
    --seq-length ${seq_len} \
    --max-position-embeddings ${seq_len} \
    --train-iters ${train_iters} \
    --lr ${lr} \
    --min-lr ${min_lr} \
    --lr-decay-style ${lr_decay_style} \
    --split 949,50,1 \
    --log-interval ${log_interval} \
    --eval-interval ${eval_interval} \
    --eval-iters ${eval_iters} \
    --weight-decay 1e-2 \
    --clip-grad 1.0 \
    --fp16-lm-cross-entropy \
    --fp16 "


template_json="ds_config_bert_test_TEMPLATE.json"
config_json="ds_config_bert_bsz${global_batch_size}_mbsz${batch_size}_log${log_interval}_zero${zero_stage}.json"
if [[ $zero_stage -gt 0 ]]; then
sed "s/CONFIG_BATCH_SIZE/${global_batch_size}/" ${template_json} \
    | sed "s/CONFIG_MBSIZE/${batch_size}/" \
    | sed "s/LOG_INTERVAL/${log_interval}/" \
    | sed "s/ZERO_STAGE/${zero_stage}/" \
    | sed "s/PRESCALE_GRAD/false/" \
    | sed "s/CONFIG_FP16_ENABLED/false/" \
    | sed "s/CONFIG_BF16_ENABLED/false/" \
      > ${config_json}
else
sed "s/CONFIG_BATCH_SIZE/${global_batch_size}/" ${template_json} \
    | sed "s/CONFIG_MBSIZE/${batch_size}/" \
    | sed "s/LOG_INTERVAL/${log_interval}/" \
    | sed "s/ZERO_STAGE/${zero_stage}/" \
    | sed "s/PRESCALE_GRAD/false/" \
    | sed "s/CONFIG_FP16_ENABLED/true/" \
    | sed "s/CONFIG_BF16_ENABLED/false/" \
      > ${config_json}
fi

deepspeed_options=" \
    --deepspeed \
    --deepspeed_config ${config_json} \
    --zero-stage ${zero_stage} "

if [[ "${no_pp}" = "true" ]]; then
deepspeed_options="${deepspeed_options} \
    --no-pipeline-parallel"
fi

deepspeed ${dir}/../../pretrain_bert.py ${megatron_options} ${deepspeed_options} 