#!/bin/bash
DIR=`pwd`

# num_gpus
NUM_GPUS=4

# batch_size
BATCH_SIZE=1
GLOBAL_BATCH_SIZE=$((BATCH_SIZE * NUM_GPUS))

# model config
NUM_LAYERS=2
HIDDEN_SIZE=4096
NUM_ATTN_HEADS=32
SEQ_LEN=1024
LR=2.0e-4
MIN_LR=2.0e-5


# mp&pp size
MP_SIZE=2
PP_SIZE=1




###############################################################################
### MoE configs
## Number of experts. EP_SIZE 1 means dense model without MoE
EP_SIZE=1

if [[ $EP_SIZE -gt $NUM_GPUS ]]; then
    EP_PARALLEL_SIZE=$NUM_GPUS
else
    EP_PARALLEL_SIZE=$EP_SIZE
fi

## Original GPT-3 model always set min LR at 10% of max LR. For MoE model, we
## found that lower LR and min LR (than the base dense model) helps.
## For 1.3B MoE-128 model we used LR=1.2e-4 and MIN_LR=1.0e-6.
## For 350M MoE-128 model we used LR=2.0e-4 and MIN_LR=2.0e-6, but they are not
## heavily tuned.
# LR=2.0e-4
# MIN_LR=2e-06

## Coefficient for MoE loss. We find that 0.01 is a good value at least for
## 1.3B MoE-128 model
MLC=0.01

## Below configs adjust the MoE expert token capacity limit during training and
## eval. To completely disable capacity limit, set MOE_DROP_TOKEN to false.
## Larger capacity factor or disabling capacity limit could improve training
## convergence, but will also reduce training throughput.
MOE_TRAIN_CAP_FACTOR=1.0
MOE_EVAL_CAP_FACTOR=1.0
MOE_MIN_CAP=4
MOE_DROP_TOKEN="true"
# MOE_DROP_TOKEN="false"
###############################################################################
### Curriculum learning (CL) configs
## Enable/disable CL
CL_ENABLED="false"
## Consult the tutorial https://www.deepspeed.ai/tutorials/curriculum-learning/
## for tuning the following configs
CL_START_SEQLEN=80
CL_AVG_SEQLEN=$(( (${CL_START_SEQLEN} + ${SEQ_LEN}) / 2 ))
CL_TOKENS=60
CL_TOKENS=$((${CL_TOKENS} * 1000000000))
CL_STEP=$(( ${CL_TOKENS} / (${GLOBAL_BATCH_SIZE} * ${CL_AVG_SEQLEN}) ))
###############################################################################
### Misc configs
ITER=50
LOG_INTERVAL=10
EVAL_ITERS=1
EVAL_INTERVAL=1000
SAVE_INTERVAL=1000

## Standard deviation for weight initialization
## We used 0.014 for 350M/1.3B dense/MoE models, and used 0.01 for 6.7B
## dense model. Usually larger model needs lower std.
INIT_STD=0.014
# INIT_STD=0.01

## Activation checkpointing saves GPU memory, but reduces training speed
# ACTIVATION_CHECKPOINT="true"
ACTIVATION_CHECKPOINT="false"
###############################################################################
### Output and data configs
current_time=$(date "+%Y.%m.%d-%H.%M.%S")
host="${HOSTNAME}"
MODEL_SIZE=0
NAME="gpt-${MODEL_SIZE}B-lr-${LR}-minlr-${MIN_LR}-bs-${GLOBAL_BATCH_SIZE}-gpus-${NUM_GPUS}-mp-${MP_SIZE}-pp-${PP_SIZE}"
if [[ $EP_SIZE -gt 1 ]]; then
    NAME="${NAME}-ep-${EP_SIZE}-mlc-${MLC}-cap-${MOE_TRAIN_CAP_FACTOR}-drop-${MOE_DROP_TOKEN}"
fi
if [ "${CL_ENABLED}" = "true" ]; then
    NAME="${NAME}-cl-${CL_START_SEQLEN}-${CL_STEP}"
fi

OUTPUT_BASEPATH=$DIR/output
mkdir -p "${OUTPUT_BASEPATH}/tensorboard/"
mkdir -p "${OUTPUT_BASEPATH}/checkpoint/"
mkdir -p "${OUTPUT_BASEPATH}/log/"
TENSORBOARD_DIR="${OUTPUT_BASEPATH}/tensorboard/${NAME}_${host}_${current_time}"
mkdir -p ${TENSORBOARD_DIR} 
## Note that for MoE model with billion-scale base model, the checkpoint can be
## as large as TB-scale which normal NFS cannot handle efficiently.
CHECKPOINT_PATH="${OUTPUT_BASEPATH}/checkpoint/${NAME}"

# USE_INTERNAL_DATA="true"
USE_INTERNAL_DATA="false"

if [ "${USE_INTERNAL_DATA}" = "true" ]; then
    ## The internal data is only accessible within Microsoft
    ## For cluster Azure-EastUS-V100-32GB-4, Azure-WestUS3-A100
    # BASE_DATA_PATH=/vc_data/Megatron-LM/data
    # DATA_HOME="/vc_data/pile-cc1-cc2-shuf"
    ## For cluster Lab-RR1-V100
    BASE_DATA_PATH=/data/Megatron-LM/data
    DATA_HOME="/turing-ssd/users/conglli/data/pile-cc1-cc2-shuf"
    ## For cluster Azure-CentralUS-A100
    # BASE_DATA_PATH=/data/Megatron-LM/data
    # DATA_HOME=/vc_data_1/users/amawa/blended

    VOCAB_PATH=${BASE_DATA_PATH}/gpt2-vocab.json
    MERGE_PATH=${BASE_DATA_PATH}/gpt2-merges.txt
    ARX="${DATA_HOME}/ArXiv_ftfy_cleaned_id_shuf_text_document"
    BC2="${DATA_HOME}/BookCorpus2_ftfy_cleaned_id_shuf_text_document"
    B3="${DATA_HOME}/Books3_ftfy_cleaned_id_shuf_text_document"
    CC2020="${DATA_HOME}/CC-2020-50_id_cleaned_shuf_text_document"
    CC2021="${DATA_HOME}/CC-2021-04_id_cleaned_shuf_text_document"
    GIT="${DATA_HOME}/Github_ftfy_id_shuf_text_document"
    GUT="${DATA_HOME}/Gutenberg_PG-19_ftfy_cleaned_id_cleaned_shuf_text_document"
    NIH="${DATA_HOME}/NIH_ExPorter_ftfy_id_shuf_text_document"
    OWT2="${DATA_HOME}/OpenWebText2_ftfy_cleaned_id_shuf_text_document"
    PCC="${DATA_HOME}/Pile-CC_id_cleaned_shuf_text_document"
    PM="${DATA_HOME}/PubMed_Abstracts_ftfy_id_shuf_text_document"
    RN="${DATA_HOME}/rn_dedup_shuf_cleaned_0.7_cleaned_shuf_text_document"
    SE="${DATA_HOME}/StackExchange_ftfy_id_shuf_text_document"
    ST="${DATA_HOME}/stories_dedup0.7_shuf_cleaned_shuf_text_document"
    WIK="${DATA_HOME}/Wikipedia_en_ftfy_id_shuf_text_document"
    DATA_BLEND="0.14336 ${B3} 0.08962 ${RN} 0.19336 ${OWT2} 0.05689 ${SE} \
    0.00859 ${ST} 0.02897 ${PM} 0.04771 ${WIK} 0.00873 ${GUT} 0.01007 ${BC2} \
    0.00208 ${NIH} 0.13017 ${CC2020} 0.09446 ${PCC} 0.15652 ${CC2021} \
    0.01359 ${ARX} 0.01588 ${GIT}"
else
    VOCAB_PATH=/data/the_pile_public_merged_nopreprocessing/gpt2-vocab.json
    MERGE_PATH=/data/the_pile_public_merged_nopreprocessing/gpt2-merges.txt
    # Public the Pile dataset, can be downloaded at https://mystic.the-eye.eu/public/AI/pile_neox/
    DATA_BLEND=/data/the_pile_public_merged_nopreprocessing/pile_text_document
fi
###############################################################################
# data_options=" \
#          --vocab-file ${VOCAB_PATH} \
#          --merge-file ${MERGE_PATH} \
#          --data-path ${DATA_BLEND} \
#          --data-impl mmap"
        
megatron_options=" \
        --train-iters ${ITER} \
        --vocab-size 51200 \
        --no-pipeline-parallel \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --tensor-model-parallel-size ${MP_SIZE} \
        --moe-expert-parallel-size ${EP_PARALLEL_SIZE} \
        --num-experts ${EP_SIZE} \
        --moe-loss-coeff ${MLC} \
        --moe-train-capacity-factor ${MOE_TRAIN_CAP_FACTOR} \
        --moe-eval-capacity-factor ${MOE_EVAL_CAP_FACTOR} \
        --moe-min-capacity ${MOE_MIN_CAP} \
        --init-method-std ${INIT_STD} \
        --micro-batch-size ${BATCH_SIZE} \
        --global-batch-size ${GLOBAL_BATCH_SIZE} \
        --num-layers ${NUM_LAYERS} \
        --hidden-size ${HIDDEN_SIZE} \
        --num-attention-heads ${NUM_ATTN_HEADS} \
        --seq-length ${SEQ_LEN} \
        --max-position-embeddings ${SEQ_LEN} \
        --lr ${LR} \
        --min-lr ${MIN_LR} \
        --lr-decay-style cosine \
        --split 98,2,0 \
        --log-interval ${LOG_INTERVAL} \
        --eval-interval ${EVAL_INTERVAL} \
        --eval-iters ${EVAL_ITERS} \
        --weight-decay 0.1 \
        --clip-grad 1.0 \
        --hysteresis 2 \
        --num-workers 0 \
        --tensorboard-queue-size 1 \
        --log-timers-to-tensorboard \
        --log-batch-size-to-tensorboard \
        --log-validation-ppl-to-tensorboard \
        --fp16 \
        --tensorboard-dir ${TENSORBOARD_DIR}"

if [ "${ACTIVATION_CHECKPOINT}" = "true" ]; then
megatron_options="${megatron_options} \
        --checkpoint-activations"
fi

if [[ $EP_SIZE -gt 1 ]]; then
megatron_options="${megatron_options} \
        --create-moe-param-group"
fi

if [ "${MOE_DROP_TOKEN}" = "false" ]; then
megatron_options="${megatron_options} \
        --disable-moe-token-dropping"
fi

template_json="ds_config_gpt_test_TEMPLATE.json"
config_json="ds_config_gpt_${NAME}.json"
sed "s/CONFIG_BATCH_SIZE/${GLOBAL_BATCH_SIZE}/" ${template_json} \
    | sed "s/CONFIG_MBSIZE/${BATCH_SIZE}/" \
    | sed "s/LOG_INTERVAL/${LOG_INTERVAL}/" \
    | sed "s/ZERO_STAGE/0/" \
    | sed "s/PRESCALE_GRAD/true/" \
    | sed "s/CONFIG_FP16_ENABLED/true/" \
    | sed "s/CONFIG_BF16_ENABLED/false/" \
    | sed "s/CONFIG_CL_ENABLED/${CL_ENABLED}/" > ${config_json}

deepspeed_options=" \
		    --deepspeed \
		    --deepspeed_config ${config_json} \
		    --pipeline-model-parallel-size ${PP_SIZE}"

# Currently MoE is not compatible with pipeline parallel
if [[ $EP_SIZE -gt 1 ]]; then
deepspeed_options="${deepspeed_options} \
        --no-pipeline-parallel"
fi

if [ "${ACTIVATION_CHECKPOINT}" = "true" ]; then
deepspeed_options="${deepspeed_options} \
        --deepspeed-activation-checkpointing"
fi
# ${data_options}
deepspeed ${DIR}/../../pretrain_gpt.py ${megatron_options} ${deepspeed_options} 

set +x
