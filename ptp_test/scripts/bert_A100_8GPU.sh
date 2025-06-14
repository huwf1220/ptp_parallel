#!/bin/bash

DEVICE_NUM=8
DATA_TYPE="float32"

TEST_CONFIG=(
    "C, 8, 8"
    "C, 16, 4"
    "C, 32, 2"
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BERT_DIR="$(cd "$SCRIPT_DIR/../bert" && pwd)" 

for CONFIG_LINE in "${TEST_CONFIG[@]}"; do
    IFS=',' read -r CONFIG BATCH_SIZE NUM_LAYER <<< "$CONFIG_LINE"
    $BERT_DIR/bert_test.sh \
        --config "$CONFIG" \
        --batch_size "$BATCH_SIZE" \
        --num_layer "$NUM_LAYER" \
        --device_num "$DEVICE_NUM" \
        --data_type "$DATA_TYPE" \
        --parallel_block_opt 1 

    echo "================================================================="
done