#!/bin/bash

CONFIG="E"
BATCH_SIZE=32
NUM_LAYER=2
DEVICE_NUM=4
DATA_TYPE="float32"
DEBUG_DIR="tmp"
DUMP_COMM_VOLUME="false"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --config) CONFIG="$2"; shift ;;
        --batch_size) BATCH_SIZE="$2"; shift ;;
        --num_layer) NUM_LAYER="$2"; shift ;;
        --device_num) DEVICE_NUM="$2"; shift ;;
        --data_type) DATA_TYPE="$2"; shift ;;
        --debug_dir) DEBUG_DIR="$2"; shift ;;
        --dump_comm_volume) DUMP_COMM_VOLUME="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python $SCRIPT_DIR/moe_test.py \
    --config "$CONFIG" \
    --batch_size "$BATCH_SIZE" \
    --num_layer "$NUM_LAYER" \
    --device_num "$DEVICE_NUM" \
    --data_type "$DATA_TYPE" \
    --debug_dir "$DEBUG_DIR" \
    --dump_comm_volume "$DUMP_COMM_VOLUME"