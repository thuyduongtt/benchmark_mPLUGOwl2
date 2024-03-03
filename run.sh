#!/bin/bash

case $1 in
  1)
    DS_NAME="unbalanced"
    MODEL_NAME="LORA-BIAS-7B"
    LLAMA_TYPE="llama-2-7b"
    ;;
  2)
    DS_NAME="balanced_10"
    MODEL_NAME="LORA-BIAS-7B"
    LLAMA_TYPE="llama-2-7b"
    ;;
esac

DS_DIR="../dataset/${DS_NAME}"
python start.py \
 --path_to_ds $DS_DIR \
 --output_dir_name output_${MODEL_NAME}_${LLAMA_TYPE}_${DS_NAME} \
 --model_name $MODEL_NAME \
 --llama_type $LLAMA_TYPE

python start.py \
 --path_to_ds $DS_DIR \
 --output_dir_name output_${MODEL_NAME}_${LLAMA_TYPE}_${DS_NAME}_test \
 --split test \
 --model_name $MODEL_NAME \
 --llama_type $LLAMA_TYPE
