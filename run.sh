#!/bin/bash

case $1 in
  1)
    DS_NAME="unbalanced"
    ;;
  2)
    DS_NAME="balanced_10"
    ;;
esac

DS_DIR="../dataset/${DS_NAME}"
python start.py \
 --path_to_ds $DS_DIR \
 --output_dir_name output_${DS_NAME}

python start.py \
 --path_to_ds $DS_DIR \
 --output_dir_name output_${DS_NAME}_test \
 --split test
