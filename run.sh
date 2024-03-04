#!/bin/bash

LIMIT=40000

case $1 in
  1)
    DS_NAME="unbalanced"
    START=0
    ;;
  2)
    DS_NAME="unbalanced"
    START=40000
    ;;
  3)
    DS_NAME="unbalanced"
    START=80000
    ;;
  4)
    DS_NAME="unbalanced"
    START=120000
    ;;
  5)
    DS_NAME="unbalanced"
    START=160000
    ;;
esac

DS_DIR="../dataset/${DS_NAME}"
python start.py \
 --path_to_ds $DS_DIR \
 --output_dir_name output_${DS_NAME}_${START} \
 --start_at $START \
 --limit $LIMIT

python start.py \
 --path_to_ds $DS_DIR \
 --output_dir_name output_${DS_NAME}_test_${START} \
 --split test \
 --start_at $START \
 --limit $LIMIT
