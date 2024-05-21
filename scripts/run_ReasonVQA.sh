#!/bin/bash

LIMIT=20000
DS_NAME="ReasonVQA"
DS_VERSION="unbalanced"
MULTICHOICE=true

DS_DIR="../dataset/${DS_VERSION}"

case $1 in
  1)
    START=0
    ;;
  2)
    START=20000
    ;;
  3)
    START=40000
    ;;
  4)
    START=60000
    ;;
  5)
    START=80000
    ;;
  6)
    START=100000
    ;;
  7)
    START=120000
    ;;
esac

if [ "$MULTICHOICE" = true ] ; then
  python start.py \
   --ds_name $DS_NAME \
   --ds_dir $DS_DIR \
   --output_dir_name output_mc_${DS_NAME}_${DS_VERSION}_${START} \
   --start_at $START \
   --limit $LIMIT \
   --multichoice

  python start.py \
  --ds_name $DS_NAME \
   --ds_dir $DS_DIR \
   --output_dir_name output_mc_${DS_NAME}_${DS_VERSION}_test_${START} \
   --split test \
   --start_at $START \
   --limit $LIMIT \
   --multichoice

else
  python start.py \
  --ds_name $DS_NAME \
   --ds_dir $DS_DIR \
   --output_dir_name output_${DS_NAME}_${DS_VERSION}_${START} \
   --start_at $START \
   --limit $LIMIT

  python start.py \
  --ds_name $DS_NAME \
   --ds_dir $DS_DIR \
   --output_dir_name output_${DS_NAME}_${DS_VERSION}_test_${START} \
   --split test \
   --start_at $START \
   --limit $LIMIT
fi
