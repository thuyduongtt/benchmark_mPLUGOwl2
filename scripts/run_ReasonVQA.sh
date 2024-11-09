#!/bin/bash

LIMIT=20000
DS_NAME="ReasonVQA"
DS_VERSION="unbalanced"

DS_DIR="../dataset/${DS_VERSION}"

case $1 in
  1)
    START=0
    MULTICHOICE=true
    ;;
  2)
    START=20000
    MULTICHOICE=true
    ;;
  3)
    START=40000
    MULTICHOICE=true
    ;;
  4)
    START=60000
    MULTICHOICE=true
    ;;
  5)
    START=0
    MULTICHOICE=false
    ;;
  6)
    START=20000
    MULTICHOICE=false
    ;;
  7)
    START=40000
    MULTICHOICE=false
    ;;
  8)
    START=60000
    MULTICHOICE=false
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
else
  python start.py \
  --ds_name $DS_NAME \
   --ds_dir $DS_DIR \
   --output_dir_name output_${DS_NAME}_${DS_VERSION}_${START} \
   --start_at $START \
   --limit $LIMIT
fi
