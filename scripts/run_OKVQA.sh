#!/bin/bash

DS_NAME="OKVQA"
MULTICHOICE=true

DS_DIR="../dataset/${DS_NAME}"
IMG_DIR="../dataset/COCO/val2014"

if [ "$MULTICHOICE" = true ] ; then
  python start.py \
   --ds_name $DS_NAME \
   --ds_dir $DS_DIR \
   --img_dir $IMG_DIR \
   --output_dir_name output_mc_${DS_NAME} \
   --multichoice

else
  python start.py \
   --ds_name $DS_NAME \
   --ds_dir $DS_DIR \
   --img_dir $IMG_DIR \
   --output_dir_name output_${DS_NAME} \
fi
