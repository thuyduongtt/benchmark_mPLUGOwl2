#!/bin/bash

cd ../../dataset

# OKVQA
mkdir OKVQA
cd OKVQA

#wget https://okvqa.allenai.org/static/data/mscoco_val2014_annotations.json.zip
#unzip -q mscoco_val2014_annotations.json.zip
#rm mscoco_val2014_annotations.json.zip
#wget https://okvqa.allenai.org/static/data/OpenEnded_mscoco_val2014_questions.json.zip
#unzip -q OpenEnded_mscoco_val2014_questions.json.zip
#rm OpenEnded_mscoco_val2014_questions.json.zip

# Install gdown
pip install --upgrade --no-cache-dir gdown

ZIP_ID='1Rro5L20br0acFFp-ZY-QVxN9zm0-CkgI'
ZIP_NAME='mscoco_val2014_annotations.json'
gdown $ZIP_ID -O $ZIP_NAME
unzip -q $ZIP_NAME
rm $ZIP_NAME

ZIP_ID='1m9LyAV-uHkGroHaFvIsbaaVWdKXN0zHs'
ZIP_NAME='OpenEnded_mscoco_val2014_questions.json'
gdown $ZIP_ID -O $ZIP_NAME
unzip -q $ZIP_NAME
rm $ZIP_NAME
