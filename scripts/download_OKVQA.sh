#!/bin/bash

cd ../../dataset

# OKVQA
mkdir OKVQA
cd OKVQA
wget https://okvqa.allenai.org/static/data/mscoco_val2014_annotations.json.zip
unzip -q mscoco_val2014_annotations.json.zip
rm mscoco_val2014_annotations.json.zip
wget https://okvqa.allenai.org/static/data/OpenEnded_mscoco_val2014_questions.json.zip
unzip -q OpenEnded_mscoco_val2014_questions.json.zip
rm OpenEnded_mscoco_val2014_questions.json.zip

