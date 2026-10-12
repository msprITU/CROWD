#!/bin/bash
set -e

DATA_DIR="./data"
mkdir -p $DATA_DIR

echo "Downloading visdrone_data.zip"
gdown https://drive.google.com/file/d/1v-SR0EoqZjeC4cl0TzCGacXfO3yD6QwC/view?usp=sharing -O $DATA_DIR/visdrone_data.zip

echo "Unzipping visdrone_data.zip"
unzip -q $DATA_DIR/visdrone_data.zip -d $DATA_DIR/visdrone_data/
rm -f $DATA_DIR/visdrone_data.zip

echo "Processing raw data (.h5 and .npy)..."
python prepare_dataset.py --data_dir $DATA_DIR/visdrone_data

echo "ALL DONE"