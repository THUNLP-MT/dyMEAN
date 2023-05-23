#!/bin/bash
########## adjust configs according to your needs ##########
# usually only the DATA_DIR need to be revised
CODE_DIR=`realpath $(dirname "$0")/../..`
# DATA_DIR=${CODE_DIR}/all_data/SKEMPI
DATA_DIR=/data/private/kxz/tmp/ab_data/SKEMPI
GPU="${GPU:-0}"  # using GPU 0 by default
CKPT=$1
SAVE_DIR="${2:-"$(dirname "$CKPT")/predictor"}"  # save to the same directory as the checkpoint by default
######### end of adjust ##########
echo "Locate the project folder at ${CODE_DIR}"
if [ -z "$CKPT" ]; then
	echo "Usage: bash $0 <checkpoint> [save_dir]"
	exit 1;
else
	CKPT=`realpath $CKPT`
	SAVE_DIR=`realpath $SAVE_DIR`
fi


########## setup GPU ##########
GPU="${GPU:-0}" # default GPU 0
echo "Using GPUs: $GPU"
export CUDA_VISIBLE_DEVICES=$GPU

cd $CODE_DIR
########## generate dataset ##########
for i in "train" "valid" "test" ;
do
echo "Generating ddg dataset for the predictor ($i)"
# python -m data.gen_ddg_dataset \
#     --ckpt $CKPT \
#     --summary_json $DATA_DIR/${i}.json \
#     --save_dir $DATA_DIR/ddg/${i} \
#     --gpu 0
done

########## start training ##########
echo "Training the predictor"
python train_predictor.py \
    --train_set $DATA_DIR/ddg/train/data.jsonl \
    --valid_set $DATA_DIR/ddg/valid/data.jsonl \
    --test_set $DATA_DIR/ddg/test/data.jsonl \
    --save_dir $SAVE_DIR \
    --gpu 0
