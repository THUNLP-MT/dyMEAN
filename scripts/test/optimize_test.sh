#!/bin/bash

########## adjust configs according to your needs ##########
# usually only the DATA_DIR need to be revised
CODE_DIR=`realpath $(dirname "$0")/../..`
DATA_DIR=/data/private/kxz/antibody/AbTrans/SKEMPI_11_12
TEST_SET=${DATA_DIR}/test.json
NUM_SAMPLE=100
BATCH_SIZE=8
GPU="${GPU:-0}"  # using GPU 0 by default
CKPT=$1
PREDICTOR_CKPT=$2
TEST_SET=$3
NUM_CHANGE="${4:-0}" # default is 0 (no restriction)
OPT_STEP="${5:-10}"  # default is 10
SAVE_DIR="${6:-"$(dirname "$CKPT")/opt_results_${NUM_CHANGE}_${OPT_STEP}"}"
######### end of adjust ##########

# validity check
if [ -z "$CKPT" ] || [ -z "$PREDICTOR_CKPT" ]; then
	echo "Usage: bash $0 <checkpoint> <predictor checkpoint> <test set> [num_change] [opt_step] [save_dir]"
	exit 1;
else
	CKPT=`realpath $CKPT`
	PREDICTOR_CKPT=`realpath $PREDICTOR_CKPT`
	SAVE_DIR=`realpath $SAVE_DIR`
fi

# echo configurations
echo "Locate the project folder at $CODE_DIR"
echo "Using GPUs: $GPU"
echo "Using checkpoint: $CKPT"
echo "Using predictor checkpoint: $PREDICTOR_CKPT"
[ $NUM_CHANGE == 0 ] && echo "Not restricting maxium number of residues to change" || echo "Upperbound of the number of residues to change: ${NUM_CHANGE}"
echo "Number of optimization step: $OPT_STEP"
echo "Results will be saved to $SAVE_DIR"

# set GPU
export CUDA_VISIBLE_DEVICES=$GPU

# run codes
cd ${CODE_DIR}
python opt_generate.py \
	--ckpt ${CKPT} \
	--predictor_ckpt ${PREDICTOR_CKPT} \
	--n_samples ${NUM_SAMPLE} \
	--num_residue_changes ${NUM_CHANGE} \
	--num_optimize_steps ${OPT_STEP} \
	--summary_json ${TEST_SET} \
	--save_dir ${SAVE_DIR} \
	--batch_size ${BATCH_SIZE} \
	--gpu 0