#!/bin/zsh
CODE_DIR=`realpath $(dirname "$0")/../..`
DATA_DIR=${CODE_DIR}/RAbD
DATASET=${DATA_DIR}/test.json
CDR=H3

MODEL=$1
CKPT=$2
if [ -z "$GPU" ]; then
    GPU=-1
fi

DATA_OUT_DIR=${DATA_DIR}/${MODEL}_pipeline/templates
RESULT_DIR=${DATA_DIR}/${MODEL}_pipeline/results


cd ${CODE_DIR}
export CUDA_VISIBLE_DEVICES=$GPU

echo "CDR Model: ${MODEL}, checkpoint: ${CKPT}"
MODEL_CODE_DIR=`echo -e "from configs import ${MODEL}_DIR\nprint(${MODEL}_DIR)" | python3`
echo "Model source: ${MODEL_CODE_DIR}"

echo "Generating complex templates"
python -m baselines.pipeline.gen_complex_templates \
    --dataset_json ${DATASET} \
    --cdr_model ${MODEL} \
    --out_dir ${DATA_DIR}/complex_templates \
    --data_out_dir ${DATA_OUT_DIR} \
    --cdr_type ${CDR}

echo "Inference"

if [ "${MODEL}" = "MEAN" ]; then
    PYTHONPATH=$MODEL_CODE_DIR python baselines/pipeline/cdr_models/mean_gen.py \
        --ckpt ${CKPT} \
        --dataset ${DATA_OUT_DIR}/summary.json \
        --batch_size 32 \
        --cdr_type ${CDR} \
        --gpu 0 \
        --out_dir ${RESULT_DIR}
elif [ "${MODEL}" = "Rosetta" ]; then
    PYTHONPATH=$CODE_DIR python baselines/pipeline/cdr_models/rosetta_gen.py \
        --dataset ${DATA_OUT_DIR}/summary.json \
        --cdr_type ${CDR} \
        --out_dir ${RESULT_DIR}
elif [ "${MODEL}" = "DiffAb" ]; then
    conda activate agdiffab
    PYTHONPATH=$MODEL_CODE_DIR python baselines/pipeline/cdr_models/diffab_gen.py \
        --dataset ${DATA_OUT_DIR}/summary.json \
        --out_dir ${RESULT_DIR} \
        --config ${CKPT}
    conda deactivate
fi

echo "Evaluate"
if [ "${MODEL}" = "Rosetta" ]; then
    python -m baselines.pipeline.evaluate \
        --summary_json ${DATASET} \
        --ref_dir ${DATA_DIR}/complex_templates \
        --gen_dir ${RESULT_DIR} \
        --cdr_type ${CDR}
else
    python -m baselines.pipeline.evaluate \
        --summary_json ${DATASET} \
        --ref_dir ${DATA_DIR}/complex_templates \
        --gen_dir ${RESULT_DIR} \
        --cdr_type ${CDR} \
        --sidechain_packing
fi

