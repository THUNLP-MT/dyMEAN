#!/bin/bash
########## Start of Your Settings ##########
PDB_DIR=$1  # Directory to the IMGT renumbered structures downloaded from SAbDab
OUT_DIR=$2  # Output Directory (~5G)
CDR=H3
########## END of Your Settings ##########
# validity check
if [ -z "$PDB_DIR" ] || [ -z "$OUT_DIR" ]; then
	echo "Usage: bash $0 <pdb directory> <output directory>"
	exit 1;
fi

########## Generate Configures ##########
PDB_DIR=`realpath ${PDB_DIR}`
[ ! -d $OUT_DIR ] && mkdir $OUT_DIR
OUT_DIR=`realpath ${OUT_DIR}`
RABD_OUT_DIR=${OUT_DIR}/RAbD
IGFOLD_OUT_DIR=${OUT_DIR}/IgFold
SKEMPI_OUT_DIR=${OUT_DIR}/SKEMPI


########## setup project directory ##########
CODE_DIR=`realpath $(dirname "$0")/..`
echo "Locate the project folder at ${CODE_DIR}"
cd ${CODE_DIR}


########## SAbDab ##########
echo "Processing SAbDab with output directory ${OUT_DIR}"
# Note: All tasks separate subsets from SAbDab for training models
python -m data.download \
    --summary summaries/sabdab_summary.tsv \
    --pdb_dir $PDB_DIR \
    --fout $OUT_DIR/sabdab_all.json \
    --type sabdab \
    --numbering imgt \
    --pre_numbered \
    --n_cpu 4


########## RAbD (CDR-H3 design) ##########
echo "Processing RAbD with output directory ${RABD_OUT_DIR}"
# generate summary
python -m data.download \
    --summary $OUT_DIR/sabdab_all.json \
    --pdb_dir $PDB_DIR \
    --fout $OUT_DIR/rabd_all.json \
    --type rabd \
    --numbering imgt \
    --pre_numbered \
    --n_cpu 4
# split train/validation
python -m data.split \
    --data $OUT_DIR/sabdab_all.json \
    --out_dir $RABD_OUT_DIR \
    --valid_ratio 0.1 \
    --filter 111 \
    --benchmark $OUT_DIR/rabd_all.json \
    --cdr H3
# data transformation
python -m data.dataset --dataset $RABD_OUT_DIR/test.json
python -m data.dataset --dataset $RABD_OUT_DIR/valid.json
python -m data.dataset --dataset $RABD_OUT_DIR/train.json


########## IgFold Test Set (Complex Structure Prediction) ##########
echo "Processing IgFold test set with output directory ${IGFOLD_OUT_DIR}"
# generate summary
python -m data.download \
    --summary $OUT_DIR/sabdab_all.json \
    --pdb_dir $PDB_DIR \
    --fout $OUT_DIR/igfold_all.json \
    --type igfold_test \
    --numbering imgt \
    --pre_numbered \
    --n_cpu 4
# split train/validation
python -m data.split \
    --data $OUT_DIR/sabdab_all.json \
    --out_dir $IGFOLD_OUT_DIR \
    --valid_ratio 0.1 \
    --filter 111 \
    --benchmark $OUT_DIR/igfold_all.json
# data transformation
python -m data.dataset --dataset $IGFOLD_OUT_DIR/test.json
python -m data.dataset --dataset $IGFOLD_OUT_DIR/valid.json
python -m data.dataset --dataset $IGFOLD_OUT_DIR/train.json


########## SKEMPI (Affinity Optimization) ##########
echo "Processing SKEMPI V2.0 with output directory ${SKEMPI_OUT_DIR}"
# generate summary
python -m data.download \
    --summary $OUT_DIR/sabdab_all.json \
    --pdb_dir $PDB_DIR \
    --fout $OUT_DIR/skempi_all.json \
    --type skempi \
    --numbering imgt \
    --pre_numbered \
    --n_cpu 4
# split train/validation
python -m data.split \
    --data $OUT_DIR/sabdab_all.json \
    --out_dir $SKEMPI_OUT_DIR \
    --valid_ratio 0.1 \
    --filter 111 \
    --benchmark $OUT_DIR/skempi_all.json \
    --cdr H3
# data transformation
python -m data.dataset --dataset $SKEMPI_OUT_DIR/test.json
python -m data.dataset --dataset $SKEMPI_OUT_DIR/valid.json
python -m data.dataset --dataset $SKEMPI_OUT_DIR/train.json
