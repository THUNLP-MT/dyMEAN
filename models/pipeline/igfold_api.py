#!/usr/bin/python
# -*- coding:utf-8 -*-
import sys

# import igfold from its repo
from configs import IGFOLD_DIR, IGFOLD_CKPTS
sys.path.append(IGFOLD_DIR)
from igfold import IgFoldRunner
IGFOLD = IgFoldRunner(model_ckpts=IGFOLD_CKPTS)
sys.path.remove(IGFOLD_DIR)


def pred(heavy_chain_seq, light_chain_seq, out_file, do_refine=True):
    global IGFOLD

    sequences = { 'H': heavy_chain_seq, 'L': light_chain_seq }
    pred_pdb = out_file

    IGFOLD.fold(
        pred_pdb, # Output PDB file
        sequences=sequences, # Antibody sequences
        do_refine=do_refine, # Refine the antibody structure with PyRosetta
        do_renum=False, # Renumber predicted antibody structure (Chothia)
    )