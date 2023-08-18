#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import json
from tqdm import tqdm
from copy import deepcopy
from os.path import splitext, basename

import numpy as np
import torch
from torch.utils.data import DataLoader

from generate import to_cplx
from data.pdb_utils import Protein, AgAbComplex, Peptide
from utils.relax import openmm_relax
from utils.logger import print_log
from .design import design


def structure_prediction(ckpt, gpu, pdbs, epitope_defs, seqs, out_dir,
           identifiers=None, enable_openmm_relax=True,
           batch_size=32, num_workers=4):
    design(ckpt, gpu, pdbs, epitope_defs, seqs, out_dir, identifiers,
           remove_chains=None, enable_openmm_relax=enable_openmm_relax,
           batch_size=batch_size, num_workers=num_workers)


if __name__ == '__main__':
    ckpt = './checkpoints/struct_prediction.ckpt'
    root_dir = './demos/data'
    n_sample = 10  # sample 10 conformations
    pdbs = [os.path.join(root_dir, '1nca_antigen.pdb') for _ in range(n_sample)]
    epitope_defs = [os.path.join(root_dir, '1nca_epitope.json') for _ in range(n_sample)]
    identifiers = [f'1nca_model_{i}' for i in range(n_sample)]
    
    seqs = [
        (
            ('H', 'QIQLVQSGPELKKPGETVKISCKASGYTFTNYGMNWVKQAPGKGLKWMGWINTNTGEPTYGEEFKGRFAFSLETSASTANLQINNLKNEDTATFFCARGEDNFGSLSDYWGQGTTVTVSS'),
            ('L', 'DIVMTQSPKFMSTSVGDRVTITCKASQDVSTAVVWYQQKPGQSPKLLIYWASTRHIGVPDRFAGSGSGTDYTLTISSVQAEDLALYYCQQHYSPPWTFGGGTKLEIK')
        ) \
        for _ in pdbs
    ]  # the first item of each tuple is heavy chain, the second is light chain

    structure_prediction(
        ckpt=ckpt,  # path to the checkpoint of the trained model
        gpu=0,      # the ID of the GPU to use
        pdbs=pdbs,  # paths to the PDB file of each antigen (here antigen is all TRPV1)
        epitope_defs=epitope_defs,  # paths to the epitope definitions
        seqs=seqs,      # the given sequences of the framework regions
        out_dir=root_dir,           # output directory
        identifiers=identifiers,    # name of each output antibody
        enable_openmm_relax=True)   # use openmm to relax the generated structure