#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import re
import time
from copy import deepcopy

from data.pdb_utils import Peptide, Protein, merge_to_one_chain
from configs import CACHE_DIR


def exec_bin(mod_pdb, ref_pdb, log, backbone_only):
    options = '-x'
    if backbone_only:
        options += ' -c'
    cmd = f'lddt {options} {mod_pdb} {ref_pdb} > {log} 2>&1'
    return os.system(cmd)


def lddt(mod_protein: Protein, ref_protein: Protein, backbone_only=False):
    # concatenate all chains to one chain
    mod_protein = merge_to_one_chain(mod_protein)
    ref_protein = merge_to_one_chain(ref_protein)

    mod_sign, ref_sign = id(mod_protein), id(ref_protein)
    mod_pdb = os.path.join(CACHE_DIR, f'lddt_{mod_sign}_mod_{time.time()}.pdb')
    ref_pdb = os.path.join(CACHE_DIR, f'lddt_{ref_sign}_ref_{time.time()}.pdb')
    log = os.path.join(CACHE_DIR, f'lddt_log_{mod_sign}_{ref_sign}.txt')
    
    mod_protein.to_pdb(mod_pdb)
    ref_protein.to_pdb(ref_pdb)

    res_code = exec_bin(mod_pdb, ref_pdb, log, backbone_only)
    if res_code != 0:
        raise ValueError(f'lddt execution failed')
    with open(log, 'r') as fin:
        text = fin.read()
    res = re.search(r'Global LDDT score: ([0-1]\.?[0-9]*)', text)
    score = float(res.group(1))
    os.remove(mod_pdb)
    os.remove(ref_pdb)
    os.remove(log)
    return score, text