#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import re
import time

from data.pdb_utils import Protein, merge_to_one_chain

FILE_DIR = os.path.split(__file__)[0]
TMEXEC = os.path.join(FILE_DIR, 'TMscore')
CACHE_DIR = os.path.join(FILE_DIR, '__cache__')
# create cache dir
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)


def tm_score(mod_protein: Protein, ref_protein: Protein):
    mod_protein = merge_to_one_chain(mod_protein)
    ref_protein = merge_to_one_chain(ref_protein)

    mod_sign, ref_sign = id(mod_protein), id(ref_protein)
    mod_pdb = os.path.join(CACHE_DIR, f'tmscore_{mod_sign}_mod_{time.time()}.pdb')
    ref_pdb = os.path.join(CACHE_DIR, f'tmscore_{ref_sign}_ref_{time.time()}.pdb')

    mod_protein.to_pdb(mod_pdb)
    ref_protein.to_pdb(ref_pdb)

    p = os.popen(f'{TMEXEC} {mod_pdb} {ref_pdb}')
    text = p.read()
    p.close()
    res = re.search(r'TM-score\s*= ([0-1]\.[0-9]+)', text)
    score = float(res.group(1))

    os.remove(mod_pdb)
    os.remove(ref_pdb)

    return score