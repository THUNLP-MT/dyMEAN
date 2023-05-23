#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
from os.path import splitext, basename
import sys
import json
from tqdm.contrib.concurrent import process_map


from data.pdb_utils import AgAbComplex
from utils.relax import openmm_relax
from evaluation.affinity import rosetta_affinity
from configs import CACHE_DIR
from utils.random_seed import setup_seed
setup_seed(2023)

root_dir, summary = sys.argv[1], sys.argv[2]
with open(os.path.join(root_dir, 'log.txt'), 'r') as log:
    lines = log.read().strip().split('\n')[:-1]
log_out = open(os.path.join(root_dir, 'openmm_rosetta_log.txt'), 'w')


with open(summary, 'r') as fin:
    items = [json.loads(l) for l in fin.read().strip().split('\n')]

info = { item['pdb']: (item['heavy_chain'], item['light_chain'], item['antigen_chains']) for item in items }

n_tries = 5


def log_print(s, log):
    log.write(s + '\n')
    log.flush()
    print(s)


def eval_one(l):
    pdb = l[:4]
    mut_pdb = l.split(' ')[-1]
    _id = splitext(basename(mut_pdb))[0]
    wt_pdb = os.path.join(root_dir, pdb, pdb + '_original.pdb')
    h, l, antigen_chains = info[pdb]
    ori_affinity = rosetta_affinity(wt_pdb, h, l, _async=False)
    min_score = 1e6
    for t in range(n_tries):
        tmp_pdb = os.path.join(CACHE_DIR, f'{pdb}_{t}.pdb')
        try:
            openmm_relax(mut_pdb, tmp_pdb, excluded_chains=[h, l], inverse_exclude=True)
        except Exception:
            continue
        opt_affinity = rosetta_affinity(tmp_pdb, h, l, _async=False)
        affinity = opt_affinity - ori_affinity
        min_score = min(affinity, min_score)
        os.remove(tmp_pdb)

    ori_cdr = AgAbComplex.from_pdb(wt_pdb, h, l, antigen_chains).get_cdr().get_seq()
    cur_cdr = AgAbComplex.from_pdb(mut_pdb, h, l, antigen_chains).get_cdr().get_seq()
    change = 0
    for x, y in zip(ori_cdr, cur_cdr):
        if x != y:
            change += 1
    message = f'{pdb} {_id} {min_score} {change}'
    print(message)
    return min_score, change, message

scores, changes = [], []
results = process_map(eval_one, lines, max_workers=8)
for score, change, message in results:
    log_out.write(message + '\n')
    scores.append(score)
    changes.append(change)

hit = [1 if s < 0 else 0 for s in scores]
suc_score = [score for i, score in enumerate(scores) if hit[i] == 1]
suc_change = [c for i, c in enumerate(changes) if hit[i] == 1]
log_print(f'success rate {sum(hit) / len(hit)}', log_out)
log_print(f'success average {sum(suc_score) / len(suc_score)}', log_out)
log_print(f'all average {sum(scores) / len(scores)}', log_out)
log_print(f'success average changes {sum(suc_change) / len(suc_change)}', log_out)
log_print(f'all average change {sum(changes) / len(changes)}', log_out)
log_out.close()