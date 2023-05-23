#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import json
import argparse
import multiprocessing
from tqdm import tqdm
from time import time
from tqdm.contrib.concurrent import process_map
from os.path import splitext, basename

import numpy as np

from data.pdb_utils import Protein
from api.design import design
from evaluation.pred_ddg import foldx_minimize_energy, foldx_dg
from utils.logger import print_log
from utils.random_seed import setup_seed
setup_seed(2023)


def cal_affinity(inputs):
    out_dir, filename, rec_chains, lig_chains = inputs
    f = os.path.join(out_dir, filename + '.pdb')
    relax_f = foldx_minimize_energy(f)
    energy = foldx_dg(relax_f, rec_chains, lig_chains)

    return filename, energy


def display(args):
    print_log('Prepraring settings')
    
    epitope = json.load(open(args.epitope_def, 'r'))
    pdb_id = splitext(basename(args.pdb))[0]
    prot = Protein.from_pdb(args.pdb)
    epitope_chains = {}
    for chain, res_id in epitope:
        epitope_chains[chain] = True
    remove_chains = [c for c in prot.get_chain_names() if c not in epitope_chains]
    out_dir = os.path.join(args.save_dir, pdb_id)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # extract framework templates
    print_log(f'Loading framework library from {args.library}')
    with open(args.library, 'r') as fin:   
        items = [json.loads(line) for line in fin.read().strip().split('\n')]
    fr_lib = []
    for item in tqdm(items):
        H, L = item['heavy_chain_seq'], item['light_chain_seq']
        fr_lib.append((('H', H), ('L', L)))
    print_log(f'{len(fr_lib)} frameworks loaded')
    if args.n_sample > len(fr_lib):
        print_log(f'Number of samples exceeds the size of framework library, set n_sample to {len(fr_lib)}', level='WARN')
        args.n_sample = len(fr_lib)

    # generate candidates for each target
    fr_idxs = [i for i in range(len(fr_lib))]
    rand_fr_idx = np.random.choice(fr_idxs, size=args.n_sample, replace=False)
    rand_fr_lib = [fr_lib[i] for i in rand_fr_idx]
    print_log(f'Design for {args.pdb}')
    start_time = time()

    pdbs = [args.pdb for _ in rand_fr_lib]
    epitope_defs = [args.epitope_def for _ in rand_fr_lib]
    identifiers = [pdb_id + f'_{i}' for i in rand_fr_idx]
    remove_chains = [remove_chains for _ in rand_fr_lib]
    design(ckpt=args.ckpt,
           gpu=args.gpu,
           pdbs=pdbs,
           epitope_defs=epitope_defs,
           frameworks=rand_fr_lib,
           identifiers=identifiers,
           out_dir=out_dir,
           remove_chains=remove_chains,
           batch_size=args.batch_size,
           num_workers=args.num_workers,
           enable_openmm_relax=True,
           auto_detect_cdrs=True)
    print_log(f'Finished generating {len(pdbs)} candidates, elapsed {round(time() - start_time, 2)}s')

    # calculate rough affinity energy
    print_log('Start calculating rough affinity score')
    start_time = time()
    aff_log_file = os.path.join(out_dir, 'affinity.txt')
    aff_log = open(aff_log_file, 'w')
    results = []
    rec_chains = list(epitope_chains.keys())
    lig_chains = ['H', 'L']
    pbar = tqdm(total=len(identifiers))
    pool = multiprocessing.Pool(args.num_workers)

    
    def affinity_callback(return_data):
        filename, energy = return_data
        pbar.update(1)
        aff_log.write(f'{filename} {energy}\n')
        aff_log.flush()
        results.append((filename, energy))

    for filename in identifiers:
        pool.apply_async(func=cal_affinity, args=((out_dir, filename, rec_chains, lig_chains),), callback=affinity_callback)

    pool.close()
    pool.join()
    aff_log.close()
    # sort energy (lower means better) and rewrite log
    results = sorted(results, key=lambda tup: tup[1])
    with open(aff_log_file, 'w') as fout:
        fout.writelines([f'{_id} {energy}\n' for _id, energy in results])
    print_log(f'energy score saved to {aff_log_file} in best-to-worst order, elapsed {round(time() - start_time, 2)}s')

    return


def parse():
    parser = argparse.ArgumentParser(description='Generate antibodies given epitopes')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--pdb', type=str, required=True, help='Path to the antigen PDB')
    parser.add_argument('--epitope_def', type=str, required=True, help='Path to the definition of the epitope')
    parser.add_argument('--library', type=str, required=True, help='Path to framework library')
    parser.add_argument('--n_sample', type=int, required=True, help='Total number of display samples')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save generated dataset')

    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for inference')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers to use')

    parser.add_argument('--gpu', type=int, default=-1, help='GPU to use, -1 for cpu')
    return parser.parse_args()


if __name__ == '__main__':
    display(parse())
