#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import json
import argparse
import multiprocessing
from tqdm import tqdm
from time import time
from tqdm.contrib.concurrent import process_map

import numpy as np

from data import Protein
from api.design import design
from api.binding_interface import get_interface
from utils.relax import openmm_relax
from utils.pyrosetta_tools import interface_energy, fast_relax
from utils.logger import print_log
from utils.random_seed import setup_seed
setup_seed(2023)


def cal_affinity(inputs):
    target_info, filename, flexible_map = inputs
    f = os.path.join(target_info['out_dir'], filename + '.pdb')
    # # openmm_relax(f, f)
    # fast_relax(f, flexible_map=flexible_map)
    # # fast_relax(f, f)
    # energy = interface_energy(f, ['H', 'L'])

    origin_prot = Protein.from_pdb(target_info['pdb'])
    cur_prot = Protein.from_pdb(f)

    centers = []
    for chain in [origin_prot.get_chain('H'), origin_prot.get_chain('I'), cur_prot.get_chain('H'), cur_prot.get_chain('L')]:
        coords = []
        for residue in chain:
            coords.append(residue.get_coord('CA'))
        centers.append(np.mean(coords, axis=0))

    energy = np.sqrt(((centers[0] - centers[2]) ** 2).sum()) + \
             np.sqrt(((centers[1] - centers[3]) ** 2).sum())  # RMSD

    return filename, energy


def display(args):
    with open(args.test_set, 'r') as fin:
        items = [json.loads(line) for line in fin.read().strip().split('\n')]
    # extract epitopes
    print_log('Loading epitopes')
    target_infos = []
    for item in tqdm(items):
        epitope, _ = get_interface(item['pdb_data_path'], item['antigen_chains'],
                                   #[item['heavy_chain'], item['light_chain']])
                                   item['toxin_chains'])
        pdb_id = item['pdb']
        epitope_def = [(chain_name, res.get_id()) for res, chain_name, _ in epitope]
        out_dir = os.path.join(args.save_dir, pdb_id)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        epitope_def_out = os.path.join(out_dir, pdb_id + '_epitope.json')
        with open(epitope_def_out, 'w') as fout:
            json.dump(epitope_def, fout)
        target_infos.append({
            'out_dir': out_dir,
            'epitope_def': epitope_def_out,
            'pdb_id': pdb_id,
            'origin_antibody_chains': item['toxin_chains'],
            # 'origin_antibody_chains': [item['heavy_chain'], item['light_chain']],
            'pdb': item['pdb_data_path']
        })
    print_log(f'{len(target_infos)} targets loaded')

    # extract framework templates
    print_log(f'Loading framework library from {args.library}')
    with open(args.library, 'r') as fin:   
        items = [json.loads(line) for line in fin.read().strip().split('\n')]
    fr_lib = []
    for item in tqdm(items):
        H, L = item['heavy_chain_seq'], item['light_chain_seq']
        # (start, end), cdrh3_seq = item['cdrh3_pos'], item['cdrh3_seq']
        # assert H[start:end + 1] == cdrh3_seq
        # H = H[:start] + '-' * len(cdrh3_seq) + H[end + 1:]
        fr_lib.append((('H', H), ('L', L)))
    print_log(f'{len(fr_lib)} frameworks loaded')
    if args.n_sample > len(fr_lib):
        print_log(f'Number of samples exceeds the size of framework library, set n_sample to {len(fr_lib)}', level='WARN')
        args.n_sample = len(fr_lib)

    # generate candidates for each target
    fr_idxs = [i for i in range(len(fr_lib))]
    for target_id, target_info in enumerate(target_infos):
        pdb_id = target_info['pdb_id']
        rand_fr_idx = np.random.choice(fr_idxs, size=args.n_sample, replace=False)
        rand_fr_lib = [fr_lib[i] for i in rand_fr_idx]
        print_log(f'Design for {pdb_id}, {target_id + 1} / {len(target_infos)}')
        start_time = time()

        pdbs = [target_info['pdb'] for _ in rand_fr_lib]
        epitope_defs = [target_info['epitope_def'] for _ in rand_fr_lib]
        identifiers = [pdb_id + f'_{i}' for i in rand_fr_idx]
        remove_chains = [target_info['origin_antibody_chains'] for _ in rand_fr_lib]
        out_dir=target_info['out_dir']
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
               auto_detect_cdrs=True)
        print_log(f'Finished generating {len(pdbs)} candidates, elapsed {round(time() - start_time, 2)}s')

        # calculate rough affinity energy (with high correlation but not the same scale)
        print_log('Start calculating rough affinity score')
        start_time = time()
        aff_log_file = os.path.join(out_dir, 'affinity.txt')
        aff_log = open(aff_log_file, 'w')
        results = []
        flexible_map={'H': 'ALL', 'L': 'ALL'}
        for chain, _ in json.load(open(epitope_defs[0])):
            flexible_map[chain] = 'ALL'
        pbar = tqdm(total=len(identifiers))
        pool = multiprocessing.Pool(args.num_workers)

        
        def affinity_callback(return_data):
            filename, energy = return_data
            pbar.update(1)
            aff_log.write(f'{filename} {energy}\n')
            aff_log.flush()
            results.append((filename, energy))

        for filename in identifiers:
            pool.apply_async(func=cal_affinity, args=((target_info, filename, flexible_map),), callback=affinity_callback)
        # for filename in tqdm(identifiers):
        #     f = os.path.join(target_info['out_dir'], filename + '.pdb')
        #     # openmm_relax(f, f)
        #     fast_relax(f, flexible_map=flexible_map)
        #     energy = interface_energy(f, ['H', 'L'])
        #     aff_log.write(f'{filename} {energy}\n')
        #     aff_log.flush()
        #     results.append((filename, energy))

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
    parser.add_argument('--test_set', type=str, required=True, help='Path to test set')
    parser.add_argument('--library', type=str, required=True, help='Path to framework library')
    parser.add_argument('--n_sample', type=int, required=True, help='Number of display samples')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save generated dataset')

    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for inference')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers to use')

    parser.add_argument('--gpu', type=int, default=-1, help='GPU to use, -1 for cpu')
    return parser.parse_args()


if __name__ == '__main__':
    display(parse())
