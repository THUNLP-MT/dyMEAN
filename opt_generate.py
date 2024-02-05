#!/usr/bin/python
# -*- coding:utf-8 -*-
'''
generate optimized candidates
'''
import os
import json
import argparse
from tqdm import tqdm

import numpy as np
import torch

from api.optimize import optimize, ComplexSummary
from evaluation.pred_ddg import pred_ddg, foldx_ddg, foldx_minimize_energy
from utils.relax import openmm_relax
from utils.logger import print_log
from utils.random_seed import setup_seed


def opt_generate(args):
    # cdr type
    cdr_type = args.cdr_type
    print(f'CDR type: {cdr_type}')

    # load model
    model = torch.load(args.ckpt, map_location='cpu')
    print(f'Model type: {type(model)}')
    device = torch.device('cpu' if args.gpu == -1 else f'cuda:{args.gpu}')
    model.to(device)
    model.eval()
    predictor = torch.load(args.predictor_ckpt, map_location='cpu')
    predictor.to(device)
    predictor.eval()

    with open(args.summary_json, 'r') as fin:
        items = [json.loads(line) for line in fin.read().strip().split('\n')]

    # create save dir
    if args.save_dir is None:
        save_dir = '.'.join(args.ckpt.split('.')[:-1]) + f'_{args.num_residue_changes}_opt_results'
    else:
        save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    log = open(os.path.join(save_dir, 'log.txt'), 'w')
    best_scores, success = [], []
    changes = []
    for item_id, item in enumerate(items):
        summary = ComplexSummary(
            pdb=item['pdb_data_path'],
            heavy_chain=item['heavy_chain'],
            light_chain=item['light_chain'],
            antigen_chains=item['antigen_chains']
        )
        pdb_id = item['pdb']
        out_dir = os.path.join(save_dir, pdb_id)
        print_log(f'Optimizing {pdb_id}, {item_id + 1} / {len(items)}')
        gen_pdbs, gen_cdrs = optimize(
            ckpt=model,
            predictor_ckpt=predictor,
            gpu=args.gpu,
            cplx_summary=summary,
            num_residue_changes=[args.num_residue_changes for _ in range(args.n_samples)],
            out_dir=out_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            enable_openmm_relax=False,  # for fast evaluation
            optimize_steps=args.num_optimize_steps,
            mask_only=args.use_foldx
        )

        ori_cdr, ori_pdb, scores = item[f'cdr{cdr_type.lower()}_seq'], summary.pdb, []

        item_log = open(os.path.join(out_dir, 'detail.txt'), 'w')
        different_cnt, cur_changes = 0, []
        for gen_pdb, gen_cdr in tqdm(zip(gen_pdbs, gen_cdrs), total=len(gen_pdbs)):
            change_cnt = 0
            if gen_cdr != ori_cdr:
                if args.use_foldx:
                    gen_pdb = openmm_relax(gen_pdb, gen_pdb)
                    gen_pdb = foldx_minimize_energy(gen_pdb)
                    try:
                        score = foldx_ddg(ori_pdb, gen_pdb, summary.antigen_chains, [summary.heavy_chain, summary.light_chain])
                    except ValueError as e:
                        print(e)
                        score = 0
                else:
                    score = pred_ddg(ori_pdb, gen_pdb)
                # inputs.append((gen_pdb, summary, ori_dg, interface))
                different_cnt += 1
                for a, b in zip(gen_cdr, ori_cdr):
                    if a != b:
                        change_cnt += 1
            else:
                # continue
                score = 0
            scores.append(score)
            cur_changes.append(change_cnt)

        avg_change = sum(cur_changes) / different_cnt
        print_log(f'obtained {different_cnt} candidates, average change {avg_change}')
        
        sucess_rate = sum(1 if s < 0 else 0 for s in scores) / len(scores)
        success.append(sucess_rate)
        mean_score = round(np.mean(scores), 3)
        best_score_idx = min([k for k in range(len(scores))], key=lambda k: scores[k])
        best_scores.append(scores[best_score_idx])
        changes.append(cur_changes[best_score_idx])
        message = f'{pdb_id}: mean ddg {mean_score}, best ddg {round(scores[best_score_idx], 3)}, diff cnt {different_cnt}, success rate {sucess_rate}, change: {cur_changes[best_score_idx]}, sample {gen_pdbs[best_score_idx]}\n'
        item_log.write(message)
        item_log.close()
        
        log.write(message)
        log.flush()
        
        print_log(message)

    final_message = f'average best scores: {np.mean(best_scores)}, IMP: {np.mean(success)}, changes: {np.mean(changes)}'
    print_log(final_message)
    log.write(final_message)
    log.close()


def parse():
    parser = argparse.ArgumentParser(description='Optimize antibody')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--predictor_ckpt', type=str, required=True, help='Path to predictor checkpoint')
    parser.add_argument('--use_foldx', action='store_true', help='Use foldx to predict ddg')
    parser.add_argument('--cdr_type', type=str, default='H3', help='The type of CDR to optimize (only support single CDR)')

    parser.add_argument('--n_samples', type=int, default=100, help='Number of samples to generate')
    parser.add_argument('--num_residue_changes', type=int, default=0, help='Number of residues to chain, <= 0 for random number')
    parser.add_argument('--num_optimize_steps', type=int, default=20, help='Number of optimization steps')

    parser.add_argument('--summary_json', type=str, required=True, help='Path to summary file of the dataset')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save generated dataset')

    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for inference')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers to use')

    parser.add_argument('--gpu', type=int, default=-1, help='GPU to use, -1 for cpu')
    return parser.parse_args()


if __name__ == '__main__':
    from utils.random_seed import setup_seed
    setup_seed(2023)
    opt_generate(parse())
