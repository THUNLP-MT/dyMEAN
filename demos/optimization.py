#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import argparse
import random
from multiprocessing import Pool
from copy import copy
from tqdm import tqdm

# from utils.relax import openmm_relax
from api.optimize import optimize, ComplexSummary
from utils.pyrosetta_tools import interface_energy, fast_relax
from utils.logger import print_log
from evaluation.lddt import lddt
from evaluation.pred_ddg import pred_ddg
from data.pdb_utils import AgAbComplex
from configs import CACHE_DIR

# checkers
from checkers import RELAX_CHECKERS, EVAL_CHECKERS, PRE_EVAL_CHECKERS


def interact_pair(gen_pdb):
    cplx = AgAbComplex.from_pdb(gen_pdb, 'A', 'L', ['B'])
    delta = [5, 53, 83, 106, 111, 112, 118, 126, 143, 150, 152, 156, 167]
    xbb = [5] + [x - 1 for x in delta[1:]]
    cdr = cplx.get_cdr()
    cov = cplx.antigen.get_chain('B')
    forbid_res = delta if len(cov) == 193 else xbb
    cnt = 0
    for i in forbid_res:
        ag_res = cov.get_residue(i)
        for ab_res in cdr:
            if ag_res.dist_to(ab_res) < 5.0:
                cnt += 1
    return cnt


def relax(pdb, out_pdb):
    fast_relax(pdb, out_pdb, flexible_map={'L': 'ALL', 'A': [0, 1, 2, 3, 5, 6, 7, 8, 9, 10]})


def check_relax(ori_cdr, gen_pdb, gen_cdr):
    # 1. checkers
    for checker in RELAX_CHECKERS:
        passed, reason = checker(ori_cdr, gen_pdb, gen_cdr)
        if not passed:
            return False, f'Failed to pass {checker.__name__}. {reason}'

    # 2. relax
    os.system(f'sed -i "s/MASK/M3L/g" {gen_pdb}')
    relax(gen_pdb, gen_pdb)
    return True, ''


def check_affinity(summary, gen_pdb, base_metric):
    for checker in PRE_EVAL_CHECKERS:
        passed, reason = checker(summary, gen_pdb)
        if not passed:
            return False, f'Faild to pass {checker.__name__}. {reason}'

    ddg = pred_ddg(summary.pdb, gen_pdb)
    # inter_pair = interact_pair(gen_pdb)
    aff = interface_energy(gen_pdb, f'{summary.heavy_chain}{summary.light_chain}_{"".join(summary.antigen_chains)}', return_dict=True)
    detail = [ddg, aff['dG_separated'], aff['dG_separated/dSASAx100']]
    # detail = [ddg, 0, 0]
    # detail = [ddg, inter_pair]

    # checkers
    for checker in EVAL_CHECKERS:
        passed, reason = checker(summary, gen_pdb, base_metric, detail)
        if not passed:
            return False, f'Faild to pass {checker.__name__}. {reason}'
    
    return True, detail


def write_log(log, values):
    # identity  CDRL1   CDRL2...CDRH3   metric1 metric2 ... metricN change
    log.write('\t'.join([str(val) for val in values]))
    log.write('\n')
    log.flush()


def main(args):
    root_dir = args.save_dir
    summary = ComplexSummary(
        pdb=args.pdb,
        heavy_chain=args.heavy_chain,
        light_chain=args.light_chain,
        antigen_chains=args.antigen_chains
    )
    ori_cplx = AgAbComplex.from_pdb(summary.pdb, summary.heavy_chain, summary.light_chain, summary.antigen_chains)
    ori_cdr = ' '.join([ori_cplx.get_cdr(cdr).get_seq() for cdr in args.cdr_type])
    candidates, num_cand = [], args.n_sample
    idx, num_trial, trial_size = 0, 0, 96
    cand_dir = os.path.join(root_dir, 'candidates')
    if not os.path.exists(cand_dir):
        os.makedirs(cand_dir)
    summary.pdb = os.path.join(cand_dir, 'relax.pdb')
    relax(args.pdb, summary.pdb)
    # os.system(f'cp {args.pdb} {summary.pdb}')
    aff = interface_energy(summary.pdb, f'{summary.heavy_chain}{summary.light_chain}_{"".join(summary.antigen_chains)}', return_dict=True)
    # inter_pair = interact_pair(summary.pdb)
    log = open(os.path.join(cand_dir, 'log.txt'), 'w')
    # write_log(log, ['candidate'] + args.cdr_type + ['ddG', 'pair', 'change'])
    write_log(log, ['candidate'] + args.cdr_type + ['ddG', 'dG', 'dG/dSASAx100', 'change'])
    history_best, best_pdb = [0, aff['dG_separated'], aff['dG_separated/dSASAx100']], None
    # history_best = [0, inter_pair]
    base_metric = copy(history_best)
    write_log(log, ['native'] + [ori_cplx.get_cdr(cdr).get_seq() for cdr in args.cdr_type] + history_best + [0])

    summary.pdb = args.pdb
    while len(candidates) < num_cand:
        num_trial += 1
        print_log(f'Trial {num_trial} start with {trial_size} samples! Generating...')
        gen_pdbs, gen_cdrs = optimize(
            ckpt=args.ckpt,
            predictor_ckpt=args.predictor_ckpt,
            gpu=args.gpu,
            cplx_summary=summary,
            num_residue_changes=[random.randint(1, 9) for _ in range(trial_size)],
            out_dir=root_dir,
            batch_size=8,
            num_workers=4,
            optimize_steps=50,
            cdr_type=args.cdr_type,
            quiet=True
        )
        print_log(f'Finished generation, starting evaluation...')
        pool = Pool(args.num_workers)

        relax_results = []
        
        for pdb, cdr in zip(gen_pdbs, gen_cdrs):
            relax_results.append((
                pdb, cdr, pool.apply_async(func=check_relax, args=(ori_cdr, pdb, cdr))
            ))
    
        for finish_id in range(len(gen_pdbs)):
            idx += 1
            results = relax_results[finish_id]
            results[-1].wait()
            (pdb, cdr), (relaxed, reason) = results[:-1], results[-1].get()
            if not relaxed:
                accept, detail = relaxed, reason
            else:
                accept, detail = check_affinity(summary, pdb, base_metric)
            
            if not accept:
                print_log(f'candidate {idx} failed, reason: {detail}')
                continue
            else:
                history_best = [min(val1, val2) for val1, val2 in zip(history_best, detail)]
                print_log(f'candidate {idx} passed check! CDR: {cdr}, metric: {[round(val, 2) for val in detail]}, ' + \
                          f'history best: {[round(val, 2) for val in history_best]}, progress: {len(candidates) + 1} / {num_cand}')
                save_path = os.path.join(cand_dir, f'candidate_{idx}.pdb')
                candidates.append(save_path)
                os.system(f'mv {pdb} {save_path}')
                change = 0
                for a, b in zip(ori_cdr, cdr):
                    if a != b:
                        change += 1
                write_log(log, [idx] + cdr.split() + detail + [change])

        pool.close()
        pool.join()

        print_log(f'Finished trial {num_trial}!')

    log.close()

    # sort log by ensemble of all metrics
    log_path = os.path.join(cand_dir, 'log.txt')
    num_metric = len(base_metric)
    with open(log_path, 'r') as fin:
        lines = fin.readlines()
    values = [l.split('\t') for l in lines[2:]]  # 0 is head, 1 is native data
    ranks = []
    for i in range(num_metric):
        offset = 1 + num_metric - i  # 1 for the n_change at the end
        cur_metrics = [float(val[-offset]) for val in values]
        rank = list(range(len(cur_metrics)))
        sort_idx = sorted(rank, key=lambda i: cur_metrics[i])  # the lower, the better
        r, last_idx, cnt = 1, -1, 0
        for idx in sort_idx:
            cnt += 1
            if last_idx == -1 or cur_metrics[idx] != cur_metrics[last_idx]:
                r = cnt
            last_idx = idx
            rank[idx] = r
        ranks.append(rank)

    avg_rank = [sum([ranks[metric_idx][cand_idx] for metric_idx in range(num_metric)]) / num_metric for cand_idx in range(len(values))]
    sorted_cand_idx = sorted(list(range(len(values))), key=lambda i: avg_rank[i])
    with open(log_path, 'w') as fout:
        fout.write(lines[0].strip() + '\trank\n')
        fout.write(lines[1].strip() + '\t-\n')
        for i in sorted_cand_idx:
            fout.write(lines[2 + i].strip() + f'\t{avg_rank[i]}\n')


def parse():
    parser = argparse.ArgumentParser(description='Generate antibodies given epitopes')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--predictor_ckpt', type=str, required=True, help='Path to predictor checkpoint')
    parser.add_argument('--n_sample', type=int, required=True, help='Number of optimized samples')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save generated dataset')

    parser.add_argument('--pdb', type=str, required=True, help='Path to the pdb of the complex')
    parser.add_argument('--heavy_chain', type=str, required=True, help='Chain id of the heavy chain')
    parser.add_argument('--light_chain', type=str, required=True, help='Chain id of the light chain')
    parser.add_argument('--antigen_chains', type=str, required=True, nargs='+', help='Chain ids of the antigen')
    parser.add_argument('--cdr_type', type=str, default=['H3'], choices=['H1', 'H2', 'H3', 'L1', 'L2', 'L3'],
                        nargs='+', help='Path to the pdb of the complex')

    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for inference')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers to use')

    parser.add_argument('--gpu', type=int, default=0, help='GPU to use, -1 for cpu')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse())
