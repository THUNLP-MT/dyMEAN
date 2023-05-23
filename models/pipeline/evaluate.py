#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import json
import argparse

from tqdm.contrib.concurrent import process_map

from cal_metrics import cal_metrics
from data.pdb_utils import AgAbComplex, Peptide, Protein
from configs import CACHE_DIR
from utils.logger import print_log


def main(args):
    with open(args.summary_json, 'r') as fin:
        summary = fin.read().strip().split('\n')
    inputs, pdbs, tmp_files = [], [], []
    for item in summary:
        item = json.loads(item)
        H, L, A = item['heavy_chain'], item['light_chain'], item['antigen_chains']
        pdb = item['pdb']
        mod_pdb = os.path.join(args.gen_dir, pdb + '_generated.pdb')
        # mod_pdb = os.path.join(args.gen_dir, pdb + '_gt_align.pdb')
        ref_pdb = os.path.join(args.ref_dir, pdb + '_original.pdb')
        if not os.path.exists(mod_pdb):
            print_log(f'{mod_pdb} not exists!', level='ERROR')
            continue
        try:
            mod_cplx = AgAbComplex.from_pdb(mod_pdb, H, L, A)
        except Exception as e:
            print_log(f'parse {mod_pdb} failed for {e}', level='ERROR')
            continue
        ref_cplx = AgAbComplex.from_pdb(ref_pdb, H, L, A)

        # tmp_cplx = Protein.from_pdb(os.path.join(args.gen_dir, '..', 'cdr_out_2', pdb + '.pdb_pred.pdb'))
        # s, e = mod_cplx.get_cdr_pos()
        # for aaa in range(e - s + 1):
        #     mod_cplx.antibody.peptides[H].set_residue_coord(s + aaa, tmp_cplx.peptides['H'].residues[aaa].coordinate)

        mod_revised, ref_revised = False, False
        for chain_name in [H, L]:
            mod_chain = mod_cplx.antibody.get_chain(chain_name)
            ref_chain = ref_cplx.antibody.get_chain(chain_name)
            if len(mod_chain) != len(ref_chain):
                print_log(f'{pdb} chain {chain_name} length not consistent: {len(mod_chain)} != {len(ref_chain)}. Trying to align by position number.', level='WARN')

                mod_residues, ref_residues = [], []
                pos_map = {'-'.join(str(a) for a in res.get_id()): False for res in ref_chain}
                for res in mod_chain:
                    _id = '-'.join(str(a) for a in res.get_id())
                    if _id in pos_map:
                        pos_map[_id] = True
                        mod_residues.append(res)
                for res in ref_chain:
                    _id = '-'.join(str(a) for a in res.get_id())
                    if pos_map[_id]:
                        ref_residues.append(res)
                mod_chain = Peptide(mod_chain.get_id(), mod_residues)
                ref_chain = Peptide(ref_chain.get_id(), ref_residues)
                mod_revised, ref_revised = True, True
                print_log(f'{pdb} chain {chain_name} length after aligned: {len(mod_chain)} == {len(ref_chain)}', level='WARN')
                mod_cplx.antibody.peptides[chain_name] = mod_chain
                ref_cplx.antibody.peptides[chain_name] = ref_chain

        if mod_revised:
            tmp_mod_pdb = os.path.join(CACHE_DIR, pdb + '_mod.pdb')
            mod_cplx.to_pdb(tmp_mod_pdb)
            mod_pdb = tmp_mod_pdb
            tmp_files.append(tmp_mod_pdb)
        if ref_revised:
            tmp_ref_pdb = os.path.join(CACHE_DIR, pdb + '_ref.pdb')
            ref_cplx.to_pdb(tmp_ref_pdb)
            ref_pdb = tmp_ref_pdb
            tmp_files.append(tmp_ref_pdb)

        inputs.append((mod_pdb, ref_pdb, H, L, A, args.cdr_type, args.sidechain_packing))
        pdbs.append(pdb)

    metrics = process_map(cal_metrics, inputs, max_workers=4)
    for name in metrics[0]:
        vals = [item[name] for item in metrics]
        print(f'{name}: {sum(vals) / len(vals)}')
        lowest_i = min([i for i in range(len(vals))], key=lambda i: vals[i])
        highest_i = max([i for i in range(len(vals))], key=lambda i: vals[i])
        print(f'\tlowest: {vals[lowest_i]}, pdb: {pdbs[lowest_i]}')
        print(f'\thighest: {vals[highest_i]}, pdb: {pdbs[highest_i]}')

    for f in tmp_files:
        os.remove(f)


def parse():
    parser = argparse.ArgumentParser(description='evaluate AAR, TM-score, RMSD, LDDT')
    parser.add_argument('--summary_json', type=str, required=True, help='Path to the summary in json format providing H/L/antigen')
    parser.add_argument('--ref_dir', type=str, required=True, help='Path to the reference data')
    parser.add_argument('--gen_dir', type=str, required=True, help='Path to the generated data')
    parser.add_argument('--cdr_type', type=str, default='H3', help='Type of CDR',
                        choices=['H3'])
    parser.add_argument('--sidechain_packing', action='store_true', help='Do sidechain packing with rosetta')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse())