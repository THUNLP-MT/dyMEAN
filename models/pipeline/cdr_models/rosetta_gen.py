#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import json
import argparse
from tqdm import tqdm

from configs import Rosetta_DIR
from data.pdb_utils import AgAbComplex, Protein

def main(args):
    with open(args.dataset, 'r') as fin:
        summary = fin.read().strip().split('\n')

    designer = os.path.join(Rosetta_DIR, 'antibody_designer.static.linuxgccrelease')
    converter = os.path.join(Rosetta_DIR, 'antibody_numbering_converter.static.linuxgccrelease') 
    cdr = args.cdr_type

    out_dir = os.path.join(os.path.split(args.ckpt)[0], 'results') if args.out_dir is None else args.out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for item in tqdm(summary):
        item = json.loads(item)
        pdb, pdb_path = item['pdb'], item['renumbered']
        p = os.popen(f'cd {out_dir}; nohup {designer} -s {pdb_path} -primary_cdrs {cdr} -graft_design_cdrs {cdr} -light_chain lambda -random_start -nstruct 1 -overwrite -check_cdr_chainbreaks false')
        p.read()
        p.close()
        tmp_filename = os.path.split(pdb_path)[-1][:-4] + '_0001.pdb'
        p = os.popen(f'cd {out_dir}; nohup {converter} -s {tmp_filename} -input_ab_scheme AHO -output_ab_scheme IMGT -overwrite; rm {tmp_filename}')
        p.read()
        p.close()
        tmp_filename2 = tmp_filename[:-4] + '_0001.pdb'
        if not os.path.exists(os.path.join(out_dir, tmp_filename2)):
            print(f'optimizing {pdb} failed')
            continue
        protein = Protein.from_pdb(os.path.join(out_dir, tmp_filename2))
        peptides = {}
        for chain_name, chain in protein:
            ori_id = item['mapping'][chain_name]
            chain.set_id(ori_id)
            peptides[ori_id] = chain
        new_prot = Protein(pdb, peptides)
        new_prot.to_pdb(os.path.join(out_dir, pdb + '_generated.pdb'))
        os.system(f'rm {os.path.join(out_dir, tmp_filename2)}')


def parse():
    parser = argparse.ArgumentParser(description='generation by Rosetta')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--cdr_type', type=str, default='H3', help='CDR type',
                        choices=['H3'])
    parser.add_argument('--out_dir', type=str, default=None, help='Output directory')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse())