#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import json
import argparse
from tqdm import tqdm
from shutil import rmtree
import logging

logging.disable('INFO')

from diffab.tools.runner.design_for_pdb import design_for_pdb


class Arg:
    def __init__(self, pdb, heavy, light, config, out_root):
        self.pdb_path = pdb
        self.heavy = heavy
        self.light = light
        self.no_renumber = True
        self.config = config
        self.out_root = out_root
        self.tag = ''
        self.seed = 0
        self.device = 'cuda'
        self.batch_size = 16


def main(args):
    # load dataset
    with open(args.dataset, 'r') as fin:
        lines = fin.read().strip().split('\n')

    out_dir = os.path.join(os.path.split(args.config)[0], 'results') if args.out_dir is None else args.out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for line in tqdm(lines):
        item = json.loads(line)
        pdb = item['pdb_data_path']
        heavy, light = item['heavy_chain'], item['light_chain']
        design_for_pdb(Arg(pdb, heavy, light, args.config, out_dir))

    tmp_dir = os.path.join(out_dir, 'codesign_single')
    for f in os.listdir(tmp_dir):
        pdb_id = f[:4]
        pdb_file = os.path.join(tmp_dir, f, 'H_CDR3', '0000.pdb')
        tgt_file = os.path.join(out_dir, pdb_id + '_generated.pdb')
        os.system(f'cp {pdb_file} {tgt_file}')

    rmtree(tmp_dir)


def parse():
    parser = argparse.ArgumentParser(description='generation by diffab')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--out_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--config', type=str, required=True, help='config to the diffab model')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse())