#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from data.dataset import EquiAACDataset
from generate import set_cdr


def main(args):
    # load dataset
    test_set = EquiAACDataset(args.dataset)
    test_set.mode = '111'
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, num_workers=args.num_workers,
        shuffle=False, collate_fn=test_set.collate_fn)
    
    # load model
    model = torch.load(args.ckpt, map_location='cpu')
    device = torch.device('cpu' if args.gpu==-1 else f'cuda:{args.gpu}')
    model.to(device)
    model.eval()

    gen_seqs, gen_xs = [], []
    for batch in tqdm(test_loader):
        with torch.no_grad():
            _, seqs, xs, _, _ = model.infer(batch, device)
        gen_seqs.extend(seqs)
        gen_xs.extend(xs)

    out_dir = os.path.join(os.path.split(args.ckpt)[0], 'results') if args.out_dir is None else args.out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for cplx, seq, residue_x in tqdm(zip(test_set.data, gen_seqs, gen_xs), total=len(gen_seqs)):
        new_cplx = set_cdr(cplx, seq, residue_x, args.cdr_type)
        new_cplx.to_pdb(os.path.join(out_dir, cplx.get_id() + '_generated.pdb'))


def parse():
    parser = argparse.ArgumentParser(description='generation by MEAN')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to the checkpoint')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for dataloader')
    parser.add_argument('--cdr_type', type=str, default='H3', help='CDR type',
                        choices=['H3'])
    parser.add_argument('--gpu', type=int, default=-1, help='-1 for cpu')
    parser.add_argument('--out_dir', type=str, default=None, help='Output directory')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse())