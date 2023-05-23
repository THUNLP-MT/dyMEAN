#!/usr/bin/python
# -*- coding:utf-8 -*-
'''
Generate dataset with (h, affinity) to train a predictor
'''
import os
import json
import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch_scatter import scatter_mean

from generate import to_cplx
from evaluation.pred_ddg import pred_ddg
from data.dataset import E2EDataset
from utils.logger import print_log
from utils.random_seed import setup_seed
setup_seed(2023)


def generate(args):
    # load model
    model = torch.load(args.ckpt, map_location='cpu')
    device = torch.device('cpu' if args.gpu == -1 else f'cuda:{args.gpu}')
    model.to(device)
    model.eval()

    # model_type
    print_log(f'Model type: {type(model)}')
    
    # cdr type
    cdr_type = model.cdr_type
    print_log(f'CDR type: {cdr_type}')

    # load test set
    test_set = E2EDataset(args.summary_json, cdr=cdr_type, paratope=['H3'])  # paratope definition is not used here
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=args.num_workers,
                             collate_fn=E2EDataset.collate_fn)
    with open(args.summary_json, 'r') as fin:
        items = [json.loads(l) for l in fin.read().strip().split('\n')]
        items = { item['pdb']: item for item in items }


    # create save dir
    if args.save_dir is None:
        save_dir = '.'.join(args.ckpt.split('.')[:-1]) + '_results'
    else:
        save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    idx = 0
    # write done the dataset
    data_file = os.path.join(save_dir, 'data.jsonl')
    data_out = open(data_file, 'w')
    for batch in tqdm(test_loader):
        with torch.no_grad():
            # move data
            for k in batch:
                if hasattr(batch[k], 'to'):
                    batch[k] = batch[k].to(device)
            # generate
            del batch['xloss_mask']
            del batch['paratope_mask']
            del batch['template']
            X, S, pmets, H = model.sample(**batch, return_hidden=True)
            X, S, pmets = X.tolist(), S.tolist(), pmets.tolist()
            X_list, S_list = [], []
            cur_bid = -1
            if 'bid' in batch:
                batch_id = batch['bid']
            else:
                lengths = batch['lengths']
                batch_id = torch.zeros_like(batch['S'])  # [N]
                batch_id[torch.cumsum(lengths, dim=0)[:-1]] = 1
                batch_id.cumsum_(dim=0)  # [N], item idx in the batch
            H_list = scatter_mean(H, batch_id, dim=0).tolist() # [bs, hidden_size]
            
            for i, bid in enumerate(batch_id):
                if bid != cur_bid:
                    cur_bid = bid
                    X_list.append([])
                    S_list.append([])
                X_list[-1].append(X[i])
                S_list[-1].append(S[i])
                
        for i, (x, s, h) in enumerate(zip(X_list, S_list, H_list)):
            ori_cplx = test_set.data[idx]
            pdb_id = ori_cplx.get_id().split('(')[0]
            cplx = to_cplx(ori_cplx, x, s)
            mod_pdb = os.path.join(save_dir, pdb_id + '.pdb')
            cplx.to_pdb(mod_pdb)
            # get affinity
            mod_cdr = ''.join([cplx.get_cdr(cdr).get_seq() for cdr in cdr_type])
            ori_cdr = ''.join([items[pdb_id][f'cdr{cdr.lower()}_seq'] for cdr in cdr_type])
            if mod_cdr == ori_cdr:
                ddg = 0
            else:
                try:
                    ddg = pred_ddg(items[pdb_id]['pdb_data_path'], mod_pdb)
                except Exception:
                    print(items[pdb_id]['pdb_data_path'], mod_pdb)
                    idx += 1
                    continue
            data_out.write(json.dumps({
                    'pdb': pdb_id,
                    'ddg': ddg,
                    'hidden': h,
                }) + '\n')
            data_out.flush()
            idx += 1
    data_out.close()

    print_log(f'Generated data written to {data_file}')


def parse():
    parser = argparse.ArgumentParser(description='Generate antibodies given epitopes')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--summary_json', type=str, required=True, help='Path to summary file of the dataset')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save generated dataset')

    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for inference')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers to use')

    parser.add_argument('--gpu', type=int, default=-1, help='GPU to use, -1 for cpu')
    return parser.parse_args()


if __name__ == '__main__':
    generate(parse())
