#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
from tqdm import tqdm
from os.path import splitext, basename
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import DataLoader

from generate import to_cplx
from data.pdb_utils import VOCAB, AgAbComplex, Protein, Peptide
from data.dataset import _generate_chain_data
from utils.relax import openmm_relax
from utils.logger import print_log
# from models.dyMEAN.predictor import Predictor


class Dataset(torch.utils.data.Dataset):
    def __init__(self, pdb, heavy_chain, light_chain, antigen_chains, num_residue_changes, cdr='H3') -> None:
        super().__init__()
        self.pdb = pdb
        self.num_residue_changes = num_residue_changes
        self.cdr = cdr
        cplx = AgAbComplex.from_pdb(pdb, heavy_chain, light_chain, antigen_chains)
        
        # generate antigen data
        ag_residues = []
        for residue, chain, i in cplx.get_epitope():
            ag_residues.append(residue)
        ag_data = _generate_chain_data(ag_residues, VOCAB.BOA)

        hc, lc = cplx.get_heavy_chain(), cplx.get_light_chain()
        hc_residues, lc_residues = [], []

        # generate heavy chain data
        for i in range(len(hc)):
            hc_residues.append(hc.get_residue(i))
        hc_data = _generate_chain_data(hc_residues, VOCAB.BOH)

        # generate light chain data
        for i in range(len(lc)):
            lc_residues.append(lc.get_residue(i))
        lc_data = _generate_chain_data(lc_residues, VOCAB.BOL)

        data = { key: np.concatenate([ag_data[key], hc_data[key], lc_data[key]], axis=0) \
                 for key in hc_data}  # <X, S, residue_pos>
        
        cmask = [0 for _ in ag_data['S']] + [0] + [1 for _ in hc_data['S'][1:]] + [0] + [1 for _ in lc_data['S'][1:]]
        smask = [0 for _ in range(len(ag_data['S']) + len(hc_data['S']) + len(lc_data['S']))]
        data['cmask'], data['smask'] = cmask, smask

        self.cdr_idxs = []
        cdrs = [self.cdr] if type(self.cdr) == str else self.cdr
        for cdr in cdrs:
            cdr_range = cplx.get_cdr_pos(cdr)
            offset = len(ag_data['S']) + 1 + (0 if cdr[0] == 'H' else len(hc_data['S']))
            for idx in range(offset + cdr_range[0], offset + cdr_range[1] + 1):
                self.cdr_idxs.append(idx)

        self.cplx, self.data = cplx, data

    def __getitem__(self, idx):
        data = deepcopy(self.data)
        num_residue_chain = min(self.num_residue_changes[idx], len(self.cdr_idxs))
        if num_residue_chain <= 0:
            num_residue_chain = np.random.randint(1, len(self.cdr_idxs))
        mask_idxs = np.random.choice(self.cdr_idxs, size=num_residue_chain, replace=False)
        for i in mask_idxs:
            data['smask'][i] = 1
        return data

    def __len__(self):
        return len(self.num_residue_changes)

    @classmethod
    def collate_fn(cls, batch):
        keys = ['X', 'S', 'smask', 'cmask', 'residue_pos']
        types = [torch.float, torch.long, torch.bool, torch.bool, torch.long]
        res = {}
        for key, _type in zip(keys, types):
            val = []
            for item in batch:
                val.append(torch.tensor(item[key], dtype=_type))
            res[key] = torch.cat(val, dim=0)
        lengths = [len(item['S']) for item in batch]
        res['lengths'] = torch.tensor(lengths, dtype=torch.long)
        return res


class ComplexSummary:
    def __init__(self, pdb, heavy_chain, light_chain, antigen_chains) -> None:
        self.pdb = pdb
        self.heavy_chain = heavy_chain
        self.light_chain = light_chain
        self.antigen_chains = antigen_chains


def optimize(
        ckpt,
        predictor_ckpt,
        gpu,
        cplx_summary,
        num_residue_changes,
        out_dir,
        batch_size=32,
        num_workers=4,
        optimize_steps=10,
        cdr_type='auto',
        enable_openmm_relax=True,
        mask_only=True,
        quiet=False
    ):
    '''
    :param num_residue_changes: number of residue in CDR to change. If <= 0 is given, randomly draw a number from uniform distribution
    :param cdr_type: type of cdr to be optimized. Can be auto, single cdr and multiple cdrs:
                        1. auto: automatically decided by the model checkpoint
                        2. single cdr: e.g. H3
                        3. multiple cdrs: e.g. ['L3', 'H3']
    '''
    # create out dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    pdb_id = splitext(basename(cplx_summary.pdb))[0]
    
    # load model
    device = torch.device('cpu' if gpu == -1 else f'cuda:{gpu}')
    # if path provided, load model, else the model is directly provided
    model = torch.load(ckpt) if type(ckpt) == type('str') else ckpt
    model.to(device)
    model.eval()
    predictor = torch.load(predictor_ckpt, map_location='cpu') if type(predictor_ckpt) == type('str') else predictor_ckpt
    predictor.to(device)
    predictor.eval()

    # generate dataset
    if cdr_type == 'auto':
        cdr_type = model.cdr_type
    if type(cdr_type) == str:
        cdr_type = [cdr_type]
    dataset = Dataset(**cplx_summary.__dict__, num_residue_changes=num_residue_changes, cdr=cdr_type)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                             num_workers=num_workers,
                             collate_fn=Dataset.collate_fn)
    # save original pdb
    dataset.cplx.to_pdb(os.path.join(out_dir, pdb_id + '_original.pdb'))
    
    # generate antibody
    idx, gen_pdbs, gen_cdrs = 0, [], []
    iterator = dataloader if quiet else tqdm(dataloader)
    for batch in iterator:
        # move data
        for k in batch:
            if hasattr(batch[k], 'to'):
                batch[k] = batch[k].to(device)
        # generate
        X, S, pmets = model.optimize_sample(**batch, predictor=predictor, opt_steps=optimize_steps, mask_only=mask_only)
        X, S, pmets = X.tolist(), S.tolist(), pmets.tolist()
        X_list, S_list = [], []
        cur_bid = -1
        # generate batch id
        lengths = batch['lengths']
        batch_id = torch.zeros_like(batch['S'])  # [N]
        batch_id[torch.cumsum(lengths, dim=0)[:-1]] = 1
        batch_id.cumsum_(dim=0)  # [N], item idx in the batch
        for i, bid in enumerate(batch_id):
            if bid != cur_bid:
                cur_bid = bid
                X_list.append([])
                S_list.append([])
            X_list[-1].append(X[i])
            S_list[-1].append(S[i])
                
        for i, (x, s) in enumerate(zip(X_list, S_list)):
            ori_cplx = deepcopy(dataset.cplx)
            cplx = to_cplx(ori_cplx, x, s)
            mod_pdb = os.path.join(out_dir, f'{pdb_id}_{idx}_{num_residue_changes[idx]}.pdb')

            cplx.to_pdb(mod_pdb)
            if enable_openmm_relax:
                print_log('Openmm relaxing...')
                # interface = cplx.get_epitope()
                # if_chain = cplx.antigen.get_chain_names()[0]
                # if_residues = [deepcopy(tup[0]) for tup in interface]
                # for i, residue in enumerate(if_residues):
                #     residue.id = (i + 1, ' ')
                # if_peptide = Peptide(if_chain, if_residues)
                #     
                # if_cplx = AgAbComplex(
                #     antigen=Protein(cplx.get_id(), {if_chain: if_peptide}),
                #     # antigen=Protein(cplx.get_id(), if_peptides),
                #     antibody=cplx.antibody,
                #     heavy_chain=cplx.heavy_chain,
                #     light_chain=cplx.light_chain
                # )


                # if_cplx.to_pdb(mod_pdb)
                openmm_relax(mod_pdb, mod_pdb,
                             excluded_chains=[cplx.heavy_chain, cplx.light_chain],
                             inverse_exclude=True)
                # if_cplx = AgAbComplex.from_pdb(mod_pdb, cplx.heavy_chain, cplx.light_chain, [if_chain])
                # cplx.antibody = if_cplx.antibody
                cplx = AgAbComplex.from_pdb(mod_pdb, cplx.heavy_chain, cplx.light_chain, cplx.antigen.get_chain_names())

            # cplx.to_pdb(mod_pdb)
            gen_cdr = ' '.join([cplx.get_cdr(cdr).get_seq() for cdr in cdr_type])
            ori_cdr = ' '.join([ori_cplx.get_cdr(cdr).get_seq() for cdr in cdr_type])
            gen_pdbs.append(mod_pdb)
            gen_cdrs.append(gen_cdr)
            change = 0
            for a, b, in zip(gen_cdr, ori_cdr):
                if a != b:
                    change += 1
            idx += 1
    return gen_pdbs, gen_cdrs


if __name__ == '__main__':
    ckpt = './checkpoints/cdrh3_opt.ckpt'
    predictor_ckpt = './checkpoints/cdrh3_ddg_predictor.ckpt'
    root_dir = './demos/data/1nca_opt'
    summary = ComplexSummary(
        pdb='./demos/data/1nca.pdb',
        heavy_chain='H',
        light_chain='L',
        antigen_chains=['N']
    )
    optimize(
        ckpt=ckpt,  # path to the checkpoint of the trained model
        predictor_ckpt=predictor_ckpt,  # path to the checkpoint of the trained ddG predictor
        gpu=0,      # the ID of the GPU to use
        cplx_summary=summary,   # summary of the complex as well as its PDB file
        num_residue_changes=[1, 2, 3, 4, 5],  # generate 5 samples, changing at most 1, 2, 3, 4, and 5 residues, respectively
        out_dir=root_dir,  # output directory
        batch_size=16,     # batch size
        num_workers=4,     # number of workers to use
        optimize_steps=50  # number of steps for gradient desend
    )
