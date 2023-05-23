#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import json
from tqdm import tqdm
from copy import deepcopy
from os.path import splitext, basename

import numpy as np
import torch
from torch.utils.data import DataLoader

from generate import to_cplx
from data.pdb_utils import Protein, VOCAB, Residue, AgAbComplex, Peptide
from data.dataset import _generate_chain_data
from data.framework_templates import ConserveTemplateGenerator
from utils.renumber import renumber_seq
from utils.relax import openmm_relax
from utils.logger import print_log
from configs import IMGT


class Dataset(torch.utils.data.Dataset):
    def __init__(self, pdbs, epitopes, frameworks, remove_chains=None,
                 cdr='H3', paratope='H3', auto_detect_cdrs=False) -> None:
        super().__init__()
        self.pdbs = pdbs
        self.epitopes = epitopes
        self.frameworks = frameworks
        self.remove_chains = remove_chains
        self.cdr = [cdr] if type(cdr) == str else cdr
        self.paratope = [paratope] if type(paratope) == str else paratope
        self.auto_detect_cdrs = auto_detect_cdrs

    def get_epitope(self, idx):
        pdb, epitope_def = self.pdbs[idx], self.epitopes[idx]
        prot = Protein.from_pdb(pdb)
        if self.remove_chains is not None:
            for chain in self.remove_chains[idx]:
                if chain in prot.peptides:
                    del prot.peptides[chain]

        with open(epitope_def, 'r') as fin:
            epitope = json.load(fin)
        to_str = lambda pos: f'{pos[0]}-{pos[1]}'
        epi_map = {}
        for chain_name, pos in epitope:
            if chain_name not in epi_map:
                epi_map[chain_name] = {}
            epi_map[chain_name][to_str(pos)] = True
        residues = []
        for chain_name in epi_map:
            chain = prot.get_chain(chain_name)
            for residue in chain:
                if to_str(residue.get_id()) in epi_map[chain_name]:
                    residues.append(residue)
        return residues, prot
    
    def generate_ab_chain(self, chain_seq: str):
        smask = [1 if s == '-' else 0 for s in chain_seq]
        seq = [VOCAB.idx_to_symbol(np.random.randint(0, VOCAB.get_num_amino_acid_type())) \
               if s == '-' else s for s in chain_seq]
        seq = ''.join(seq)
        fv, position, chain_type = renumber_seq(seq, scheme='imgt')
        start = seq.index(fv)
        end = start + len(fv)
        assert start != -1, f'fv not found for {chain_seq}'
        smask = smask[start:end]
        seq = seq[start:end]
        residues = []
        fake_coords = { atom: [0, 0, 0] for atom in VOCAB.backbone_atoms }
        for s, pos in zip(seq, position):
            residues.append(Residue(s, fake_coords, pos))
        if self.auto_detect_cdrs:
            for cdr in self.cdr:
                if not cdr.startswith(chain_type):
                    continue
                start, end = getattr(IMGT, cdr)
                for i, (pos_num, _) in enumerate(position):
                    if pos_num >= start and pos_num <= end:
                        smask[i] = 1
        return residues, smask

    def __getitem__(self, idx):
        framework = self.frameworks[idx]
        epitope_residues, antigen = self.get_epitope(idx)
        epitope_data = _generate_chain_data(epitope_residues, VOCAB.BOA)
        (h_id, h_seq), (l_id, l_seq) = framework[0], framework[1]
        # auto_detect_cdr = '-' not in h_seq and '-' not in l_seq
        # auto_detect_cdr = False
        hc_residues, hc_smask = self.generate_ab_chain(h_seq)
        hc_data = _generate_chain_data(hc_residues, VOCAB.BOH)
        lc_residues, lc_smask = self.generate_ab_chain(l_seq)
        lc_data = _generate_chain_data(lc_residues, VOCAB.BOL)
        data = { key: np.concatenate([epitope_data[key], hc_data[key], lc_data[key]], axis=0) \
                 for key in hc_data}
        cmask = [0 for _ in epitope_data['S']] + [0] + [1 for _ in hc_data['S'][1:]] + [0] + [1 for _ in lc_data['S'][1:]]
        smask = [0 for _ in epitope_data['S']] + [0] + hc_smask + [0] + lc_smask
        data['cmask'], data['smask'] = cmask, smask

        # related to template and cdrs
        cplx = AgAbComplex(
            antigen=antigen,
            antibody=Protein('', {h_id: Peptide(h_id, hc_residues), l_id: Peptide(l_id, lc_residues)}),
            heavy_chain=h_id, light_chain=l_id, numbering='imgt',
            skip_epitope_cal=True
        )

        paratope_mask = [0 for _ in range(len(epitope_data['S']) + len(hc_data['S']) + len(lc_data['S']))]
        for cdr in self.paratope:
            cdr_range = cplx.get_cdr_pos(cdr)
            offset = len(epitope_data['S']) + 1 + (0 if cdr[0] == 'H' else len(hc_data['S']))
            for idx in range(offset + cdr_range[0], offset + cdr_range[1] + 1):
                paratope_mask[idx] = 1
        data['paratope_mask'] = paratope_mask

        template = ConserveTemplateGenerator().construct_template(cplx, align=False)
        data['template'] = template
        data['cplx'] = cplx
        return data

    def __len__(self):
        return len(self.pdbs)

    @classmethod
    def collate_fn(cls, batch):
        keys = ['X', 'S', 'smask', 'cmask', 'paratope_mask', 'residue_pos', 'template']
        types = [torch.float, torch.long, torch.bool, torch.bool, torch.bool, torch.long, torch.float]
        res = {}
        for key, _type in zip(keys, types):
            val = []
            for item in batch:
                val.append(torch.tensor(item[key], dtype=_type))
            res[key] = torch.cat(val, dim=0)
        lengths = [len(item['S']) for item in batch]
        res['lengths'] = torch.tensor(lengths, dtype=torch.long)
        res['cplxes'] = [item['cplx'] for item in batch]
        return res


def design(ckpt, gpu, pdbs, epitope_defs, frameworks, out_dir,
           identifiers=None, remove_chains=None, enable_openmm_relax=True,
           auto_detect_cdrs=False, batch_size=32, num_workers=4):
    # create out dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if identifiers is None:
        identifiers = [splitext(basename(pdb))[0] for pdb in pdbs]
    # load model
    device = torch.device('cpu' if gpu == -1 else f'cuda:{gpu}')
    model = torch.load(ckpt)
    model.to(device)
    model.eval()

    # generate dataset
    dataset = Dataset(pdbs, epitope_defs, frameworks, remove_chains,
                      model.cdr_type, model.paratope, auto_detect_cdrs)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                             num_workers=num_workers,
                             collate_fn=Dataset.collate_fn)
    
    # generate antibody
    idx = 0
    for batch in tqdm(dataloader):
        with torch.no_grad():
            cplxes = batch['cplxes']
            del batch['cplxes']
            # move data
            for k in batch:
                if hasattr(batch[k], 'to'):
                    batch[k] = batch[k].to(device)
            # generate
            X, S, pmets = model.sample(**batch)
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
            ori_cplx = cplxes[i]
            cplx = to_cplx(ori_cplx, x, s)
            mod_pdb = os.path.join(out_dir, identifiers[idx] + '.pdb')
            cplx.to_pdb(mod_pdb)
            if enable_openmm_relax:
                print_log('Openmm relaxing...')
                interface = cplx.get_epitope()
                if_chain = cplx.antigen.get_chain_names()[0]
                if_residues = [deepcopy(tup[0]) for tup in interface]
                for i, residue in enumerate(if_residues):
                    residue.id = (i + 1, ' ')
                if_peptide = Peptide(if_chain, if_residues)
                if_cplx = AgAbComplex(
                    antigen=Protein(cplx.get_id(), {if_chain: if_peptide}),
                    antibody=cplx.antibody,
                    heavy_chain=cplx.heavy_chain,
                    light_chain=cplx.light_chain
                )
                if_cplx.to_pdb(mod_pdb)
                openmm_relax(mod_pdb, mod_pdb,
                             excluded_chains=[cplx.heavy_chain, cplx.light_chain],
                             inverse_exclude=True)
                if_cplx = AgAbComplex.from_pdb(mod_pdb, cplx.heavy_chain, cplx.light_chain, [if_chain])
                cplx.antibody = if_cplx.antibody
            
            cplx.to_pdb(mod_pdb)
            print_log(f'{mod_pdb} saved.')
            idx += 1


if __name__ == '__main__':
    ckpt = './checkpoints/cdrh3_design.ckpt'
    root_dir = './demos/data'
    pdbs = [os.path.join(root_dir, '7l2m.pdb') for _ in range(4)]
    toxin_chains = ['E', 'e', 'F', 'f']
    remove_chains = [toxin_chains for _ in range(4)]
    epitope_defs = [os.path.join(root_dir, c + '_epitope.json') for c in toxin_chains]
    identifiers = [f'{c}_antibody' for c in toxin_chains]
    
    # use '-' for masking amino acids
    frameworks = [
        (
            ('H', 'QVQLKESGPGLLQPSQTLSLTCTVSGISLSDYGVHWVRQAPGKGLEWMGIIGHAGGTDYNSNLKSRVSISRDTSKSQVFLKLNSLQQEDTAMYFC----------WGQGIQVTVSSA'),
            ('L', 'YTLTQPPLVSVALGQKATITCSGDKLSDVYVHWYQQKAGQAPVLVIYEDNRRPSGIPDHFSGSNSGNMATLTISKAQAGDEADYYCQSWDGTNSAWVFGSGTKVTVLGQ')
        ) \
        for _ in pdbs
    ]  # the first item of each tuple is heavy chain, the second is light chain

    design(ckpt=ckpt,  # path to the checkpoint of the trained model
           gpu=0,      # the ID of the GPU to use
           pdbs=pdbs,  # paths to the PDB file of each antigen (here antigen is all TRPV1)
           epitope_defs=epitope_defs,  # paths to the epitope definitions
           frameworks=frameworks,      # the given sequences of the framework regions
           out_dir=root_dir,           # output directory
           identifiers=identifiers,    # name of each output antibody
           remove_chains=remove_chains,# remove the original ligand
           enable_openmm_relax=True,   # use openmm to relax the generated structure
           auto_detect_cdrs=False)  # manually use '-'  to represent CDR residues