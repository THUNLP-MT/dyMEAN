#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import json
import argparse
from tqdm import tqdm

import numpy as np

from .pdb_utils import VOCAB, AgAbComplex
from evaluation.rmsd import compute_rmsd, kabsch
from utils.singleton import singleton
from utils.logger import print_log


@singleton
class ConserveTemplateGenerator:
    def __init__(self, json_path=None):
        if json_path is None:
            folder = os.path.split(__file__)[0]
            json_path = os.path.join(folder, 'template.json')
        with open(json_path, 'r') as fin:
            self.template_map = json.load(fin)
    
    def _chain_template(self, cplx: AgAbComplex, poses, n_channel, heavy=True):
        chain = cplx.get_heavy_chain() if heavy else cplx.get_light_chain()
        chain_name = 'H' if heavy else 'L'
        hit_map = { pos: False for pos in poses }
        X, hit_index = [], []
        for i, residue in enumerate(chain):
            pos, _ = residue.get_id()
            pos = str(pos)
            if pos in hit_map:
                coord = self.template_map[chain_name][pos]  # N, CA, C, O
                ca, num_sc = coord[1], n_channel - len(coord)
                coord.extend([ca for _ in range(num_sc)])
                hit_index.append(i)
                coord = np.array(coord)
            else:
                coord = [[0, 0, 0] for _ in range(n_channel)]
            X.append(coord)
        # uniform distribution between residues and extension at two ends
        for left_i, right_i in zip(hit_index[:-1], hit_index[1:]):
            left, right = X[left_i], X[right_i]
            span, index_span = right - left, right_i - left_i
            span = span / index_span
            for i in range(left_i + 1, right_i):
                X[i] = X[i - 1] + span
        # start and end
        if hit_index[0] != 0:
            left_i = hit_index[0]
            span = X[left_i] - X[left_i + 1]
            for i in reversed(range(0, left_i)):
                X[i] = X[i + 1] + span
        if hit_index[-1] != len(X) - 1:
            right_i = hit_index[-1]
            span = X[right_i] - X[right_i - 1]
            for i in range(right_i + 1, len(X)):
                X[i] = X[i - 1] + span
        return X, hit_index

    def construct_template(self, cplx: AgAbComplex, n_channel=VOCAB.MAX_ATOM_NUMBER, align=True):
        hc, hc_hit = self._chain_template(cplx, self.template_map['H'], n_channel, heavy=True)
        lc, lc_hit = self._chain_template(cplx, self.template_map['L'], n_channel, heavy=False)
        template = np.array(hc + lc)  # [N, n_channel, 3]
        if align:
            # align (will be dropped in the future)
            true_X_bb, temp_X_bb = [], []
            chains = [cplx.get_heavy_chain(), cplx.get_light_chain()]
            temps, hits = [hc, lc], [hc_hit, lc_hit]
            for chain, temp, hit in zip(chains, temps, hits):
                for i, residue_temp in zip(hit, temp):
                    residue = chain.get_residue(i)
                    bb = residue.get_backbone_coord_map()
                    for ai, atom in enumerate(VOCAB.backbone_atoms):
                        if atom not in bb:
                            continue
                        true_X_bb.append(bb[atom])
                        temp_X_bb.append(residue_temp[ai])
            true_X_bb, temp_X_bb = np.array(true_X_bb), np.array(temp_X_bb)
            _, Q, t = kabsch(temp_X_bb, true_X_bb)
            template = np.dot(template, Q) + t
        return template


def parse():
    parser = argparse.ArgumentParser(description='framework template statistics')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--out', type=str, required=True, help='Path to save template (json)')
    parser.add_argument('--conserve_th', type=float, default=0.95, help='Threshold of hit ratio for deciding whether the residue is conserved')
    return parser.parse_args()


def main(args, dataset):
    conserve_pos = {'H': [], 'L': []}
    pos2type = {'H': {}, 'L': {}}
    chain_names = ['H', 'L']
    for cplx in tqdm(dataset.data):
        for chain in chain_names:
            for seg in ['1', '2', '3', '4']:
                fr = cplx.get_framework(chain + 'FR' + seg)
                if fr is None:
                    continue
                for residue in fr:
                    pos, insert_code = residue.get_id()
                    if pos not in pos2type[chain]:
                        pos2type[chain][pos] = {}
                    if insert_code.strip() != '':
                        pos2type[chain][pos]['insert_code'] = 1
                        continue
                    symbol = residue.get_symbol()
                    if symbol not in pos2type[chain][pos]:
                        pos2type[chain][pos][symbol] = 0
                    pos2type[chain][pos][symbol] += 1
    th_num = len(dataset) / 2
    for chain in chain_names:
        for pos in sorted(list(pos2type[chain].keys())):
            if 'insert_code' in pos2type[chain][pos]:
                continue
            # normalize
            total_num = sum(list(pos2type[chain][pos].values()))
            for symbol in pos2type[chain][pos]:
                ratio = pos2type[chain][pos][symbol] / total_num
                if ratio > args.conserve_th and total_num > th_num:  # exclude some rarely seen positions
                    print_log(f'{chain}, {pos}, {pos2type[chain][pos]}')
                    conserve_pos[chain].append(pos)
    print_log(conserve_pos)
    print_log(f'number of conserved residues: {len(conserve_pos["H"]) + len(conserve_pos["L"])}')

    # form vague templates
    templates, masks = [], []
    for cplx in tqdm(dataset.data):
        template, mask = [], []
        for chain in chain_names:
            poses = conserve_pos[chain]
            chain = cplx.get_heavy_chain() if chain == 'H' else cplx.get_light_chain()
            hit_map = {pos: False for pos in poses}
            skip = False
            for residue in chain:
                symbol = residue.get_symbol()
                pos, insert_code = residue.get_id()
                if pos not in hit_map:
                    continue
                # hit
                hit_map[pos] = True
                if insert_code.strip() != '':
                    print_log(f'insert code {insert_code}, pos {pos}')
                    skip = False
                    break
                residue_template = []
                for atom in VOCAB.backbone_atoms:
                    bb = residue.get_backbone_coord_map()
                    if atom not in bb:
                        skip = True
                        break
                    residue_template.append(bb[atom])
                template.append(residue_template)
                if skip:
                    break
            if skip:
                break
            mask.extend([hit_map[pos] for pos in poses])
        if skip:
            continue
        mask = np.array(mask)
        full_template = []
        i = 0
        for m in mask:
            if m:
                full_template.append(template[i])
                i += 1
            else:
                full_template.append([[0, 0, 0] for _ in range(4)])
        template = np.array(full_template)
        templates.append(template)
        masks.append(mask)

    # align
    # find the most complete one
    max_hit, max_hit_idx = 0, -1
    for i, mask in enumerate(masks):
        hit_cnt = mask.sum()
        if hit_cnt > max_hit:
            max_hit, max_hit_idx = hit_cnt, i
    ref, ref_mask = templates[max_hit_idx], masks[max_hit_idx]
    print_log(f'max hit number: {max_hit}')
    for i, template in enumerate(templates):
        align_mask = np.logical_and(masks[i], ref_mask)
        ref_temp = ref[align_mask]
        cur_temp = template[align_mask]
        _, Q, t = kabsch(cur_temp.reshape(-1, 3), ref_temp.reshape(-1, 3))
        # aligned_template = np.dot(template - template.reshape(-1, 3).mean(axis=0), Q) + t
        aligned_template = np.dot(template, Q) + t
        templates[i] = aligned_template
        
    final_template = np.sum(templates, axis=0) / np.sum(masks, axis=0).reshape(-1, 1, 1)
    rmsds = []
    for template, mask in zip(templates, masks):
        template, ref = template[mask], final_template[mask]
        rmsds.append(compute_rmsd(template.reshape(-1, 3), ref.reshape(-1, 3), aligned=False))
    print_log(f'rmsd: max {max(rmsds)}, min {min(rmsds)}, mean {np.mean(rmsds)}')

    # save templates
    template_json = { 'H': {}, 'L': {} }
    i = 0
    for chain in chain_names:
        for pos in conserve_pos[chain]:
            template_json[chain][pos] = final_template[i].tolist()
            i += 1
    assert i == len(final_template)
    with open(args.out, 'w') as fout:
        json.dump(template_json, fout)


if __name__ == '__main__':
    from data.dataset import E2EDataset
    args = parse()
    dataset = E2EDataset(args.dataset)
    main(args, dataset)