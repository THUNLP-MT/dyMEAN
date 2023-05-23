#!/usr/bin/python
# -*- coding:utf-8 -*-
import argparse
import numpy as np

from data.pdb_utils import Protein, VOCAB


def get_interface(pdb, receptor_chains, ligand_chains, num_epitope_residues=48):
    prot = Protein.from_pdb(pdb)
    for c in receptor_chains:
        assert c in prot.peptides, f'Chain {c} not found for receptor'
    for c in ligand_chains:
        assert c in prot.peptides, f'Chain {c} not found for ligand'
    receptor = Protein(prot.get_id(), {c: prot.get_chain(c) for c in receptor_chains})
    ligand = Protein(prot.get_id(), {c: prot.get_chain(c) for c in ligand_chains})

    rec_rids, rec_xs, lig_rids, lig_xs = [], [], [], []    
    rec_mask, lig_mask = [], []
    for _type, protein in zip(['rec', 'lig'], [receptor, ligand]):
        is_rec = _type == 'rec'
        rids = []
        if is_rec: 
            rids, xs, masks = rec_rids, rec_xs, rec_mask
        else:
            rids, xs, masks = lig_rids, lig_xs, lig_mask
        for chain_name, chain in protein:
            for i, residue in enumerate(chain):
                bb_coord = residue.get_backbone_coord_map()
                sc_coord = residue.get_sidechain_coord_map()
                coord = {}
                coord.update(bb_coord)
                coord.update(sc_coord)
                num_pad = VOCAB.MAX_ATOM_NUMBER - len(coord)
                x = [coord[key] for key in coord] + [[0, 0, 0] for _ in range(num_pad)]
                mask = [1 for _ in coord] + [0 for _ in range(num_pad)]
                rids.append((chain_name, i))
                xs.append(x)
                masks.append(mask)
    
    # calculate distance
    rec_xs, lig_xs = np.array(rec_xs), np.array(lig_xs) # [Nrec/lig, M, 3], M == MAX_ATOM_NUM
    rec_mask, lig_mask = np.array(rec_mask).astype('bool'), np.array(lig_mask).astype('bool')  # [Nrec/lig, M]
    dist = np.linalg.norm(rec_xs[:, None] - lig_xs[None, :], axis=-1)  # [Nrec, Nlig, M]
    dist = dist + np.logical_not(rec_mask[:, None] * lig_mask[None, :]) * 1e6  # [Nrec, Nlig, M]
    dist_mat = np.min(dist, axis=-1)  # [Nrec, Nlig]
    min_dists = np.min(dist_mat, axis=-1)  # [rec_len]
    topk = min(len(min_dists), num_epitope_residues)
    ind = np.argpartition(-min_dists, -topk)[-topk:]
    lig_idxs = np.argmin(dist_mat, axis=-1)  # [Nrec]
    epitope, dists = [], []
    for idx in ind:
        # epitope
        chain_name, i = rec_rids[idx]
        residue = receptor.peptides[chain_name].get_residue(i)
        epitope.append((residue, chain_name, i))
        # nearest ligand residue
        chain_name, i = lig_rids[lig_idxs[idx]]
        residue = ligand.peptides[chain_name].get_residue(i)
        dists.append((residue, chain_name, i ,min_dists[idx]))
    return epitope, dists


if __name__ == '__main__':
    import json
    parser = argparse.ArgumentParser(description='get interface')
    parser.add_argument('--pdb', type=str, required=True, help='Path to the complex pdb')
    parser.add_argument('--receptor', type=str, nargs='+', required=True, help='Specify receptor chain ids')
    parser.add_argument('--ligand', type=str, nargs='+', required=True, help='Specify ligand chain ids')
    parser.add_argument('--k', type=int, default=48, help='Maximal of K residues nearest ligand are extracted as epitope')
    parser.add_argument('--out', type=str, default=None, help='Save epitope information to json file if specified')
    args = parser.parse_args()
    epitope, dists = get_interface(args.pdb, args.receptor, args.ligand, args.k)
    para_res = {}
    for _, chain_name, i, d in dists:
        key = f'{chain_name}-{i}'
        para_res[key] = 1
    print(f'REMARK: {len(epitope)} residues in epitope, with {len(para_res)} residues in paratope:')
    print(f' \tchain\tposition\tsymbol\tchain\tposition\tsymbol\tdistance')
    for i, (e, p) in enumerate(zip(epitope, dists)):
        e_res, e_chain_name, _ = e
        p_res, p_chain_name, _, d = p
        print(f'{i+1}\t{e_chain_name}\t{e_res.get_id()}\t{e_res.get_symbol()}\t' + \
              f'{p_chain_name}\t{p_res.get_id()}\t{p_res.get_symbol()}\t{round(d, 3)}')

    if args.out:
        data = []
        for e in epitope:
            res, chain_name, _ = e
            data.append((chain_name, res.get_id()))
        with open(args.out, 'w') as fout:
            json.dump(data, fout)