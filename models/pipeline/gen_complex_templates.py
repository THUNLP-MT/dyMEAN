#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import json
import argparse
from time import sleep
from shutil import rmtree
from multiprocessing import Process

from tqdm import tqdm
import numpy as np

from configs import CACHE_DIR
from utils.logger import print_log
from data.pdb_utils import VOCAB, AgAbComplex, Protein, Peptide

from .igfold_api import pred
from .hdock_api import dock
from .data_formatter import get_formatter


def dock_wrapper(origin_pdb, out_pdb, cdrh3_pos, *args, **kwargs):
    # generate lsite info
    ab_pdb = args[1]
    hc = Protein.from_pdb(ab_pdb).get_chain('H')
    start, end = cdrh3_pos
    lsite = []
    for i in range(start, end + 1):
        lsite.append(('H', hc.get_residue(i).get_id()[0]))
    kwargs['binding_lsite'] = lsite
    kwargs['sample'] = 100
    results = dock(*args, **kwargs)
    # check epitope
    chains = Protein.from_pdb(results[0]).get_chain_names()
    chains = [c for c in chains if c != 'H' and c != 'L']
    gt_epitope = { f'{chain_name} {res_id}': True for chain_name, res_id in kwargs['binding_rsite']}
    best_template, best_recall = None, -1
    for template_pdb in results:
        recall = 0
        cplx = AgAbComplex.from_pdb(template_pdb, 'H', 'L', chains, skip_epitope_cal=True, skip_validity_check=True)
        epitope = cplx.get_epitope(cdrh3_pos)
        for residue, chain_name, _ in epitope:
            _id = f'{chain_name} {residue.get_id()[0]}'
            if _id in gt_epitope:
                recall += 1
        if recall > best_recall:
            best_template, best_recall = template_pdb, recall
    # template_pdb = results[0]
    template_pdb = best_template
    print_log(f'best template {best_template}, epitope recall {best_recall / len(gt_epitope)}')
    clean(origin_pdb, template_pdb, out_pdb)


def clean(origin_pdb, template_pdb, out_pdb):
    origin_cplx = Protein.from_pdb(origin_pdb)
    template_cplx = Protein.from_pdb(template_pdb)
    peptides = {}

    ori_chain_to_id, id_to_temp_chain = {}, {}
    for chain_name, chain in origin_cplx:
        ori_chain_to_id[chain_name] = f'{chain.get_seq()[:5]}'
    for chain_name, chain in template_cplx:
        id_to_temp_chain[f'{chain.get_seq()[:5]}'] = chain_name
    
    for chain_name in origin_cplx.get_chain_names():
        ori_chain = origin_cplx.get_chain(chain_name)
        temp_chain = template_cplx.get_chain(id_to_temp_chain[ori_chain_to_id[chain_name]])
        for i, residue in enumerate(ori_chain):
            if i < len(temp_chain):
                # renumber
                temp_chain.residues[i].id = residue.id
                # delete Hs
                for atom in temp_chain.residues[i].coordinate:
                    if atom[0] == 'H':
                        del temp_chain.residues[i].coordinate[atom]
            else:
                print_log(f'{origin_cplx.get_id()}, chain {chain_name} lost residues {len(ori_chain)} > {len(temp_chain)}')
                break
        temp_chain.set_id(chain_name)
        peptides[chain_name] = temp_chain
    renumber_cplx = Protein(template_cplx.get_id(), peptides)
    renumber_cplx.to_pdb(out_pdb)


def main(args):
    tmp_dir = os.path.join(CACHE_DIR, f'{args.cdr_model}_pipeline')
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    out_dir = args.out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with open(args.dataset_json, 'r') as fin:
        lines = fin.read().strip().split('\n')
    data = {
        'pdb': [],
        'heavy_chain': [],
        'light_chain': [],
        'antigen_chains': [],
        'pdb_data_path': []
    }

    hdock_p = []

    for line in tqdm(lines):
        item = json.loads(line)
        # load informations
        pdb = item['pdb']
        heavy_chain = item['heavy_chain_seq']
        light_chain = item['light_chain_seq']
        out_pdb = os.path.join(out_dir, f'{pdb}_template.pdb')
        for key in ['pdb', 'heavy_chain', 'light_chain', 'antigen_chains']:
            data[key].append(item[key])
        data['pdb_data_path'].append(out_pdb)
        if os.path.exists(out_pdb):
            print_log(f'{out_pdb} exists. Skip.', level='WARN')
            continue
        # perturb CDR
        if args.cdr_type != '-':  # no cdr to generate, i.e. structure prediction
            cdr_start, cdr_end = item[f'cdr{args.cdr_type.lower()}_pos']
            pert_cdr = np.random.randint(low=0, high=VOCAB.get_num_amino_acid_type(), size=(cdr_end - cdr_start + 1,))
            pert_cdr = ''.join([VOCAB.idx_to_symbol(int(i)) for i in pert_cdr])
            if args.cdr_type[0] == 'H':
                l, r = heavy_chain[:cdr_start], heavy_chain[cdr_end + 1:]
                heavy_chain = l + pert_cdr + r
            else:
                l, r = light_chain[:cdr_start], heavy_chain[cdr_end + 1:]
                light_chain = l + pert_cdr + r
        # save antigen and epitope information
        ag_pdb = os.path.join(tmp_dir, f'{pdb}_ag.pdb')
        cplx = AgAbComplex.from_pdb(item['pdb_data_path'], item['heavy_chain'], item['light_chain'], item['antigen_chains'])
        epitope = cplx.get_epitope()
        # # full antigen
        cplx.get_antigen().to_pdb(ag_pdb)
        # epitope_prot = Protein(cplx.get_id(), { 'A': Peptide('A', [res[0] for res in epitope]) })
        # epitope_prot.to_pdb(ag_pdb)
        origin_pdb = os.path.join(out_dir, f'{pdb}_original.pdb')
        cplx.to_pdb(origin_pdb)

        # 1.IgFold
        ab_pdb = os.path.join(tmp_dir, f'{pdb}.pdb')
        pred(heavy_chain, light_chain, ab_pdb, do_refine=False)

        # 2.HDock
        # check queue
        while len(hdock_p) >= args.num_workers:
            removed = False
            for p in hdock_p:
                if not p.is_alive():
                    p.join()
                    p.close()
                    hdock_p.remove(p)
                    removed = True
            if not removed:
                sleep(10)

        out_folder = os.path.join(tmp_dir, pdb)
        binding_rsite = [(chain_name, residue.get_id()[0]) for residue, chain_name, _ in epitope]
        p = Process(target=dock_wrapper,
                    args=(origin_pdb, out_pdb, item['cdrh3_pos'], ag_pdb, ab_pdb),
                    kwargs={
                        'out_folder': out_folder,
                        'sample': 1,
                        'binding_rsite': binding_rsite
                    })
        p.start()
        hdock_p.append(p)

    while len(hdock_p):
        p = hdock_p[0]
        p.join()
        p.close()
        hdock_p = hdock_p[1:]

    rmtree(tmp_dir)
    print_log(f'template complex pdbs located in {out_dir}')

    # form dataset
    data_out_dir = out_dir if args.data_out_dir is None else args.data_out_dir
    if not os.path.exists(data_out_dir):
        os.makedirs(data_out_dir)
    formatter = get_formatter(args.cdr_model)
    summary = formatter(out_dir=data_out_dir, **data)
    print_log(f'formatted data saved to {summary}')


def parse():
    parser = argparse.ArgumentParser(description='inference of pipeline method')
    parser.add_argument('--cdr_model', type=str, required=True, help='Type of model that generates CDRs',
                        choices=['MEAN', 'Rosetta', 'DiffAb'])
    parser.add_argument('--dataset_json', type=str, required=True, help='Path to the summary file of dataset in json format')
    parser.add_argument('--out_dir', type=str, required=True, help='Path to save generated PDBs')
    parser.add_argument('--data_out_dir', type=str, default=None, help='Directory to save formatted data')
    parser.add_argument('--num_workers', type=str, default=4, help='Number of cores to use for speeding up')
    parser.add_argument('--cdr_type', type=str, default='H3', help='Type of cdr to generate',
                        choices=['H3', '-'])
    return parser.parse_args()


if __name__ == '__main__':
    main(parse())