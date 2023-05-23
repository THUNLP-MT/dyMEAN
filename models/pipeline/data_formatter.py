#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import json
from tqdm import tqdm

from data.pdb_utils import AgAbComplex, Protein
from utils.logger import print_log
from configs import Rosetta_DIR


def get_formatter(model_type):
    mapping = {
        'MEAN': mean_format_data,
        'Rosetta': rosetta_format_data,
        'DiffAb': mean_format_data  # still apply
    }
    return mapping[model_type]


def mean_format_data(pdb, heavy_chain, light_chain, antigen_chains, pdb_data_path, out_dir):
    out_file = os.path.join(out_dir, 'summary.json')
    if os.path.exists(out_file):
        print_log(f'summary file {out_file} exists. Skip.', level='WARN')
        return out_file

    data = []
    for i in range(len(pdb)):
        data.append({
            'pdb': pdb[i],
            'heavy_chain': heavy_chain[i],
            'light_chain': light_chain[i],
            'antigen_chains': antigen_chains[i],
            'pdb_data_path': pdb_data_path[i]
        })
    with open(out_file, 'w') as fout:
        fout.writelines(list(map(lambda item: json.dumps(item) + '\n', data)))
    return out_file


def rosetta_format_data(pdb, heavy_chain, light_chain, antigen_chains, pdb_data_path, out_dir):
    out_file = os.path.join(out_dir, 'summary.json')
    if os.path.exists(out_file):
        print_log(f'summary file {out_file} exists. Skip.', level='WARN')
        return out_file
    # renumbering with aho
    data = []
    converter = os.path.join(Rosetta_DIR, 'antibody_numbering_converter.static.linuxgccrelease') 
    pdb_list = pdb
    for i, pdb in enumerate(tqdm(pdb_data_path)):
        filename = os.path.split(pdb)[-1]
        tmp_pdb = filename[:-4] + '_0001.pdb'
        # change to H/L
        H, L, A = heavy_chain[i], light_chain[i], antigen_chains[i]
        prot = Protein.from_pdb(pdb)
        mapping, peptides = {}, {}
        for ci, (chain_name, chain) in enumerate(prot):
            if chain_name == H:
                new_id = 'H'
            elif chain_name == L:
                new_id = 'L'
            else:
                new_id = chr(65 + ci) # suppose not large enough to H
            mapping[new_id] = chain_name
            chain.set_id(new_id)
            peptides[new_id] = chain
        new_prot = Protein(prot.get_id(), peptides)
        out_pdb = os.path.join(out_dir, filename)
        new_prot.to_pdb(out_pdb)

        p = os.popen(f'cd {out_dir}; {converter} -s {filename} -input_ab_scheme IMGT -output_ab_scheme AHO -overwrite; mv {tmp_pdb} {filename}')
        p.read()
        p.close()

        data.append({
            'pdb': pdb_list[i],
            'heavy_chain': heavy_chain[i],
            'light_chain': light_chain[i],
            'antigen_chains': antigen_chains[i],
            'pdb_data_path': pdb_data_path[i],
            'renumbered': out_pdb,
            'mapping': mapping
        })
    with open(out_file, 'w') as fout:
        fout.writelines(list(map(lambda item: json.dumps(item) + '\n', data)))
    return out_file