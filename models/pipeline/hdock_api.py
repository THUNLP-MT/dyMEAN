#!/usr/bin/python
# -*- coding:utf-8 -*-
import os

from configs import HDOCK_DIR, CACHE_DIR
from utils.time_sign import get_time_sign


HDOCK = os.path.join(HDOCK_DIR, 'hdock')
CREATEPL = os.path.join(HDOCK_DIR, 'createpl')

TMP_DIR = os.path.join(CACHE_DIR, 'hdock')
if not os.path.exists(TMP_DIR):
    os.makedirs(TMP_DIR)


def dock(pdb1, pdb2, out_folder, sample=1, binding_rsite=None, binding_lsite=None):
    # working directory
    out_folder = os.path.abspath(out_folder)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    # pdb1 is the receptor (antigen), pdb2 is the ligand (antibody)
    unique_id = get_time_sign()
    tmp_file_list = []
    ori_pdb1, ori_pdb2 = pdb1, pdb2
    pdb1 = f'{unique_id}_{os.path.split(pdb1)[-1]}'
    pdb2 = f'{unique_id}_{os.path.split(pdb2)[-1]}'
    os.system(f'cd {out_folder}; cp {ori_pdb1} {pdb1}')
    os.system(f'cd {out_folder}; cp {ori_pdb2} {pdb2}')
    tmp_file_list.append(pdb1)
    tmp_file_list.append(pdb2)

    # binding site
    arg_rsite, rsite_name = '', f'{unique_id}_rsite.txt'
    if binding_rsite is not None:
        rsite = os.path.join(out_folder, rsite_name)
        with open(rsite, 'w') as fout:
            sites = []
            for chain_name, residue_id in binding_rsite:
                sites.append(f'{residue_id}:{chain_name}')
            fout.write(', '.join(sites))
        arg_rsite = f'-rsite {rsite_name}'
        tmp_file_list.append(rsite_name)

    arg_lsite, lsite_name = '', f'{unique_id}_lsite.txt'
    if binding_lsite is not None:
        lsite = os.path.join(out_folder, lsite_name)
        with open(lsite, 'w') as fout:
            sites = []
            for chain_name, residue_id in binding_lsite:
                sites.append(f'{residue_id}:{chain_name}')
            fout.write(', '.join(sites))
        arg_lsite = f'-lsite {lsite_name}'
        tmp_file_list.append(lsite_name)

    # dock
    dock_out = f'{unique_id}_Hdock.out'
    tmp_file_list.append(dock_out)
    p = os.popen(f'cd {out_folder}; {HDOCK} {pdb1} {pdb2} {arg_rsite} {arg_lsite} -out {dock_out}')
    p.read()
    p.close()

    p = os.popen(f'cd {out_folder}; {CREATEPL} {dock_out} top{sample}.pdb -nmax {sample} -complex -models') 
    p.read()
    p.close()

    # # create complex
    # for condition in [arg_rsite + ' ' + arg_lsite, arg_rsite, '']:
    #     p = os.popen(f'cd {out_folder}; {CREATEPL} {dock_out} top{sample}.pdb -nmax {sample} {condition} -complex -models') 
    #     p.read()
    #     p.close()
    #     if os.path.exists(os.path.join(out_folder, 'model_1.pdb')):
    #         break

    for f in tmp_file_list:
        os.remove(os.path.join(out_folder, f))

    results = [os.path.join(out_folder, f'model_{i + 1}.pdb') for i in range(sample)]
    return results