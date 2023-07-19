#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import re
import json
import requests
from argparse import ArgumentParser
from collections import defaultdict
from functools import partial
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map, process_map

from configs import AG_TYPES, RAbD_PDB, IGFOLD_TEST_PDB, SKEMPI_PDB
from configs import IMGT
from .pdb_utils import Protein, AgAbComplex, Peptide
from utils.io import read_csv
from utils.network import url_get
from utils.logger import print_log


def fetch_from_pdb(identifier, tries=5):
    # example identifier: 1FBI

    identifier = identifier.upper()
    url = 'https://data.rcsb.org/rest/v1/core/entry/' + identifier

    res = url_get(url, tries)
    if res is None:
        return None

    url = f'https://files.rcsb.org/download/{identifier}.pdb'

    text = url_get(url, tries)
    if text is None:
        return None

    data = res.json()
    data['pdb'] = text.text
    return data


'''SAbDab Summary reader'''
def read_sabdab(fpath, n_cpu):
    heads, entries = read_csv(fpath, sep='\t')
    head2idx = { head: i for i, head in enumerate(heads) }
    items, pdb2idx = [], {}
    head_mapping = {
        'heavy_chain': head2idx['Hchain'],
        'light_chain': head2idx['Lchain'],
        'antigen_chains': head2idx['antigen_chain']
    }
    for entry in entries:
        pdb = entry[head2idx['pdb']]
        if pdb in pdb2idx:  # (redundant)
            continue

        # extract useful summary
        item = { 'pdb': pdb }
        for head in head_mapping:
            item[head] = entry[head_mapping[head]]
            if item[head] == 'nan':
                item[head] = ''
    
        # antigen
        if len(item['antigen_chains']):
            ag_types = entry[head2idx['antigen_type']].strip().split(' | ')
            ag_chains = item['antigen_chains'].strip().split(' | ')
            assert len(ag_types) == len(ag_chains)
            cleaned_chains = []
            for _type, _c in zip(ag_types, ag_chains):
                if _type not in AG_TYPES:
                    continue
                assert _c != item['heavy_chain']
                assert _c != item['light_chain']
                cleaned_chains.append(_c)
            item['antigen_chains'] = cleaned_chains
        else:
            item['antigen_chains'] = []

        if pdb in pdb2idx:
            items[pdb2idx[pdb]] = item
        else:
            pdb2idx[pdb] = len(items)
            items.append(item)
    # example of an item:
    # {
    #   'pdb': xxxx,
    #   'heavy_chain': A,
    #   'light_chain': B,
    #   'antigen_chains': [N, L]
    # }
    return items


'''RABD reader'''
def read_rabd(fpath):
    with open(fpath, 'r') as fin:
        lines = fin.readlines()
    items = [json.loads(s.strip('\n')) for s in lines]
    keys = ['pdb', 'heavy_chain', 'light_chain', 'antigen_chains']
    rabd_pdbs = { pdb: True for pdb in RAbD_PDB }
    rabd_items = []
    for item in items:
        for key in keys:
            assert key in item, f'{item} do not have {key}'
        pdb = item['pdb']
        if pdb not in rabd_pdbs:
            continue
        rabd_items.append(item)
    return rabd_items


'''IgFold testset reader'''
def read_igfold_testset(fpath):
    with open(fpath, 'r') as fin:
        lines = fin.readlines()
    items = [json.loads(s.strip('\n')) for s in lines]
    keys = ['pdb', 'heavy_chain', 'light_chain', 'antigen_chains']
    igfold_pdbs = { pdb: True for pdb in IGFOLD_TEST_PDB }
    igfold_items = []
    for item in items:
        for key in keys:
            assert key in item, f'{item} do not have {key}'
        pdb = item['pdb']
        if pdb not in igfold_pdbs:
            continue
        igfold_items.append(item)
    return igfold_items


'''SKEMPI reader'''
def read_skempi(fpath):
    with open(fpath, 'r') as fin:
        lines = fin.readlines()
    items = [json.loads(s.strip('\n')) for s in lines]
    keys = ['pdb', 'heavy_chain', 'light_chain', 'antigen_chains']
    skempi_pdbs = { pdb: True for pdb in SKEMPI_PDB }
    skempi_items = []
    for item in items:
        for key in keys:
            assert key in item, f'{item} do not have {key}'
        pdb = item['pdb']
        if pdb not in skempi_pdbs:
            continue
        skempi_items.append(item)
    return skempi_items


def download_one_item(item):
    pdb_id = item['pdb']
    try:
        pdb_data = fetch_from_pdb(pdb_id)
    except:
        pdb_data = None
    if pdb_data is None:
        print(f'{pdb_id} invalid')
        item = None
    else:
        item['pdb_data'] = pdb_data['pdb']
    return item


def download_one_item_local(pdb_dir, item):
    pdb_id = item['pdb']
    if pdb_id in ['3h3b', '2ghw', '3uzq']:
        # these complexes are in RAbD, but their pdb data in SAbDab is wrongly annotated
        file_dir = os.path.split(__file__)[0]
        with open(os.path.join(file_dir, '..', 'summaries', pdb_id + '.pdb'), 'r') as fin:
            item['pdb_data'] = fin.read()
        if pdb_id == '3h3b':
            item['heavy_chain'] = 'D'
            item['light_chain'] = 'E'
            item['antigen_chains'] = ['B']
        elif pdb_id == '2ghw':
            item['heavy_chain'] = 'B'
            item['light_chain'] = 'E'
            item['antigen_chains'] = ['A']
        elif pdb_id == '3uzq':
            item['heavy_chain'] = 'A'
            item['light_chain'] = 'C'
            item['antigen_chains'] = ['B']
        return item

    for pdb_id in [pdb_id.lower(), pdb_id.upper()]:
        fname = os.path.join(pdb_dir, pdb_id + '.pdb')
        if not os.path.exists(fname):
            continue
        with open(fname, 'r') as fin:
            item['pdb_data'] = fin.read()
        return item
    print(f'{pdb_id} not found in {pdb_dir}, try fetching from remote server')
    from_remote = fetch_from_pdb(pdb_id)
    if from_remote is not None:
        print('fetched')
        item['pdb_data'] = from_remote['pdb']
        item['pre_numbered'] = False
        return item
    return None


def cleanup_none(pdb_path):
    # os.remove(pdb_path)
    return None


def post_process(item):
    # renumbering pdb file and revise cdrs
    pdb_data_path, numbering, pdb = item['pdb_data_path'], item['numbering'], item['pdb']

    if item['light_chain'] == '' or item['heavy_chain'] == '':  # we do not handle nanobodies
        return cleanup_none(pdb_data_path)

    if numbering == 'none':
        pass
    elif numbering == 'imgt':
        if not item['pre_numbered']:
            exit_code = IMGT.renumber(pdb_data_path, pdb_data_path)
            if exit_code != 0:
                # print_log(f'renumbering failed for {pdb}. scheme {numbering}', level='ERROR')
                return cleanup_none(pdb_data_path)
        try:
            # clean pdb
            if item['heavy_chain'].lower() == item['light_chain'].lower(): # H/L is connected
                ab_chain = item['heavy_chain'].upper()
                prot = Protein.from_pdb(pdb_data_path)
                ab_chain_pep = prot.get_chain(ab_chain)
                light_chain_start = 0
                for i, residue in enumerate(ab_chain_pep):
                    if residue.get_id()[0] > IMGT.HFR4[-1]:
                        light_chain_start = i
                        break
                hc, lc = ab_chain_pep.residues[:light_chain_start], ab_chain_pep.residues[light_chain_start:]
                if ab_chain.lower() == item['heavy_chain']:  # light chain go first
                    hc, lc = lc, hc
                prot.peptides[ab_chain] = Peptide(ab_chain, hc)
                lc_id = 'A'
                while lc_id in prot.peptides:
                    lc_id = chr(ord(lc_id) + 1)
                prot.peptides[lc_id] = Peptide(lc_id, lc)
                prot.to_pdb(pdb_data_path)
                item['heavy_chain'] = ab_chain
                item['light_chain'] = lc_id

                exit_code = IMGT.renumber(pdb_data_path, pdb_data_path)
                if exit_code != 0:
                    print_log(f'renumbering failed for {pdb}. scheme {numbering}', level='ERROR')
                    return cleanup_none(pdb_data_path)

            # To complex
            H, L, A = item['heavy_chain'], item['light_chain'], item['antigen_chains']
            cplx = AgAbComplex.from_pdb(pdb_data_path, H, L, A,
                                        numbering=numbering, skip_epitope_cal=True)
            cplx.to_pdb(pdb_data_path)


        except AssertionError as e:
            print_log(f'parsing pdb failed for {pdb}, reason: {e}', level='INFO')
            return cleanup_none(pdb_data_path)
        except Exception as e:
            print_log(f'parsing pdb failed for {pdb} unexpectedly. reason: {e}', level='WARN')
            return cleanup_none(pdb_data_path)
        item['heavy_chain_seq'] = cplx.get_heavy_chain().get_seq()
        item['light_chain_seq'] = cplx.get_light_chain().get_seq()
        item['antigen_seqs'] = [ chain.get_seq() for _, chain in cplx.get_antigen()]
        for c in ['H', 'L']:
            for i in range(1, 4):
                cdr_name = f'{c}{i}'.lower()
                cdr_pos, cdr = cplx.get_cdr_pos(cdr_name), cplx.get_cdr(cdr_name)
                item[f'cdr{cdr_name}_pos'] = cdr_pos
                item[f'cdr{cdr_name}_seq'] = cdr.get_seq()
    else:
        raise NotImplementedError(f'Numbering scheme {numbering} not supported')
    return item


def download(items, out_path, ncpu=8, pdb_dir=None, numbering='imgt', pre_numbered=False):
    if pdb_dir is None:
        map_func = download_one_item
    else:
        map_func = partial(download_one_item_local, pdb_dir)

    print_log('downloading raw files')
    valid_entries = thread_map(map_func, items, max_workers=ncpu)
    valid_entries = [item for item in valid_entries if item is not None]
    print_log(f'number of downloaded entries: {len(valid_entries)}')
    pdb_out_dir = os.path.join(os.path.split(out_path)[0], 'pdb')
    if os.path.exists(pdb_out_dir):
        print_log(f'pdb file out directory {pdb_out_dir} exists!', level='WARN')
    else:
        os.makedirs(pdb_out_dir)

    print_log(f'writing PDB files to {pdb_out_dir}')
    for item in tqdm(valid_entries):
        pdb_fout = os.path.join(pdb_out_dir, item['pdb'] + '.pdb')
        with open(pdb_fout, 'w') as pfout:
            pfout.write(item['pdb_data'])
        item.pop('pdb_data')
        item['pdb_data_path'] = os.path.abspath(pdb_fout)
        item['numbering'] = numbering
        if 'pre_numbered' not in item:
            item['pre_numbered'] = pre_numbered

    print_log('post processing: data cleaning')
    valid_entries = process_map(post_process, valid_entries, max_workers=ncpu, chunksize=1)
    valid_entries = [item for item in valid_entries if item is not None]

    print_log(f'number of valid entries: {len(valid_entries)}')
    fout = open(out_path, 'w')
    for item in valid_entries:
        item_str = json.dumps(item)
        fout.write(f'{item_str}\n')
    fout.close()
    return valid_entries


def statistics(items, n_eg=5):  # show n_eg instances for each type of data
    keys = ['heavy_chain', 'light_chain', 'antigen_chains']
    cnts, example = defaultdict(int), {}
    for item in items:
        res = ''
        for key in keys:
            res += '1' if len(item[key]) else '0'
        cnts[res] += 1
        if res not in example:
            example[res] = []
        if len(example[res]) < n_eg:
            example[res].append(item['pdb'])
    sorted_desc_keys = sorted(list(cnts.keys()))
    for desc_key in sorted_desc_keys:
        desc = 'Only has '
        for key, val in zip(keys, desc_key):
            if val == '1':
                desc += key + ', '
        desc += str(cnts[desc_key])
        print(f'{desc}, examples: {example[desc_key]}')


def parse():
    parser = ArgumentParser(description='download full pdb data')
    parser.add_argument('--summary', type=str, required=True, help='Path to summary file')
    parser.add_argument('--fout', type=str, required=True, help='Path to output json file')
    parser.add_argument('--type', type=str, choices=['skempi', 'sabdab', 'rabd', 'igfold_test'], default='sabdab',
                        help='Type of the dataset')
    parser.add_argument('--pdb_dir', type=str, default=None, help='Path to local folder of PDB files')
    parser.add_argument('--numbering', type=str, default='imgt', choices=['imgt', 'none'],
                        help='Renumbering scheme')
    parser.add_argument('--pre_numbered', action='store_true', help='The files in pdb_dir is already renumbered')
    parser.add_argument('--n_cpu', type=int, default=8, help='Number of cpu to use')
    return parser.parse_args()


def main(args):
    fpath, out_path = args.summary, args.fout
    print_log(args)
    print_log(f'download {args.type} from summary file {fpath}')

    print_log('Extracting summary to json format')
    if args.type == 'skempi':
        items = read_skempi(fpath)
    elif args.type == 'sabdab':
        items = read_sabdab(fpath, args.n_cpu)
    elif args.type == 'rabd':
        items = read_rabd(fpath)
    elif args.type == 'igfold_test':
        items = read_igfold_testset(fpath)

    print_log('Start downloading pdbs in the summary')
    if args.pdb_dir is not None:
        print_log(f'using local PDB files: {args.pdb_dir}')
        if args.pre_numbered:
            print_log(f'Assume PDB file already renumbered with scheme {args.numbering}')
    items = download(items, out_path, args.n_cpu, args.pdb_dir, args.numbering, args.pre_numbered)

    print_log('Start doing statistics on the dataset')
    statistics(items)


if __name__ == '__main__':
    main(parse())
