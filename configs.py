#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
'''
Four parts:
1. basic variables
2. benchmark definitions and configs for data processing
3. definitions for antibody numbering system
4. optional dependencies for pipelines
'''

# 1. basic variables
PROJ_DIR = os.path.split(__file__)[0]
RENUMBER = os.path.join(PROJ_DIR, 'utils', 'renumber.py')
# FoldX
FOLDX_BIN = './foldx5/foldx_20231231'
# DockQ 
# IMPORTANT: change it to your path to DockQ project)
DOCKQ_DIR = './DockQ'
# cache directory
CACHE_DIR = os.path.join(PROJ_DIR, '__cache__')
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)


# 2. configs related to data process
AG_TYPES = ['protein', 'peptide']
RAbD_PDB = ['1a14', '1a2y', '1fe8', '1ic7', '1iqd', '1n8z', '1ncb', '1osp', '1uj3', '1w72', '2adf', '2b2x', '2cmr', '2dd8', '2ghw', '2vxt', '2xqy', '2xwt', '2ypv', '3bn9', '3cx5', '3ffd', '3h3b', '3hi6', '3k2u', '3l95', '3mxw', '3nid', '3o2d', '3rkd', '3s35', '3uzq', '3w9e', '4cmh', '4dtg', '4dvr', '4etq', '4ffv', '4fqj', '4g6j', '4g6m', '4h8w', '4ki5', '4lvn', '4ot1', '4qci', '4xnq', '4ydk', '5b8c', '5bv7', '5d93', '5d96', '5en2', '5f9o', '5ggs', '5hi4', '5j13', '5l6y', '5mes', '5nuz']
IGFOLD_TEST_PDB = ['7rdm', '6xsw', '7e9b', '7cj2', '7o33', '7ora', '7mzh', '7mzj', '7ahu', '7s13', '7mzk', '7e72', '7n3c', '7n3f', '7rdl', '7mzg', '7n4j', '7r9d', '7e3o', '7rah', '7or9', '7oo2', '6xsn', '7arn', '7n3e', '7o30', '7kf0', '7lfa', '7nx3', '7keo', '7mzf', '7o31', '7e5o', '7daa', '7s11', '7aj6', '7m2i', '7kf1', '7jkm', '7n4i', '6xm2', '7doh', '7o2z', '7kba', '7s3m', '7mzm', '7rks', '7n3h', '7lyw', '7rco', '7bg1', '7coe', '7n3g', '7kkz', '7kyo', '7s4s', '7rnj', '7bbg', '7l7e', '7n0u', '6xlz', '7mf7', '6xp6', '7lfb', '7kn3', '7rdk', '7s0b', '7kez', '7n3d', '7o4y']
SKEMPI_PDB = ['1ahw', '1dvf', '1vfb', '2vis', '2vir', '1kiq', '1kip', '1kir', '2jel', '1nca', '1dqj', '1jrh', '1nmb', '3hfm', '1yy9', '4gxu', '3lzf', '1n8z', '3g6d', '1xgu', '1xgp', '1xgq', '1xgr', '1xgt', '3n85', '4i77', '3l5x', '4jpk', '1bj1', '1cz8', '1mhp', '2b2x', '1mlc', '3bdy', '3be1', '2ny7', '3idx', '2nyy', '2nz9', '3ngb', '2bdn', '3w2d', '4krl', '4kro', '4krp', '4nm8', '4u6h', '4zs6', '5c6t', '5dwu', '3se8', '3se9', '1yqv']
CONTACT_DIST = 6.6  # 6.6 A between one pair of atoms means the two residues are interacting


# 3. antibody numbering, [start, end] of residue id, both start & end are included
# 3.1 IMGT numbering definition
class IMGT:
    # heavy chain
    HFR1 = (1, 26)
    HFR2 = (39, 55)
    HFR3 = (66, 104)
    HFR4 = (118, 129)

    H1 = (27, 38)
    H2 = (56, 65)
    H3 = (105, 117)

    # light chain
    LFR1 = (1, 26)
    LFR2 = (39, 55)
    LFR3 = (66, 104)
    LFR4 = (118, 129)

    L1 = (27, 38)
    L2 = (56, 65)
    L3 = (105, 117)

    Hconserve = {
        23: ['CYS'],
        41: ['TRP'],
        104: ['CYS']
    }

    Lconserve = {
        23: ['CYS'],
        41: ['TRP'],
        104: ['CYS']
    }

    @classmethod
    def renumber(cls, pdb, out_pdb):
        code = os.system(f'python {RENUMBER} {pdb} {out_pdb} imgt 0')
        return code

# 3.2 Chothia numbering definition
class Chothia:
    # heavy chain
    HFR1 = (1, 25)
    HFR2 = (33, 51)
    HFR3 = (57, 94)
    HFR4 = (103, 113)

    H1 = (26, 32)
    H2 = (52, 56)
    H3 = (95, 102)

    # light chain
    LFR1 = (1, 23)
    LFR2 = (35, 49)
    LFR3 = (57, 88)
    LFR4 = (98, 107)

    L1 = (24, 34)
    L2 = (50, 56)
    L3 = (89, 97)

    Hconserve = {
        92: ['CYS']
    }

    Lconserve = {
        88: ['CYS']
    }

    @classmethod
    def renumber(cls, pdb, out_pdb):
        code = os.system(f'python {RENUMBER} {pdb} {out_pdb} chothia 0')
        return code


# (Optional) 4. dependencies for pipelines
# 4.1 structure prediction
IGFOLD_DIR = './IgFold'
IGFOLD_CKPTS = None  # 'None' means using the default checkpoints
# 4.2 docking
HDOCK_DIR = './HDOCKlite-v1.1'
# 4.3 CDR generation models
MEAN_DIR = './MEAN'
Rosetta_DIR = './rosetta/rosetta.binary.linux.release-315/main/source/bin'
DiffAb_DIR = './diffab'