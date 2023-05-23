#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import re
from copy import deepcopy

from data.pdb_utils import AgAbComplex, Protein, Peptide, Complex
from utils.time_sign import get_time_sign
from utils.logger import print_log
from configs import DOCKQ_DIR, CACHE_DIR


def dockq(mod_cplx: AgAbComplex, ref_cplx: AgAbComplex, cdrh3_only=False):
    H, L = ref_cplx.heavy_chain, ref_cplx.light_chain
    prefix = get_time_sign(suffix=ref_cplx.get_id().replace('(', '').replace(')', ''))
    mod_pdb = os.path.join(CACHE_DIR, prefix + '_dockq_mod.pdb')
    ref_pdb = os.path.join(CACHE_DIR, prefix + '_dockq_ref.pdb')
    if cdrh3_only:
        mod_cdr, ref_cdr = mod_cplx.get_cdr(), ref_cplx.get_cdr()
        mod_peptides, ref_peptides = mod_cplx.antigen.peptides, ref_cplx.antigen.peptides
        mod_peptides[H], ref_peptides[H] = mod_cdr, ref_cdr  # union cdr and antigen chains
        pdb = mod_cplx.get_id()
        mod_cplx, ref_cplx = Protein(pdb, mod_peptides), Protein(pdb, ref_peptides)
        mod_cplx.to_pdb(mod_pdb)
        ref_cplx.to_pdb(ref_pdb)
        p = os.popen(f'{os.path.join(DOCKQ_DIR, "DockQ.py")} {mod_pdb} {ref_pdb} -native_chain1 {H} -no_needle')
    else:
        mod_cplx.to_pdb(mod_pdb)
        ref_cplx.to_pdb(ref_pdb)
        p = os.popen(f'{os.path.join(DOCKQ_DIR, "DockQ.py")} {mod_pdb} {ref_pdb} -native_chain1 {H} {L} -no_needle')
    text = p.read()
    p.close()
    res = re.search(r'DockQ\s+([0-1]\.[0-9]+)', text)
    score = float(res.group(1))
    os.remove(mod_pdb)
    os.remove(ref_pdb)
    return score


def general_dockq(mod_cplx: Complex, ref_cplx: Complex):
    prefix = get_time_sign(suffix=ref_cplx.get_id().replace('(', '').replace(')', ''))
    mod_pdb = os.path.join(CACHE_DIR, prefix + '_dockq_mod.pdb')
    ref_pdb = os.path.join(CACHE_DIR, prefix + '_dockq_ref.pdb')
    mod_cplx_reorder = Protein(mod_cplx.get_id(), {c: mod_cplx.get_chain(c) for c in ref_cplx.get_chain_names()})
    mod_cplx_reorder.to_pdb(mod_pdb)
    ref_cplx.to_pdb(ref_pdb)
    ligand_native_chain = ' '.join(ref_cplx.ligand_chains)
    p = os.popen(f'{os.path.join(DOCKQ_DIR, "DockQ.py")} {mod_pdb} {ref_pdb} -native_chain1 {ligand_native_chain} -no_needle')
    text = p.read()
    p.close()
    res = re.search(r'DockQ\s+([0-1]\.[0-9]+)', text)
    score = float(res.group(1))
    os.remove(mod_pdb)
    os.remove(ref_pdb)
    return score

