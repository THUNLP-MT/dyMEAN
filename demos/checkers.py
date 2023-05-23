#!/usr/bin/python
# -*- coding:utf-8 -*-
from data import AgAbComplex
from data.pdb_utils import dist_matrix_from_residues


'''
checkers before relax, with parameters:
:param ori_cdr: str, original CDR sequence. Different CDRs are connected by spaces
:param gen_pdb: str, path to the generated pdb.
:param gen_cdr: str, generated CDR sequence. Different CDRs are connected by spaces
return:
1. bool. Whether the candidate passed the check.
2. str. Reason of failure (Optional, empty string is acceptable).
'''

def novel_checker(ori_cdr, gen_pdb, gen_cdr):
    return ori_cdr != gen_cdr, 'same as original CDR'


def conserve_checker(ori_cdr, gen_pdb, gen_cdr):
    '''this checker examine whether at least 4aa remain the same at two ends of each CDR'''
    pass_check = True
    for seq1, seq2 in zip(ori_cdr.split(), gen_cdr.split()):
        forward_same_cnt = 0
        for a, b in zip(seq1, seq2):
            if a == b:
                forward_same_cnt += 1
            else:
                break
        
        backward_same_cnt = 0
        for a, b in zip(seq1[::-1], seq2[::-1]):
            if a == b:
                backward_same_cnt += 1
            else:
                break
        
        if forward_same_cnt + backward_same_cnt < 4:
            pass_check = False
            break

    return pass_check, ''


# RELAX_CHECKERS = [novel_checker, conserve_checker]
RELAX_CHECKERS = [novel_checker]


'''
checkers before evaluation, with parameters:
:param summary: ComplexSummary, summary of the wild type (wt). Record the basic information including path to the pdb of wt, the chain id of heavy chain / light chain / antigen
:param gen_pdb: str, path to the generated pdb.
return:
1. bool. Whether the candidate passed the check.
2. str. Reason of failure (Optional, empty string is acceptable).
'''
def aromatic_cage_checker(summary, gen_pdb):
    cplx = AgAbComplex.from_pdb(gen_pdb, summary.heavy_chain, summary.light_chain, summary.antigen_chains)
    hc = cplx.get_heavy_chain()
    lc = cplx.get_light_chain()
    h_trp_0, h_trp_1 = hc.get_residue(32), hc.get_residue(105)
    l_trp_0 = lc.get_residue(86)
    if not l_trp_0.get_symbol() == 'W':
        return False, 'Light chain TRP changed!'
    m3l = cplx.antigen.get_chain('A').get_residue(4)
    dist = dist_matrix_from_residues([m3l], [h_trp_0, h_trp_1, l_trp_0])
    for d in dist.squeeze():
        if d > 5:
            return False, f'Aromatic cages corrupted: {dist}'
    return True, ''



PRE_EVAL_CHECKERS = [aromatic_cage_checker]

'''
checkers after evaluation, with parameters:
:param summary: ComplexSummary, summary of the wild type (wt). Record the basic information including path to the pdb of wt, the chain id of heavy chain / light chain / antigen
:param gen_pdb: str, path to the generated pdb.
:param base_metric: list, evaluation metrics of the wild type.
:param cand_metric: list, evaluation metrics of the candidate.
return:
1. bool. Whether the candidate passed the check.
2. str. Reason of failure (Optional, empty string is acceptable).
'''



def better_checker(summary, gen_pdb, base_metric, cand_metric):
    '''This checker checks whether all metrics of the candidate are better than those of the wild type.'''
    for val1, val2 in zip(base_metric, cand_metric):
        if val1 < val2:
            return False, f'{[round(val, 3) for val in cand_metric]} not better than base {[round(val, 3) for val in base_metric]}'
    return True, ''


EVAL_CHECKERS = [better_checker]
