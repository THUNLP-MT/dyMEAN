#!/usr/bin/python
# -*- coding:utf-8 -*-
from anarci import run_anarci
from Bio.PDB import PDBParser, PDBIO
from Bio.SeqUtils import seq1


def renumber_seq(seq, scheme='imgt'):
    _, numbering, details, _ = run_anarci([('A', seq)], scheme=scheme, allowed_species=['mouse', 'human'])
    numbering = numbering[0]
    fv, position = [], []
    if not numbering:  # not antibody
        return None
    chain_type = details[0][0]['chain_type']
    numbering = numbering[0][0]
    for pos, res in numbering:
        if res == '-':
            continue
        fv.append(res)
        position.append(pos)
    return ''.join(fv), position, chain_type


def renumber_pdb(pdb, out_pdb, scheme='imgt', mute=False):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('anonym', pdb)
    for chain in structure.get_chains():
        seq = []
        for residue in chain:
            hetero_flag, _, _ = residue.get_id()
            if hetero_flag != ' ':
                continue
            seq.append(seq1(residue.get_resname()))
        seq = ''.join(seq)
        res = renumber_seq(seq, scheme)
        if res is None:
            continue
        fv, position, chain_type = res
        if not mute:
            print(f'chain {chain.id} type: {chain_type}')
        start = seq.index(fv)
        end = start + len(fv)
        assert start != -1, 'fv not found'
        seq_index, pos_index = -1, 0
        for r in list(chain.get_residues()):
            hetero_flag, _, _ = r.get_id()
            if hetero_flag != ' ':
                continue
            seq_index += 1
            if seq_index < start or seq_index >= end:
                chain.__delitem__(r.get_id())
                continue
            assert fv[pos_index] == seq1(r.get_resname()), f'Inconsistent residue in Fv {fv[pos_index]} at {r._id}'
            r._id = (' ', *position[pos_index])
            pos_index += 1
    io = PDBIO()
    io.set_structure(structure)
    io.save(out_pdb)


if __name__ == '__main__':
    import sys
    infile, outfile, scheme = sys.argv[1:4]
    if len(sys.argv) > 4:
        mute = bool(sys.argv[4])
    else:
        mute = False
    renumber_pdb(infile, outfile, scheme, mute)