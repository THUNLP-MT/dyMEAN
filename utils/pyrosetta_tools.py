#!/usr/bin/python
# -*- coding:utf-8 -*-
import pyrosetta
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
from pyrosetta.rosetta.protocols.minimization_packing import PackRotamersMover
from pyrosetta.rosetta.protocols.relax import FastRelax
pyrosetta.init(' '.join([
    '-mute', 'all',
    '-use_input_sc',
    '-ignore_unrecognized_res',
    '-ignore_zero_occupancy', 'false',
    '-load_PDB_components', 'false',
    '-relax:default_repeats', '2',
    '-no_fconfig',
]))


def get_scorefxn(scorefxn_name:str):
    """
    Gets the scorefxn with appropriate corrections.
    Taken from: https://gist.github.com/matteoferla/b33585f3aeab58b8424581279e032550
    """
    import pyrosetta

    corrections = {
        'beta_july15': False,
        'beta_nov16': False,
        'gen_potential': False,
        'restore_talaris_behavior': False,
    }
    if 'beta_july15' in scorefxn_name or 'beta_nov15' in scorefxn_name:
        # beta_july15 is ref2015
        corrections['beta_july15'] = True
    elif 'beta_nov16' in scorefxn_name:
        corrections['beta_nov16'] = True
    elif 'genpot' in scorefxn_name:
        corrections['gen_potential'] = True
        pyrosetta.rosetta.basic.options.set_boolean_option('corrections:beta_july15', True)
    elif 'talaris' in scorefxn_name:  #2013 and 2014
        corrections['restore_talaris_behavior'] = True
    else:
        pass
    for corr, value in corrections.items():
        pyrosetta.rosetta.basic.options.set_boolean_option(f'corrections:{corr}', value)
    return pyrosetta.create_score_function(scorefxn_name)


def get_chains_from_pose(pose):
    n_chains = pose.num_chains()
    pdb_info = pose.pdb_info()
    chains = {}
    for i in range(n_chains):
        res_begin = pose.chain_begin(i + 1)
        res_end = pose.chain_end(i + 1)
        chains[pdb_info.chain(res_begin)] = (res_begin, res_end)
    return chains


def interface_energy(pdb_path, fixed_chains, normalize=False, return_dict=False):
    '''
    param fixed_chains: list of chain defining ligand or receptor,
        e.g. ['A', 'B'] or ['C', 'D'] where chain A and B compose ligand and C and D compose receptor
        antibody can be seen as ligand and antigen as receptor
    '''
    pose = pyrosetta.pose_from_pdb(pdb_path)
    chains = get_chains_from_pose(pose)
    other_chains = [c for c in chains if c not in fixed_chains]
    interface = ''.join(fixed_chains) + '_' + ''.join(other_chains)
    mover = InterfaceAnalyzerMover(interface)
    mover.set_pack_separated(True)
    mover.set_pack_input(True)
    mover.apply(pose)
    if return_dict:
        return pose.scores
    if normalize:
        return pose.scores['dG_separated/dSASAx100']
    return pose.scores['dG_separated']


def sidechain_packing(pdb_path, out_path=None, scorefxn='ref2015'):
    '''
    side-chain packing with fixed residue type and backbone
    '''
    pose = pyrosetta.pose_from_pdb(pdb_path)
    packer_task = pyrosetta.standard_packer_task(pose)
    packer_task.restrict_to_repacking()
    scorefxn = get_scorefxn(scorefxn)
    pack_mover = PackRotamersMover(scorefxn, packer_task)
    pack_mover.apply(pose)
    if out_path is None:
        out_path = pdb_path
    pose.dump_pdb(out_path)
    return out_path


def fast_relax(pdb_path, out_path=None, max_iter=1000, scorefxn='ref2015', flexible_map=None):
    '''
    fast relax (minimize energy)
    :param flexible_map: a map defining the residues that can be relaxed, e.g. { 'A': [5, 6, 8], ... }, you can set the index to "ALL" for selecting all residues
    '''
    pose = pyrosetta.pose_from_pdb(pdb_path)
    relax_task = FastRelax()
    relax_task.set_scorefxn(get_scorefxn(scorefxn))
    relax_task.set_enable_design(False)
    relax_task.max_iter(max_iter)
    if flexible_map is not None:
        movemap = pyrosetta.MoveMap()
        chains = get_chains_from_pose(pose)
        flexible = pyrosetta.rosetta.utility.vector1_bool(len(pose.residues))
        for chain in flexible_map:
            begin, end = chains[chain]
            if flexible_map[chain] == 'ALL':
                for idx in range(begin, end + 1):
                    flexible[idx] = True
            else:
                for idx in flexible_map[chain]:
                    flexible[begin + idx] = True
        movemap.set_bb(allow_bb=flexible)
        movemap.set_chi(allow_chi=flexible)
        relax_task.set_movemap(movemap)
    relax_task.apply(pose)
    if out_path is None:
        out_path = pdb_path
    pose.dump_pdb(out_path)
    return out_path