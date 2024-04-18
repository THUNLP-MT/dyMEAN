#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
from os.path import splitext, basename
from copy import deepcopy

import numpy as np
os.environ['OPENMM_CPU_THREADS'] = '4'  # prevent openmm from using all cpus available
from openmm import LangevinIntegrator, Platform, CustomExternalForce
from openmm.app import PDBFile, Simulation, ForceField, HBonds, Modeller
from simtk.unit import kilocalories_per_mole, angstroms
from pdbfixer import PDBFixer

import logging
logging.getLogger('openmm').setLevel(logging.ERROR)

from data.pdb_utils import Peptide
from evaluation.rmsd import kabsch
from configs import CACHE_DIR, Rosetta_DIR
from utils.time_sign import get_time_sign


FILE_DIR = os.path.abspath(os.path.split(__file__)[0])


def _align_(mod_chain: Peptide, ref_chain: Peptide):
    mod_ca, ref_ca = [], []
    mod_atoms = []
    for residue in mod_chain:
        coord_map = residue.get_coord_map()
        mod_ca.append(coord_map['CA'])
        for atom in residue.get_atom_names():
            mod_atoms.append(coord_map[atom])
    for residue in ref_chain:
        coord_map = residue.get_coord_map()
        ref_ca.append(coord_map['CA'])
    _, Q, t = kabsch(np.array(mod_ca), np.array(ref_ca))
    mod_atoms = np.dot(mod_atoms, Q) + t
    mod_atoms = mod_atoms.tolist()
    atom_idx = 0
    residues = []
    for mod_res, ref_res in zip(mod_chain, ref_chain):
        coord_map = {}
        for atom in mod_res.get_atom_names():
            coord_map[atom] = mod_atoms[atom_idx]
            atom_idx += 1
        residue = deepcopy(ref_res)
        residue.set_coord(coord_map)
        residues.append(residue)
    return Peptide(ref_chain.get_id(), residues)


def openmm_relax(pdb, out_pdb=None, excluded_chains=None, inverse_exclude=False):

    tolerance = 2.39 * kilocalories_per_mole
    # tolerance = 2.39
    stiffness = 10.0 * kilocalories_per_mole / (angstroms ** 2)

    if excluded_chains is None:
        excluded_chains = []

    if out_pdb is None:
        out_pdb = os.path.join(CACHE_DIR, 'output.pdb')

    fixer = PDBFixer(pdb)
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.findMissingResidues()
    fixer.findMissingAtoms()  # [OXT]
    fixer.addMissingAtoms()
    
    # force_field = ForceField("amber14/protein.ff14SB.xml")
    force_field = ForceField('amber99sb.xml')
    modeller = Modeller(fixer.topology, fixer.positions)
    modeller.addHydrogens(force_field)
    # system = force_field.createSystem(modeller.topology)
    system = force_field.createSystem(modeller.topology, constraints=HBonds)

    force = CustomExternalForce("0.5 * k * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
    force.addGlobalParameter("k", stiffness)
    for p in ["x0", "y0", "z0"]:
        force.addPerParticleParameter(p)

    # add flexible atoms
    for residue in modeller.topology.residues():
        if (not inverse_exclude and residue.chain.id in excluded_chains) or \
           (inverse_exclude and residue.chain.id not in excluded_chains): # antigen
            for atom in residue.atoms():
                system.setParticleMass(atom.index, 0)
        
        for atom in residue.atoms():
            # if atom.name in ['N', 'CA', 'C', 'CB']:
            if atom.element.name != 'hydrogen':
                force.addParticle(atom.index, modeller.positions[atom.index])

    system.addForce(force)
    integrator = LangevinIntegrator(0, 0.01, 0.0)
    # platform = Platform.getPlatformByName('CPU')
    # platform = Platform.getPlatformByName('CUDA')

    simulation = Simulation(modeller.topology, system, integrator)#, platform)
    simulation.context.setPositions(modeller.positions)
    simulation.minimizeEnergy(tolerance)
    state = simulation.context.getState(getPositions=True, getEnergy=True)

    with open(out_pdb, 'w') as fout:
        PDBFile.writeFile(simulation.topology, state.getPositions(), fout, keepIds=True)
    
    return out_pdb


def rosetta_sidechain_packing(pdb, out_pdb=None):
    unique_id = get_time_sign()
    rosetta_exe = os.path.join(Rosetta_DIR, 'fixbb.static.linuxgccrelease')
    resfile = os.path.join(CACHE_DIR, 'resfile.txt')
    if not os.path.exists(resfile):
        with open(resfile, 'w') as fout:
            fout.write(f'NATAA\nstart')
    cmd = f'{rosetta_exe} -in:file:s {pdb} -in:file:fullatom -resfile {resfile} ' +\
            f'-nstruct 1 -out:path:all {CACHE_DIR} -out:prefix {unique_id} -overwrite -mute all'
    p = os.popen(cmd)
    p.read()
    p.close()
    filename = splitext(basename(pdb))[0]
    tmp_pdb = os.path.join(CACHE_DIR, unique_id + filename + '_0001.pdb')
    if out_pdb is None:
        return tmp_pdb
    os.system(f'mv {tmp_pdb} {out_pdb}')
    return out_pdb
