from ase.io import read, write
import numpy as np
from ase.build import mx2
from ase.io import read,write
from ase import Atoms
from ase.build import graphene

def bilayer_systems(chem_form, alat, chem_form2=None, alat2=None,
        thickness=None, thickness2=None, stacking='AB', ILS=3.35):
    '''
    if heterostructure, we simply generate configurations with average lattice constant
    alat and alat2 should be similar systems
    '''
    
    if chem_form in ['BN', 'C2', 'graphene', 'c2', 'hbn', 'bn']:

        atom1 = graphene(chem_form, a=alat, vacuum=15)
        if chem_form2:
            atom2 = graphene(chem_form2, a=alat2, vacuum=15)
    else:
        atom1 = mx2(chem_form, a=alat, thickness=thickness, vacuum=15)
        if chem_form2:
            atom2 = mx2(chem_form2, a=alat2, thickness=thickness2, vacuum=15)
    if not chem_form2:
        atom2 = atom1.copy()

    if stacking[-1] == 'B':
        atom2.rotate(60, 'z', rotate_cell=False)
        atom2.positions[:,2] += ILS
        if chem_form2 not in ['C2', 'graphene', 'c2']:
            dp = (atom1.cell[0] * 2/3 + atom1.cell[1]/3)
            atom2.positions[:,:2] += dp[:2]
    else:
        atom2.positions[:,2] += ILS
    if chem_form2:
        cell = 0.5 * (atom1.cell + atom2.cell)
        scaled_positions = atom1.get_scaled_positions()
        scaled_positions = np.append(scaled_positions, atom2.get_scaled_positions())
        scaled_positions = np.reshape(scaled_positions, [-1,3])
        symbols = list(atom1.symbols) + list(atom2.symbols)
        atoms = Atoms(symbols=symbols, scaled_positions=scaled_positions, cell=cell, pbc=True)
    else:
        atoms = atom1+atom2
    return atoms
