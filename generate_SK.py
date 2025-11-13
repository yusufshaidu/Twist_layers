import os, sys, yaml,argparse
import numpy as np
from ase.build import mx2
from ase.io import read,write
from ase import Atoms
from ase.build import graphene

from twisted_tmd import twisted_tmd
from bilayer_systems import bilayer_systems
from twisted_general_SK import twisted_general
import warnings
def default_configs():
    configs = {}
    configs['chem_form_1'] = None
    configs['chem_form_2'] = None
    configs['alat_1'] = None
    configs['alat_2'] = None
    configs['thickness_1'] = -1
    configs['thickness_2'] = -1
    configs['vacuum'] = 15
    configs['twist_angle'] = None
    configs['ILS'] = 3.35
    configs['strain_threshold'] = 0.01
    configs['hbn'] = False
    configs['nat_prim'] = 3
    configs['outfile'] = 'systes.vasp'
    configs['bilayer'] = False
    configs['stacking'] = 'AB' 
    
    
    configs['n_hbn'] = 0

    configs['nmax'] = 5
    configs['format'] = 'chem_form' 
    return configs

def check_hexagonal(atom1,atom2):
    ang1 = np.arccos(np.dot(atom1.cell[0], atom1.cell[1]) 
            / np.linalg.norm(atom1.cell[0]) 
            / np.linalg.norm(atom1.cell[1])) * 180/np.pi
    ang2 = np.arccos(np.dot(atom2.cell[0], atom2.cell[1]) 
            / np.linalg.norm(atom2.cell[0]) 
            / np.linalg.norm(atom2.cell[1])) * 180/np.pi
    hexagonal = True
    if abs(ang1 - 60) > 1e-6 or abs(ang1 - 120) > 1e-6:
        hexagonal = False
    if abs(ang2 - 60) > 1e-6 or abs(ang2 - 120) > 1e-6:
        hexagonal = False

    return hexagonal

def main(chem_form_1, chem_form_2, 
         alat_1, alat_2, 
         width_1, width_2, 
         vacuum, angle, 
         ILS, strain_threshold, 
         hbn, nat_prim,
         outfile,
         bilayer,
         stacking,
         n_hbn,
         nmax=5,
         _format='chem_form'):
   
    if _format=='chem_form':
        if chem_form_1 in ['BN', 'C2', 'graphene', 'c2', 'hbn', 'bn']:
            atom1 = graphene(chem_form_1,a=alat_1, vacuum=vacuum)
        else:
            atom1 = mx2(chem_form_1,a=alat_1, thickness=width_1, vacuum=vacuum)

        if chem_form_2 in ['BN', 'C2', 'graphene', 'c2', 'hbn', 'bn']:
            atom2 = graphene(chem_form_2,a=alat_2, vacuum=vacuum)
        else:
            atom2 = mx2(chem_form_2,a=alat_2, thickness=width_2, vacuum=vacuum)
    elif _format=='file':
        atom1 = read(chem_form_1)
        atom2 = read(chem_form_2)

    if bilayer:
        atoms = bilayer_systems(chem_form_1, alat=alat_1, 
                   chem_form2=chem_form_2, alat2=alat_2,
                   thickness=width_1,thickness2=width_2, 
                   stacking=stacking, ILS=ILS)
    else:
        hexagonal = check_hexagonal(atom1,atom2)
        if hexagonal:
            twist_atoms = twisted_tmd(atom1, atom2, angle, ILS, nat_prim, n_hbn)
            if angle != 0.0:
                if chem_form_1 == chem_form_2:
                    atoms = twist_atoms.generate_moire_lattice_homo(hbn=hbn)
                elif chem_form_1 != chem_form_2:
                    atoms = twist_atoms.generate_moire_lattice_hetero(eps=strain_threshold, hbn=hbn)
            else:
                atoms = twist_atoms.generate_moire_lattice_at_zero_twist(eps=strain_threshold, hbn=hbn)
        else:
            if n_hbn > 0:
                warnings.warn('hbn substrate is currently not implemented: Only twisted bilayer will be produced')
            twist_atoms = twisted_general(atom1, atom2, angle, ILS, n_hbn=0)
            atoms = twist_atoms.generate_general_moire_lattice_homo(eps=strain_threshold,nmax=nmax)

        
    Nat = atoms.get_global_number_of_atoms()
    print('total number of atoms',Nat)
    
    write(outfile, atoms, sort=True, format='vasp', direct=True)
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='making twisted stuff')
    parser.add_argument('-c', '--config', type=str,
                        help='configuration file', required=True)
    args = parser.parse_args()


    import yaml
    with open(args.config) as f:
        configs = yaml.safe_load(f)

    #print(configs)
    _configs = default_configs()
    for key in _configs.keys():
        if key not in configs:
            configs[key] = _configs[key]

    chem_form_1 = configs['chem_form_1']
    chem_form_2 = configs['chem_form_2']
    alat_1 = configs['alat_1']
    alat_2 = configs['alat_2']
    width_1 = configs['thickness_1']
    width_2 = configs['thickness_2']
    vacuum = configs['vacuum']
    angle = configs['twist_angle']
    ILS = configs['ILS']
    strain_threshold = configs['strain_threshold']
    hbn = configs['hbn']
    nat_prim = configs['nat_prim']
    outfile = configs['outfile']
    bilayer = configs['bilayer']
    stacking = configs['stacking']
    n_hbn = 0
    if hbn:
        n_hbn = configs['n_hbn']

    nmax = configs['nmax']
    _format = configs['format']

    main(chem_form_1, chem_form_2, 
        alat_1, alat_2, 
        width_1, width_2, 
        vacuum, angle, 
        ILS, strain_threshold, 
        hbn, nat_prim,
        outfile,
        bilayer,
        stacking,
        n_hbn,
        nmax=nmax,
        _format=_format)
