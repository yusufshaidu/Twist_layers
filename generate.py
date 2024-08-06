import os, sys, yaml,argparse
import numpy as np
from ase.build import mx2
from ase.io import read,write
from ase import Atoms
from ase.build import graphene

from twisted_tmd import twisted_tmd

def main(chem_form_1, chem_form_2, 
         alat_1, alat_2, 
         width_1, width_2, 
         vacuum, angle, 
         ILS, strain_threshold, 
         hbn, nat_prim,
         outfile):
   

    if chem_form_1 == 'C2' or chem_form_1 == 'graphene':
        atom1 = graphene(chem_form_1,a=alat_1, vacuum=15)
    else:
        atom1 = mx2(chem_form_1,a=alat_1, thickness=width_1, vacuum=vacuum)

    if chem_form_2 == 'C2' or chem_form_2 == 'graphene':
        atom2 = graphene(chem_form_2,a=alat_2, vacuum=15)
    else:
        atom2 = mx2(chem_form_2,a=alat_2, thickness=width_2, vacuum=vacuum)

   
    twist_atoms = twisted_tmd(atom1, atom2, angle, ILS, nat_prim)
    if angle != 0.0:
        if chem_form_1 == chem_form_2:
            twist_atoms = twist_atoms.generate_moire_lattice_homo(hbn=hbn)
        elif chem_form_1 != chem_form_2:
            twist_atoms = twist_atoms.generate_moire_lattice_hetero(eps=strain_threshold, hbn=hbn)
    else:
        twist_atoms = twist_atoms.generate_moire_lattice_at_zero_twist(eps=strain_threshold, hbn=False)
        
    Nat = twist_atoms.get_global_number_of_atoms()
    print('total number of atoms',Nat)
    write(outfile, twist_atoms, sort=True)
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='making twisted stuff')
    parser.add_argument('-c', '--config', type=str,
                        help='configuration file', required=True)
    args = parser.parse_args()


    import yaml
    with open(args.config) as f:
        configs = yaml.safe_load(f)

    print(configs)
    
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
            
    main(chem_form_1, chem_form_2, 
         alat_1, alat_2, 
         width_1, width_2, 
         vacuum, angle, 
         ILS, strain_threshold, 
         hbn, nat_prim,
         outfile)
