import numpy as np
from scipy.optimize import fsolve, least_squares
from ase.build import mx2
from ase.io import read,write
from ase import Atoms
from ase.build import graphene

class twisted_tmd:
    '''
       This code generate configurations for twisted bilayer materials.
       We follow closely the approach presented in PHYSICAL REVIEW B 90, 155451 (2014)
       use case: twisted bilayer graphene, bilayer TMD, TMD on hBN substrate ...
    '''
    def __init__(self,atom1,atom2, angle,ILS,nat_prim=3):
        self.atom_1 = atom1
        self.atom_2 = atom2
        self.angle = angle
        self.ILS = ILS
        self.nat_prim = nat_prim
        
    def compute_angle(self, n):
        '''
        return cos theta in degrees
        
        '''
        cos_theta = (n[0]**2 + 4*n[0]*n[1] + n[1]**2) / (2*(n[0]**2 + n[0]*n[1] + n[1]**2))
        return np.arccos(cos_theta) * 180/np.pi
    
    def compute_angle_general(self, n):
        '''
        return cos theta in degrees
        
        '''
        n1,m1,n2,m2 = n
        cos_theta = (n1*n2 + 2*(n1*m2 + n2*m1) + m1*m2) / (2*(n1**2 + n1*m1 + m1**2)**0.5 * (n2**2 + n2*m2 + m2**2)**0.5 + 1e-12 )
        
        return np.arccos(cos_theta) * 180/np.pi
    
    
    def rotate_atoms(self, atom, angle):
        atom.rotate(angle, 'z',rotate_cell=True)
        return atom
    
    def get_number_of_atoms(self, n):
        return self.nat_prim*(n[0]**2 + n[0]*n[1] + n[1]**2)
    
    def search_nm_indexes(self, angle):
        if np.abs(angle) < 1e-6:
            return [1,1]
        n0 = [100,100]
        nat0 = self.get_number_of_atoms(n0)
        for i in range(0,100):
            for j in range(0,100):
                if i==j:
                    continue
                n =[i,j]
                theta = self.compute_angle(n)
               # print(theta, np.abs(theta - angle))
                #nat = self.get_number_of_atoms(3,n)
                
                if np.abs(theta - angle) < 0.1:
                    
                    nat = self.get_number_of_atoms(n)
                   # print(nat,n,theta, np.abs(theta - angle))
                    if nat < nat0:
                        n0 = n
                        nat0 = nat
                    
                    
                    #all_idx.append(n)
                    #break
            #if np.abs(theta - angle) < 0.1:
            #    break
        
        return n0
    def search_nm_indexes_general(self, angle, a1, a2, eps):
        if np.abs(angle) < 1e-6:
            return [1,1]
        n0 = [30,30,30,30]
        nat0 = self.get_number_of_atoms(n0[:2]) + self.get_number_of_atoms(n0[2:])
        for i in range(1,30):
            for j in range(1,30):
                for k in range(1,30):
                    for l in range(1,30):
                        if i==j or l == k:
                            continue
                        n =[i,j,k,l]
                        theta = self.compute_angle_general(n)
                        r1 = a1 * np.sqrt(i**2+ i * j + j**2)
                        r2 = a2 * np.sqrt(k**2+ k * l + l**2)
                        
                        da = np.abs(1 - r1/r2)
                        
                        if np.abs(theta - angle) < 0.1 and da < eps:
                            

                            nat = self.get_number_of_atoms(n[:2]) + self.get_number_of_atoms(n[2:])
                            #print(theta,angle,da, n, nat)
                           # print(nat,n,theta, np.abs(theta - angle))
                            if nat < nat0:
                                n0 = n
                                nat0 = nat
        
        return n0
    
    def generate_superperiodic_lattice(self, atom, n,m,nprim,mprim):
        
        
        a1_sc = atom.cell[0] * n + atom.cell[1] * m
        a2_sc = atom.cell[0] * nprim + atom.cell[1] * mprim
        a_cell = np.array([a1_sc, a2_sc, atom.cell[2]])
        print(a_cell)
        #print(np.linalg.norm(a_cell, axis=-1))
        
        idx = [n,m,nprim,mprim]
        print(f'expected number of for tmd atoms={(n**2+m**2+np.abs(n*m))*self.nat_prim}')
        
        #get increaments
        d = []
        for i in idx:
            if i < 0:
                d.append(-1)
            else:
                d.append(1)
        
        s_positions = []
        s_symb = []
        symb = list(atom.symbols)
        # generate 
        for k,p in enumerate(atom.positions):  
            for i1 in range(0,n+d[0],d[0]):
                for i2 in range(0,m+d[1],d[1]):
                    for j1 in range(0,nprim+d[2],d[2]):
                        for j2 in range(0,mprim+d[3],d[3]):
                            if i1 == i2 or j1 == j2 or np.sum(np.abs([i1,i2,j1,j2])) == 0.0:
                                continue
                            ij1 = i1+j1
                            ij2 = i2+j2
                            #p_s = p0 + i1 * a1 + i2 * a2 + j1 * a1 + j2 * a2
                            p_s = p + ij1 * atom.cell[0] + ij2 * atom.cell[1]
                            norm_coord = np.matmul(np.linalg.inv(a_cell.T), p_s) 
                            s_positions.append(p_s)
                            s_symb.append(symb[k])
                           

        #print(i,len(s_positions))
        
        #extract atoms that live in the super periodic cell
        atoms = Atoms(cell=a_cell,positions=s_positions,symbols=s_symb, pbc=True)
        atoms.wrap(pbc=True)
        in_positions = []
        in_symbols = []
        for i, p in enumerate(atoms.get_scaled_positions()):
            kk = 0
            #check of atoms are already considers
            for _p in in_positions:
                delta = np.linalg.norm(p-_p)
                if delta < 1e-3:
                    kk = 1
                    break
            if kk == 1:
                continue
            if np.max(np.abs(p)) < 1.0-1e-6:
                in_positions.append(p)
                in_symbols.append(atoms.symbols[i])
                #print(np.max(p), p)
        #print(len(pp))
        atoms = Atoms(cell=a_cell,scaled_positions=in_positions,symbols=in_symbols, pbc=True)
        return atoms

    def generate_moire_lattice_homo(self, hbn=False):
        '''compute twisted homobilayer TMD'''
        #get n,m for a give rotation angle
        indexes = self.search_nm_indexes(self.angle)
        print(indexes)
        # check that angle is correct
        _angle = self.compute_angle(indexes)
        if np.abs(_angle-self.angle) < 1e-1:
            print(f'indices are correctly computed')
        
        #note that these indexes depends on the choice of lattice vectors
        # for the specific ASE lattice, the following definitions are correct
            
        n,m = indexes
        nprim = m; mprim = m + n
        
        #rotate ase vectors from [90,90,120] to [90,90,60]
        #intial lattice vectors: cell = [[1,0,0],[-1/2, root(3)/2,0],[0,0,c/a]] * a
        #final lattice vectors after rotation: cell = [[root(3)/2,1/2,0],[-root(3)/2, 1/2,0],[0,0,c/a]] * a
        #this obtained by rotation of the entire system by 30 degrees
        
        self.atom_1.rotate(30,'z',rotate_cell=True)
        top_layer = self.generate_superperiodic_lattice(self.atom_1, n,-m,nprim,mprim)
        top_layer.positions[:,2] += self.ILS / 2
        top_layer = self.rotate_atoms(top_layer, self.angle/2)
        #bottom layer has different lattice vectors
        self.atom_2.rotate(30,'z',rotate_cell=True)
        nprim = n; mprim = n+m
        bottom_layer = self.generate_superperiodic_lattice(self.atom_2, m,-n,nprim,mprim)
        bottom_layer.positions[:,2] -= self.ILS / 2
        bottom_layer = self.rotate_atoms(bottom_layer, -self.angle/2)
        a11 = np.linalg.norm(top_layer.cell, axis=-1)[0]
        a22 = np.linalg.norm(bottom_layer.cell, axis=-1)[0]
        print(a11,a22)

        if a11 > a22:
            cell =  top_layer.cell
        else:
            cell =  bottom_layer.cell

        scaled_positions = top_layer.get_scaled_positions()
        scaled_positions = np.append(scaled_positions, bottom_layer.get_scaled_positions())
        scaled_positions = np.reshape(scaled_positions, [-1,3])
        symbols = list(top_layer.symbols) + list(bottom_layer.symbols)
        atoms = Atoms(symbols=symbols, scaled_positions=scaled_positions, cell=cell,pbc=True)



        if hbn:
            atoms_hbn = graphene('BN', a=2.504, vacuum=15)
            ahbn = 2.504
            nr = round(a22/ahbn)
            strain = np.abs(1 - a22 / (nr*ahbn))
            
            zmin_tmd = np.min(bottom_layer.positions[:,2])
            zmin_hbn = np.min(atoms_hbn.positions[:,2])
            atoms_hbn.positions[:,2] -= (zmin_hbn-zmin_tmd + 3.35)
            atoms_hbn = atoms_hbn.repeat((nr,nr,1))
            atoms += atoms_hbn
            
        return atoms
                
    def minimize_strain(self, a1,a2,eps=.01):
        '''reduce strain between two mismatch TMDs
        '''
        
        
        strain = np.abs(1 - a2/a1)
        print(f'initial strain is {strain}')
        for n1 in range(1,50):
            for n2 in range(1,50):
                strain = np.abs(1 - n2*a2/(n1*a1))
                if strain < eps:
                    print(strain)
                    return n1,n2,strain
                
    def generate_moire_lattice_hetero(self, eps, hbn=False):
        '''compute twisted heterobilayer TMD
           In this case, we extend the PHYSICAL REVIEW B 90, 155451 (2014)
           to account for lattice mismatch between layers
        '''
        
        atom_1 = self.atom_1
        atom_2 = self.atom_2
        
        a1 = np.linalg.norm(atom_1.cell, axis=-1)[0]
        a2 = np.linalg.norm(atom_2.cell, axis=-1)[0]
        #print(a1,a2)
        #get n1,m1,n2,m2 for a give rotation angle between
        #lattice vectors defined by n1 and m1 for layer 1 and n2 and m2 for layer 2
        n1,m1,n2,m2 = self.search_nm_indexes_general(self.angle, a1, a2,eps)
        print(n1,m1,n2,m2)
        # check that angle is correct
        _angle = self.compute_angle_general((n1,m1,n2,m2))
        if np.abs(_angle-self.angle) < 0.5e-1:
            print(f'indices are correctly computed @ {_angle} and target {self.angle}')
        
        #note that these indexes depends on the choice of lattice vectors
        # for the specific ASE lattice, the following definitions are correct
        
        
       
        nprim = m1; mprim = m1 + n1
        
        #rotate ase vectors from [90,90,120] to [90,90,60]
        #intial lattice vectors: cell = [[1,0,0],[-1/2, root(3)/2,0],[0,0,c/a]] * a
        #final lattice vectors after rotation: cell = [[root(3)/2,1/2,0],[-root(3)/2, 1/2,0],[0,0,c/a]] * a
        #this obtained by rotation of the entire system by 30 degrees
        
        atom_1.rotate(30,'z',rotate_cell=True)
        top_layer = self.generate_superperiodic_lattice(atom_1, n1,-m1,nprim,mprim)
        top_layer.positions[:,2] += self.ILS / 2
        top_layer = self.rotate_atoms(top_layer, self.angle/2)
        #write('test.vasp', top_layer)
        #this should be different
        atom_2.rotate(30,'z',rotate_cell=True)
        #bottom layer has different lattice vectors
        nprim = n2; mprim = n2+m2
        bottom_layer = self.generate_superperiodic_lattice(atom_2, m2,-n2,nprim,mprim)
        bottom_layer.positions[:,2] -= self.ILS / 2
        #match the two cell
        bottom_layer = self.rotate_atoms(bottom_layer, -self.angle/2)
        #write('test2.vasp', bottom_layer)
        #minimize the angle betwee
        a1 = np.linalg.norm(top_layer.cell, axis=-1)[0]
        a2 = np.linalg.norm(bottom_layer.cell, axis=-1)[0]
        print(a1,np.linalg.norm(top_layer.cell, axis=-1)[1])
        print(a2,np.linalg.norm(bottom_layer.cell, axis=-1)[1])
        strain = np.abs(1 - a1/a2)
        print(f'lattice parameters at strain {strain} upon relaxations are top {a1} and bottom {a2}')
        
        
        if a1 > a2:
            cell =  top_layer.cell
        else:
            cell =  bottom_layer.cell
            
        scaled_positions = top_layer.get_scaled_positions()
        scaled_positions = np.append(scaled_positions, bottom_layer.get_scaled_positions())
        scaled_positions = np.reshape(scaled_positions, [-1,3])
        symbols = list(top_layer.symbols) + list(bottom_layer.symbols)
        #print(symbols)
        #atoms.scaled_positions = scaled_positions
        #atoms.symbols = symbols
        #atoms.pbc = True
        atoms = Atoms(symbols=symbols, scaled_positions=scaled_positions, cell=cell,pbc=True)
        
                
        if hbn:
            atoms_hbn = graphene('BN', a=2.504, vacuum=15)
            a22 = n2*a2
            ahbn = 2.504
            nr = round(a22/ahbn)
            strain = np.abs(1 - a22 / (nr*ahbn))
            
            zmin_tmd = np.min(bottom_layer.positions[:,2])
            zmin_hbn = np.min(atoms_hbn.positions[:,2])
            atoms_hbn.positions[:,2] -= (zmin_hbn-zmin_tmd + 3.35)
            atoms_hbn = atoms_hbn.repeat((nr,nr,1))
            atoms += atoms_hbn
            
        return atoms
    
    def generate_moire_lattice_at_zero_twist(self, eps, hbn=False):
        '''compute strained TMD at zero twist'''
        
       
        #self.atom_1.rotate(30,'z',rotate_cell=True)
        top_layer = self.atom_1
        top_layer.positions[:,2] += self.ILS / 2
        
        
        #this can be same
        #self.atom_2.rotate(30,'z',rotate_cell=True)
        
        nprim = n; mprim = n+m
        bottom_layer = self.atom_2
        bottom_layer.positions[:,2] -= self.ILS / 2
        
        a1 = np.linalg.norm(top_layer.cell, axis=-1)[0]
        a2 = np.linalg.norm(bottom_layer.cell, axis=-1)[0]
        print(f'lattice parameters before strain relaxations are top {a1} and bottom {a2}')
      
        n1,n2, strain = self.minimize_strain(a1,a2,eps=eps)
        print(f'lattice parameters after strain relaxations are top {a1*n1} and bottom {a2*n2} with {n1} and {n2}')
        print(f'final strain {strain}')
        
        top_layer = top_layer.repeat((n1,n1,1))
        bottom_layer = bottom_layer.repeat((n2,n2,1))

        a1 = np.linalg.norm(top_layer.cell, axis=-1)[0]
        a2 = np.linalg.norm(bottom_layer.cell, axis=-1)[0]

        if a1 > a2:
            cell =  top_layer.cell
        else:
            cell =  bottom_layer.cell

        scaled_positions = top_layer.get_scaled_positions()
        scaled_positions = np.append(scaled_positions, bottom_layer.get_scaled_positions())
        scaled_positions = np.reshape(scaled_positions, [-1,3])
        symbols = list(top_layer.symbols) + list(bottom_layer.symbols)
        atoms = Atoms(symbols=symbols, scaled_positions=scaled_positions, cell=cell,pbc=True)
            
        if hbn:
            atoms_hbn = graphene('BN', a=2.504, vacuum=15)
            a22 = n2*a2
            ahbn = 2.504
            nr = round(a22/ahbn)
            strain = np.abs(1 - a22 / (nr*ahbn))
            #n11,n22,strain = self.minimize_strain(ahbn,a22, eps=eps)
            zmin_tmd = np.min(bottom_layer.positions[:,2])
            zmin_hbn = np.min(atoms_hbn.positions[:,2])
            atoms_hbn.positions[:,2] -= (zmin_hbn-zmin_tmd + 3.35)
            atoms_hbn = atoms_hbn.repeat((nr,nr,1))
            atoms += atoms_hbn
        return atoms
