import numpy as np
from scipy.optimize import fsolve, least_squares
from ase.build import mx2
from ase.io import read,write
from ase import Atoms
from ase.build import graphene
import itertools, time
from ase.build import make_supercell
from numba import jit, int32, float32, prange
from ase.geometry import get_distances

#c++ codes with openmpi support
import approximate_I_omp_module

@jit(['int32[:,:](float32,float32,float32,float32,int32,float32)'],
     nopython=True,parallel=True,
    )
def _compute_approximate_I(a_exact, b_exact, 
                             c_exact, d_exact, 
                             nmax, tol):
    '''compute ijkl, mnqr that satisfy the commensurability condition'''
    i_range = prange(-nmax,nmax+1)
    j_range = range(-nmax,nmax+1)
    k_range = range(-nmax,nmax+1)
    l_range = range(-nmax,nmax+1)

    m_range = range(-nmax,nmax+1)
    n_range = range(-nmax,nmax+1)
    q_range = range(-nmax,nmax+1)
    r_range = range(-nmax,nmax+1)
    
    I = []
     
    #print(16*nmax**8)
    error0 = 1e9
    for i in i_range:
        for j in j_range:
            for k in k_range:
                for l in l_range:
                    det = i*l - j*k
                    if abs(det)<1e-6 or 0 in [i,j,k,l]:
                        continue
                    for m in m_range:
                        for n in n_range:
                            for q in q_range:
                                for r in r_range:
                                    
                                    det2 = m*r - n*q
                                    if abs(det2)<1e-6 or 0 in [m,n,q,r]:
                                        continue

                                    a = (l*m - j*q) / det
                                    b = (l*n - j*r) / det
                                    c = (-k*m + i*q) / det
                                    d = (-k*n + i*r) / det
                                    error = -1e-9
                                    da = abs(a-a_exact)
                                    error = max(error,da)
                                    db = abs(b-b_exact)
                                    error = max(error,db)
                                    dc = abs(c-c_exact)
                                    error = max(error,dc)
                                    dd = abs(d-d_exact)
                                    error = max(error,dd)

                                    #error = max([da,db,dc,dd])
                                    if error < tol:
                                        #print(i,j,k,l,m,n,q,r, error)
                                        I.append([i,j,k,l,m,n,q,r])
                                        error0 = error
    #                                    dpar = [da,db,dc,dd]
    I = np.array(I, dtype=np.int32)
    #print('error reached',error0, 'tol=', tol, 'the number of solutions: ', len(I), 'da,db,dc,dd=: ', dpar)
    return I


class twisted_general:
    '''
       This code generate configurations for twisted bilayer materials.
       We follow closely the approach presented in https://arxiv.org/pdf/2104.09591
       use case: twisted bilayer graphene, bilayer TMD, TMD on hBN substrate ...
    '''
    def __init__(self,atom1,atom2, angle,ILS,nat_prim=3,n_hbn=1, hbn_ILS=3.5, hbn_top=0):
        self.atom_1 = atom1
        self.atom_2 = atom2
        self.angle = angle
        self.ILS = ILS
        self.nat_prim = nat_prim
        self.n_hbn = n_hbn
        self.hbn_ILS = hbn_ILS
        self.hbn_top = hbn_top
    def compute_transformation_parameters_abcd(self, atom_1, atom_2):
        '''This takes in already twisted lattices'''
        
        as1_dot_as2 = np.dot(atom_2.cell[0], atom_2.cell[1])
        as1_dot_ao1 = np.dot(atom_2.cell[0], atom_1.cell[0])
        as2_dot_ao1 = np.dot(atom_2.cell[1], atom_1.cell[0])
        as1_dot_ao2 = np.dot(atom_2.cell[0], atom_1.cell[1])
        as2_dot_ao2 = np.dot(atom_2.cell[1], atom_1.cell[1])
        
        as1_squares, as2_squares, _ = np.linalg.norm(atom_2.cell, axis=-1)**2
        
        mat_B = np.array([[as1_squares, as1_dot_as2],[as1_dot_as2,as2_squares]])
        #print(mat_B)
        v1 = np.array([as1_dot_ao1, as2_dot_ao1])
        v2 = np.array([as1_dot_ao2, as2_dot_ao2])
        a,b = np.linalg.solve(mat_B, v1)
        c,d = np.linalg.solve(mat_B, v2)
        return [a,b,c,d]

    def minimize_moire_area(self, atom_1, atom_2, I):
        Ao_prim = np.linalg.norm(np.cross(atom_1.cell[0], atom_1.cell[1]))
        As_prim = np.linalg.norm(np.cross(atom_2.cell[0], atom_2.cell[1]))
        
        Ao_min = 1e9
        Io = []
        As_min = 1e9
        Is = []
         
        for _I in I:
            i,j,k,l,m,n,q,r = _I
            loss = np.abs(i*l - k*j)
            loss_o = np.abs(m*r - q*n)
            if loss - Ao_min < 1e-6 or np.abs(m*r - q*n) - As_min<1e-6:
                Ao_min = loss
                As_min = loss_o
                Io = [i,j,k,l]
                Is = [m,n,q,r]
                print('Io',Io, Ao_min,'Is',Is, As_min )
        
            #loss = np.abs(m*r - q*n)
            #if loss - As_min < 1e-6:
            #    As_min = loss
            #    Is = [m,n,q,r]
            #    print(Is, As_min)
        No = atom_1.get_global_number_of_atoms() * Ao_min
        Ns = atom_2.get_global_number_of_atoms() * As_min
        return Io,Is, Ao_prim*Ao_min, As_min*As_prim, No, Ns
                
    
    def compute_approximate_I(self, a_exact, b_exact, 
                             c_exact, d_exact, 
                             nmax, tol): 
        #return approximate_I_omp_module.compute_approximate_I(a_exact, 
        #        b_exact, c_exact, d_exact, nmax, tol)    
        return _compute_approximate_I(a_exact, b_exact, 
                             c_exact, d_exact, 
                            nmax, tol)
    def repeat_cell(self,atoms, n,m):

        all_pos = []
        all_symbol = []
        cell=np.array([atoms.cell[0]*n, atoms.cell[1]*m, atoms.cell[2]])        
        #print('repeat cell',cell, atoms.cell[0], atoms.cell[1])
        dn = 1
        nrange = range(n)
        if n < 0:
            dn = -1
            nrange = range(n, 1, 1)
            
        dm = 1
        mrange = range(m)
        
        if m < 0:
            dm =-1
            mrange = range(m,1,1)
        #print(nrange, mrange)    
        for i in nrange:
            for j in mrange:        
                for k, p in enumerate(atoms.positions):
                    all_pos.append(p + i*atoms.cell[0] + j*atoms.cell[1])
                    all_symbol.append(atoms.symbols[k])
        #print(all_pos,all_symbol)
        _atoms = Atoms(cell=cell,positions=np.array(all_pos), symbols=all_symbol, pbc=True)
        _atoms.wrap(pbc=True)
        return _atoms
    
    def generate_general_moire_lattice_homo(self, hbn=False, eps=0.01, nmax=5):
        '''compute twisted TMD'''
        angle = self.angle
        print(angle)
      
      
        
        #note that these indexes depends on the choice of lattice vectors
        # for the specific ASE lattice, the following definitions are correct
        atom_1 = self.atom_1.copy()
        atom_1.rotate(angle/2,'z',rotate_cell=True)
        
       
        atom_2 = self.atom_2.copy()
        
        atom_2.rotate(-angle/2,'z',rotate_cell=True) 
        a,b,c,d = self.compute_transformation_parameters_abcd(atom_1, atom_2)
        time0 = time.time()  
        I = self.compute_approximate_I(a,b,c,d,nmax,eps)
        print('total time spent search for commensurate lattice: ', time.time()-time0)
        Io,Is,Ao,As,No,Ns = self.minimize_moire_area(atom_1, atom_2, I)
        
        i,j,k,l = Io
        m,n,q,r = Is
        det = i*l-k*j
        a_approx = (l*m - j*q) / det
        b_approx = (l*n - j*r) / det
        c_approx = (-k*m + i*q) / det
        d_approx = (-k*n + i*r) / det

        print(a,b,c,d, Io,Is,Ao,As,No,Ns)
        print()
        print(len(I), 'strains',np.abs(a_approx-a), np.abs(b_approx-b), np.abs(c_approx-c), np.abs(d_approx-d))
        
        top_layer = self.repeat_cell(atom_1, 4*Io[0], 4*Io[1])
        bottom_layer = self.repeat_cell(atom_2, 4*Is[0], 4*Is[1])
        
           
        R11 = Io[0] * atom_1.cell[0] + Io[1] * atom_1.cell[1]
        R12 = Is[0] * atom_2.cell[0] + Is[1] * atom_2.cell[1]
        R21 = Io[2] * atom_1.cell[0] + Io[3] * atom_1.cell[1]
        R22 = Is[2] * atom_2.cell[0] + Is[3] * atom_2.cell[1]
       
        cell1 = np.array([
            R11,
            R21,
            atom_1.cell[2]])
        cell2 = np.array([
            R12,
            R22,
            atom_2.cell[2]])
        
        cell = 0.5*(cell1+cell2)
        #print('cell', cell1,cell2,cell)
        spos_t = []
        pos_t = []
        symb_t = []
        
        top_layer = Atoms(cell=cell1, positions=top_layer.positions, symbols=top_layer.symbols, pbc=True)
        
        for k,p in enumerate(top_layer.positions):
            if len(pos_t) == 0:
                dist = 1e6
            else:
            
                _, dist = get_distances(p, p2=np.array(pos_t), cell=cell1, pbc=True)
                dist = dist.min()
            
            #scaled_p = np.linalg.inv(cell1.T) @ p
                
            if dist > 0.1: # and np.all(np.abs(scaled_p)<1.0):
                pos_t.append(p)
                symb_t.append(top_layer.symbols[k])
                #print(dist)
            
                
            
        top_layer = Atoms(cell=cell, positions=pos_t, symbols=symb_t, pbc=True)
        #print(top_layer.get_global_number_of_atoms())
        
        pos_b = []
        symb_b = []
        spos_b = []
        bottom_layer = Atoms(cell=cell2, positions=bottom_layer.positions, symbols=bottom_layer.symbols, pbc=True)
        for k,p in enumerate(bottom_layer.positions):
            if len(pos_b) == 0:
                dist = 1e6
            else:
                #print(p, pos_t)
                _, dist = get_distances(p, p2=np.array(pos_b), cell=cell2, pbc=True)
                dist = dist.min()
                
            #scaled_p = np.linalg.inv(cell2.T) @ p
                        
            if dist > 0.1:# and np.all(np.abs(scaled_p)<1.0):
                pos_b.append(p)
                symb_b.append(bottom_layer.symbols[k])
                #print(dist)
            
        
        bottom_layer = Atoms(cell=cell, positions=pos_b, symbols=symb_b, pbc=True)
        top_layer.positions[:,2] += self.ILS/2
        width = top_layer.positions[:,2].max()-top_layer.positions[:,2].min()
        bottom_layer.positions[:,2] -= (self.ILS/2 + width)
        
        #print(bottom_layer.get_global_number_of_atoms())
        #write('CrSBr/top.vasp', top_layer, sort=True)
        #write('CrSBr/botom.vasp', bottom_layer, sort=True)
        
        
        scaled_positions = top_layer.get_scaled_positions()
        scaled_positions = np.append(scaled_positions, bottom_layer.get_scaled_positions())
        scaled_positions = np.reshape(scaled_positions, [-1,3])
        symbols = list(top_layer.symbols) + list(bottom_layer.symbols)
        cell = top_layer.cell + bottom_layer.cell
        cell /= 2
        atoms = Atoms(symbols=symbols, scaled_positions=scaled_positions, cell=cell,pbc=True)
        
        return atoms    

