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
import math
#c++ codes with openmpi support
#import approximate_I_omp_module

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
                    if abs(det)<1e-6 or abs(i*j*k*l) < 1e-4: # make sure that the area is finite
                        continue
                    for m in m_range:
                        for n in n_range:
                            for q in q_range:
                                for r in r_range:
                                    
                                    det2 = m*r - n*q
                                    if abs(det2)<1e-6 or abs(m*n*q*r) < 1e-4:
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
                                        #error0 = error
    #                                    dpar = [da,db,dc,dd]
    I = np.array(I, dtype=np.int32)
    #print('error reached',error0, 'tol=', tol, 'the number of solutions: ', len(I), 'da,db,dc,dd=: ', dpar)
    return I

def _compute_approximate_I_SK(a_exact, b_exact, 
                             c_exact, d_exact, 
                             atom_1, atom_2, 
                             nmax, tol):
    '''compute ijkl, mnqr that satisfy the commensurability condition altered by SK'''
    i_range = prange(-nmax,nmax+1)
    j_range = range(-nmax,nmax+1)
    k_range = range(-nmax,nmax+1)
    #l_range = range(-nmax,nmax+1)
    l_range = range(0,nmax+1) # SK: I think we can cut this in half by symmetry

    #m_range = range(-nmax,nmax+1)
    #n_range = range(-nmax,nmax+1)
    #q_range = range(-nmax,nmax+1)
    #r_range = range(-nmax,nmax+1)
    
    I = []
     
    #print(16*nmax**8)

    error0 = 1e9
    for i in i_range:
        for j in j_range:
            for k in k_range:
                for l in l_range:
                    det = i*l - j*k
                    if abs(det)<1e-6 or abs(i*j*k*l) < 1e-4: # make sure that the area is finite
                        continue

                    m_exact = i*a_exact + j*c_exact
                    n_exact = i*b_exact + j*d_exact
                    q_exact = k*a_exact + l*c_exact
                    r_exact = k*b_exact + l*d_exact

                    m = round(m_exact)
                    n = round(n_exact)
                    q = round(q_exact)
                    r = round(r_exact)
                                    
                    det2 = m*r - n*q
                    if abs(det2)<1e-6 or abs(m*n*q*r) < 1e-4:
                        continue

                    a1_o = i*atom_1.cell[0] + j*atom_1.cell[1]
                    a2_o = k*atom_1.cell[0] + l*atom_1.cell[1]
                    a1_s = m*atom_2.cell[0] + n*atom_2.cell[1]
                    a2_s = q*atom_2.cell[0] + r*atom_2.cell[1]

                    loss_strain = math.sqrt(np.linalg.norm( (a1_o - a1_s)/np.linalg.norm(a1_o) )**2 
                    + np.linalg.norm( (a2_o - a2_s)/np.linalg.norm(a2_o) )**2)

                    error = -1e-9
                    ## SK: we used to use the difference between exact values and integer 
                    ##     but we change this to actually strain 
                    error = max(error, loss_strain)
                    #error = max(error,abs(m-m_exact))
                    #error = max(error,abs(n-n_exact))
                    #error = max(error,abs(q-q_exact))
                    #error = max(error,abs(r-r_exact))

                    #error = max([da,db,dc,dd])
                    if error < tol : #and det2 == det
                        #print(i,j,k,l,m,n,q,r, error)
                        I.append([i,j,k,l,m,n,q,r])
                                        #error0 = error
    #                                    dpar = [da,db,dc,dd]

    I = np.array(I, dtype=np.int32)
    if len(I) == 0:
        return None
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
    def generate_superperiodic_lattice(self, atom, n,m,nprim,mprim):
        a1_sc = atom.cell[0] * n + atom.cell[1] * m
        a2_sc = atom.cell[0] * nprim + atom.cell[1] * mprim
        nat_prim = atom.get_global_number_of_atoms()
        a_cell = np.array([a1_sc, a2_sc, atom.cell[2]])
        print(a_cell)
        idx = [n,m,nprim,mprim]
        print(f'expected number of in this layer is={(np.abs(n*mprim-m*nprim))*nat_prim}')
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
        i1 = range(0,n+d[0],d[0])
        i2 = range(0,m+d[1],d[1])
        j1 = range(0,nprim+d[2],d[2])
        j2 = range(0,mprim+d[3],d[3])
        idx = np.array(list(itertools.product(i1,i2,j1,j2)))

        ij1 = idx[:,0] + idx[:,2]
        ij2 = idx[:,1] + idx[:,3]

        #remove dublicates which are way too many in this approach
        idx_sum = np.zeros((idx.shape[0], 2))
        idx_sum[:,0] = ij1
        idx_sum[:,1] = ij2
        idx_sum = np.unique(idx_sum, axis=0)
        ij1 = idx_sum[:,0]
        ij2 = idx_sum[:,1]
        for k,p in enumerate(atom.positions):
            p_s = p + ij1[:,None] * atom.cell[0] + ij2[:,None] * atom.cell[1]
            s_positions = np.append(s_positions, p_s)
            s_symb = np.append(s_symb, [symb[k] for i in range(len(ij1))])
            print(f'done with atoms: {k}')
        #extract atoms that live in the super periodic cell
        s_positions = np.reshape(s_positions, [-1,3])
        print('done generating points')
        atoms = Atoms(cell=a_cell,positions=s_positions,symbols=s_symb, pbc=True)
        atoms.wrap(pbc=True)
        in_positions = []
        in_symbols = []
        nconf = len(atoms.get_scaled_positions())
        for i, p in enumerate(atoms.get_scaled_positions()):
            kk = 0
            p = (p+1e-12)%1.0
            #    continue
            #check of atoms are already considers
            if in_positions:
                delta = np.linalg.norm(p-in_positions, axis=-1)
                if np.min(delta) < 1e-6:
                    continue
            in_positions.append(p)
            in_symbols.append(atoms.symbols[i])
        atoms = Atoms(cell=a_cell,scaled_positions=in_positions,symbols=in_symbols, pbc=True)
        return atoms

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

    def minimize_moire_area(self, atom_1, atom_2, I): # also minimize the strain!
        Ao_prim = np.linalg.norm(np.cross(atom_1.cell[0], atom_1.cell[1]))
        As_prim = np.linalg.norm(np.cross(atom_2.cell[0], atom_2.cell[1]))

        Ao_min = 1e9
        Io = []
        As_min = 1e9
        Is = []
        strain_min = 1e9

        for _I in I:
            i,j,k,l,m,n,q,r = _I
            loss = np.abs(i*l - k*j)
            loss_s = np.abs(m*r - q*n)

            a1_o = i*atom_1.cell[0] + j*atom_1.cell[1]
            a2_o = k*atom_1.cell[0] + l*atom_1.cell[1]
            a1_s = m*atom_2.cell[0] + n*atom_2.cell[1]
            a2_s = q*atom_2.cell[0] + r*atom_2.cell[1]

            loss_strain = math.sqrt(np.linalg.norm( (a1_o - a1_s)/np.linalg.norm(a1_o) )**2 
            + np.linalg.norm( (a2_o - a2_s)/np.linalg.norm(a2_o) )**2)
            if loss - Ao_min < 1e-6 and loss_s - As_min<1e-6 and loss_strain - strain_min < 1e-8: # and loss_strain SK
                Ao_min = loss
                As_min = loss_s
                strain_min = loss_strain
                
                Io = [i,j,k,l]
                Is = [m,n,q,r]
                print('Io',Io, Ao_min,'Is',Is, As_min )
                print(loss_strain)
        
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
                             atom_1, atom_2, 
                             nmax, tol): 
        #return approximate_I_omp_module.compute_approximate_I(a_exact, 
        #        b_exact, c_exact, d_exact, nmax, tol)
        return _compute_approximate_I_SK(a_exact, b_exact, 
                             c_exact, d_exact, 
                             atom_1, atom_2, 
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
        I = self.compute_approximate_I(a,b,c,d,atom_1,atom_2,nmax,eps)
        print('total time spent search for commensurate lattice: ', time.time()-time0)
        if I is None:
            raise ValueError("Solution not found.")
        Io,Is,Ao,As,No,Ns = self.minimize_moire_area(atom_1, atom_2, I)
        
        i,j,k,l = Io
        m,n,q,r = Is
        det = i*l-k*j
        a_approx = (l*m - j*q) / det
        b_approx = (l*n - j*r) / det
        c_approx = (-k*m + i*q) / det
        d_approx = (-k*n + i*r) / det
        a1_exact_v  = a*atom_2.cell[0] + b*atom_2.cell[1]
        a1_exact  = np.linalg.norm(a1_exact_v)
        a1_approx_v  = a_approx*atom_2.cell[0] + b_approx*atom_2.cell[1]
        a1_approx  = np.linalg.norm(a1_approx_v)

        a2_exact_v = c*atom_2.cell[0] + d*atom_2.cell[1]
        a2_exact  = np.linalg.norm(a2_exact_v)
        a2_approx_v  = c_approx*atom_2.cell[0] + d_approx*atom_2.cell[1]
        a2_approx  = np.linalg.norm(a2_approx_v)
        a1,a2,_ = np.linalg.norm(atom_1.cell, axis=-1)

        print(a,b,c,d, Io,Is,Ao,As,No,Ns)
        print('number of solutions found=:', len(I))
        print('strain along a1',np.abs(a1_exact-a1_approx)/a1_exact*100, 'change due to transformation:', a1-a1_exact)
        print('strain along a2',np.abs(a2_exact-a2_approx)/a2_exact*100, 'change due to transformation', a2-a2_exact)
        aom1_exact = np.linalg.norm(i*atom_1.cell[0] + j*atom_1.cell[1])
        aom1_approx = np.linalg.norm(i*a1_approx_v + j*a2_approx_v)

        aom2_exact = np.linalg.norm(k*atom_1.cell[0] + l*atom_1.cell[1])
        aom2_approx = np.linalg.norm(k*a1_approx_v + l*a2_approx_v)
        
        print('strain on moire cell along a1',np.abs(aom1_exact-aom1_approx)/aom1_exact*100, aom1_exact, aom1_approx)
        print('strain on moire cell along a2',np.abs(aom2_exact-aom2_approx)/aom2_exact*100,aom2_exact, aom2_approx)

        ##SK: we remove this restrict the number of units of two layers needed to be the same
        #if abs(det)!=abs(m*r-n*q):
        #   return None
        top_layer = self.generate_superperiodic_lattice(atom_1, *Io)
        bottom_layer = self.generate_superperiodic_lattice(atom_2, *Is)

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

