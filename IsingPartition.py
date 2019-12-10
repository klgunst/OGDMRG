from idmrg import IDMRG
import numpy as np
import pickle
from time import time


def IsingMPO(β=0.5 * np.log(1 + np.sqrt(2))):
    """MPO for Ising, default value is at the critical point
    """
    from scipy.linalg import sqrtm
    matrix = np.array([[np.exp(β), np.exp(-β)],
                       [np.exp(-β), np.exp(β)]])
    sq_mat = sqrtm(matrix)

    MPO = np.zeros((2, 2, 2, 2))
    MPO[0, 0, 0, 0] = 1
    MPO[1, 1, 1, 1] = 1
    return np.einsum('ijkl,ia,jb,kc,ld->abcd', MPO, *(sq_mat,) * 4)

def spinIceMPO():
    """MPO for square spin ice
    """
    MPO = np.zeros((2, 2, 2, 2))
    MPO[1, 1, 0, 0] = 1.
    MPO[1, 0, 1, 0] = 1.
    MPO[1, 0, 0, 1] = 1.
    MPO[0, 1, 1, 0] = 1.
    MPO[0, 1, 0, 1] = 1.
    MPO[0, 0, 1, 1] = 1.

    return MPO


Ising_ref = 0.9296953983416103  # Wolfram
Ice_ref = np.log(8 * np.sqrt(3) / 9)

idmrg = IDMRG(spinIceMPO(), kind='pf', cell_size=1)
for D in [4, 8, 12, 16, 24, 50, 100]:
    t = time()
    E = idmrg.kernel(D, two_site=True, max_iter=5000, verbosity=2)
    print(f"D {D}: {E}, {E - Ice_ref} ({time() - t} sec)")
    pickle.dump(idmrg, open("ice.pkl", "wb"),
                protocol=pickle.HIGHEST_PROTOCOL)
