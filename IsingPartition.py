from idmrg import IDMRG
import numpy as np


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


idmrg = IDMRG(IsingMPO(), cell_size=1)
idmrg.kernel(D=16, two_site=True, max_iter=1000, which='LM', msweeps=1,
             verbosity=3, rotate=False)
