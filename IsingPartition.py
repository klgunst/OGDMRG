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


# reference ≅ 0.9296950200766443  # Wolfram
# reference = np.log(2) / 2 + (2 + 33193 / 50000) / (2 * np.pi)  # Wolfram

idmrg = IDMRG(IsingMPO(), kind='pf', cell_size=1)
idmrg.kernel(D=4, two_site=True, max_iter=500, msweeps=1, verbosity=2)
idmrg.kernel(D=50, two_site=True, max_iter=10000, msweeps=1, verbosity=2)
# idmrg.kernel(D=50, two_site=False, max_iter=4000, msweeps=3, verbosity=2)
