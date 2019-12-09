from idmrg import IDMRG
import numpy as np
import pickle


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


reference = 0.9296953983416103  # Wolfram

idmrg = IDMRG(IsingMPO(), kind='pf', cell_size=1)
for D in [4, 8, 12, 16, 24, 50, 100]:
    E = idmrg.kernel(D, max_iter=5000, verbosity=2)
    print(f"D: {E}, {E - reference}")
    pickle.dump(idmrg, open("ising.pkl", "wb"),
                protocol=pickle.HIGHEST_PROTOCOL)
