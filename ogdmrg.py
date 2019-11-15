import numpy as np


class OGDMRG:
    def __init__(self, H=None, multipl=2):
        self.Hloc = OGDMRG.HeisenbergInteraction(multipl) if H is None else H
        self.A = np.ones(1)

        self.p = self.Hloc.shape[0]
        self.HA = np.zeros((self.M, self.p, self.M, self.p))
        self.cnt = 0

    @property
    def M(self):
        return self.A.shape[-1]

    def S_operators(multipl=2):
        j = (multipl - 1) / 2
        # magnetic quantum number for eacht basis state in the local basis
        m = np.arange(multipl) - j

        Sz = np.diag(m)
        Sp = np.zeros(Sz.shape)
        Sp[range(1, multipl), range(0, multipl - 1)] = \
            np.sqrt((j - m) * (j + m + 1))[:-1]
        return Sp, Sp.T, Sz

    def HeisenbergInteraction(multipl=2):
        """Returns interaction between two sites.

        Interaction is given in a dense matrix with indices:
            bra_1, bra_2, ket_1, ket_2
        """
        Sp, Sm, Sz = OGDMRG.S_operators(multipl)
        H = 0.5 * (np.kron(Sp, Sm) + np.kron(Sm, Sp)) + np.kron(Sz, Sz)
        return H.reshape((multipl,) * 4)

    def Heff(self, x):
        # shape: (l i) (j j)
        x = x.reshape(self.M * self.p, -1)
        result = self.HA.reshape(self.M * self.p, -1) @ x
        result += x @ self.HA.reshape(self.M * self.p, -1).T

        # shape: l i r j
        x = x.reshape(self.M, self.p, self.M, self.p)
        result = result.reshape(self.M, self.p, self.M, self.p)
        result += np.einsum('xyij,lirj->lxry', self.Hloc, x)
        return result.ravel()

    def renormalize_basis(self, A2, D, tol=1e-10):
        """Renormalize the basis.
        """
        from numpy.linalg import svd

        u, s, v = svd(A2)
        svd_diff = (s[:-1] - s[1:]) / s[:-1]
        # Array with Trues every time this singular value is different than the
        # previous one (up to a tolerance)
        new_sval = np.concatenate(([True], svd_diff > tol))

        # Truncating renormalized basis
        #
        # The kept basis states can be larger than te one specified by the
        # user, we just try not to cut up degenerate singular values.
        for D, nsv in enumerate(new_sval[D:], start=D):
            if nsv:
                break
        self.A = u.reshape(self.M, self.p, -1)[:, :, :D]

    def update_Heff(self):
        """Update the effective Hamiltonian for the new renormalized basis
        """
        oldM = self.A.shape[0]
        Mp = oldM * self.p

        A = self.A.reshape(Mp, -1)
        tH = A.T @ self.HA.reshape(Mp, Mp) @ A
        self.HA = np.kron(tH, np.eye(self.p))
        self.HA = self.HA.reshape(self.M, self.p, self.M, self.p)

        A = self.A.reshape(oldM, self.p, self.M)
        self.HA += np.einsum('aib,ajc,ikjl->bkcl', A, A, self.Hloc)

    def kernel(self, D=16):
        from scipy.sparse.linalg import eigsh, LinearOperator

        i = 0
        while i < 100:
            i += 1
            self.cnt += 1
            H = LinearOperator(((self.M * self.p) ** 2,) * 2,
                               matvec=lambda x: self.Heff(x))

            w, v = eigsh(H, k=1)
            self.E = w[0] / self.cnt / 2
            A2 = v[:, 0].reshape((self.M * self.p, self.p * self.M))

            print(f"E: {self.E:.12f}, \tΔE: {self.E + np.log(2) - 0.25:.3g}")

            self.renormalize_basis(A2, D)
            self.update_Heff()

    def run(self, **kwargs):
        self.kernel(**kwargs)
        return self


if __name__ == '__main__':
    bla = OGDMRG().run(D=16)
