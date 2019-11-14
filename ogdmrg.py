import numpy as np


class OGDMRG:
    def __init__(self, H=None, multipl=2):
        self.Hloc = OGDMRG.HeisenbergInteraction(multipl) if H is None else H
        self.M = 1
        self.A = 1

        self.p = self.Hloc.shape[0]
        self.HA = np.zeros((self.M, self.p, self.M, self.p))
        self.cnt = 0

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

    def kernel(self, D=16):
        from numpy.linalg import svd
        from scipy.sparse.linalg import eigsh, LinearOperator

        i = 0
        while i < 5000:
            i += 1
            self.cnt += 1
            H = LinearOperator(((self.M * self.p) ** 2,) * 2,
                               matvec=lambda x: self.Heff(x))

            w, v = eigsh(H, k=1)
            self.E = w[0] / self.cnt / 2

            A2 = v[:, 0].reshape((self.M * self.p, self.p * self.M))
            u, s, v = svd(A2)

            print(f"E: {self.E:.12f}, \tΔE: {self.E + np.log(2) - 0.25:.3g}, "
                  f"\tdiscarded: {np.linalg.norm(s[D:]):.2g}")

            self.A = u.reshape(self.M, self.p, -1)[:, :, :D]
            oldM, self.M = self.M, self.A.shape[-1]
            Mp = oldM * self.p

            A = self.A.reshape(Mp, -1)
            tH = A.T @ self.HA.reshape(Mp, Mp) @ A
            self.HA = np.kron(tH, np.eye(self.p))
            self.HA = self.HA.reshape(self.M, self.p, self.M, self.p)

            A = self.A.reshape(oldM, self.p, self.M)
            self.HA += np.einsum('aib,ajc,ikjl->bkcl', A, A, self.Hloc)

    def run(self, **kwargs):
        self.kernel(**kwargs)
        return self


if __name__ == '__main__':
    bla = OGDMRG().run(D=16)
    bla.kernel(D=44)
