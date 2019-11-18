import numpy as np


class OGDMRG:
    """
    Attributes:
        NN_interaction: The Nearest neighbour interaction for the hamiltonian
        HA: The current effective Hamiltonian
        E: The current energy per site
        Etot: The current total energy of the system
    """
    def __init__(self, NN_interaction='Heisenberg', multipl=2):
        """Initializes the OGDMRG object.

        Args:
            NN_interaction: The nearest neighbour interaction.

            This can be set to "Heisenberg" or "Ising" for Heisenberg or Ising
            interactions respectively.

            This can also be set to a tensor representing the NN interaction.

            For more information how the passed NN interaction should be
            structured, see the HeisenbergInteraction function.

            multipl: If NN_interaction is not "Heisenberg" or "Ising", this
            argument is ignored, else this argument specifies the multiplicity
            of each spin for the given interaction.

            E.g. For `multipl = 2`, we work with spin-1/2.
                 For `multipl = 3`, we work with spin-1.
        """
        if NN_interaction == 'Heisenberg':
            self.NN_interaction = OGDMRG.HeisenbergInteraction(multipl)
        elif NN_interaction == 'Ising':
            self.NN_interaction = OGDMRG.IsingInteraction(multipl)
        else:
            self.NN_interaction = NN_interaction

        self.A = np.ones((1, 1))
        self.HA = np.zeros((self.M, self.p, self.M, self.p))
        self.E = 0
        self.Etot = 0

    @property
    def M(self):
        """The current bond dimension used for DMRG.

        It is equal to the last dimension of the current A tensor.
        """
        return self.A.shape[-1]

    @property
    def p(self):
        """The dimension of the local physical basis.
        """
        assert len(self.NN_interaction.shape) == 4
        return self.NN_interaction.shape[0]

    @property
    def A(self):
        """The current A-tensor.

        Internally it is stored as a deque of a certain length. The current
        A-tensor is the first element of the deque, previous A-tensors are the
        other elements of the deque.
        """
        return self._A_deque[0]

    @property
    def s(self):
        return self._s_deque[0]

    @s.setter
    def s(self, s):
        from collections import deque
        # A deque is also initialized so to keep track of previous A's
        try:
            self._s_deque.appendleft(s)
        except AttributeError:
            # The deque is not initialized yet
            self._s_deque = deque(s, maxlen=3)

    @A.setter
    def A(self, A):
        from collections import deque
        # A deque is also initialized so to keep track of previous A's
        try:
            self._A_deque.appendleft(A)
        except AttributeError:
            # The deque is not initialized yet
            self._A_deque = deque(A, maxlen=3)

    @property
    def A_diff(self):
        """The difference between the current A and the one two iterations ago.
        """
        A1 = self._A_deque[0].reshape(-1, self._A_deque[0].shape[-1]) * \
            self._s_deque[0]
        A2 = self._A_deque[-1].reshape(-1, self._A_deque[-1].shape[-1]) * \
            self._s_deque[-1]
        # print(self._s_deque[0] - self._s_deque[-1])
        # print(self._s_deque[0])
        # print(A1.T @ A2 - np.diag(self._s_deque[0] * self._s_deque[-1]))
        s_diag = np.diag(self._s_deque[0] * self._s_deque[-1])
        return np.linalg.norm(A1.T @ A2 - s_diag)

    def S_operators(multipl=2):
        """Returns the S+, S-, and Sz operators in for a spin.
The operators are represented in the Sz basis: (-j, -j + 1, ..., j)

        Args:
            multipl: defines which multiplicity the total spin of the site has.
            Thus specifies j as `j = (multipl - 1) / 2`
        """
        j = (multipl - 1) / 2
        # magnetic quantum number for eacht basis state in the local basis
        m = np.arange(multipl) - j

        Sz = np.diag(m)
        Sp = np.zeros(Sz.shape)
        Sp[range(1, multipl), range(0, multipl - 1)] = \
            np.sqrt((j - m) * (j + m + 1))[:-1]
        return Sp, Sp.T, Sz

    def HeisenbergInteraction(multipl=2):
        """Returns Heisenberg interaction between two sites.

        This is given by:
            1/2 * (S_1^+ S_2^- + S_1^- S_2^+) + S_1^z S_2^z

        Interaction is given in a dense matrix:
            Σ H_{1', 2', 1, 2} |1'〉|2'〉〈1|〈2|
        """
        Sp, Sm, Sz = OGDMRG.S_operators(multipl)
        H = 0.5 * (np.kron(Sp, Sm) + np.kron(Sm, Sp)) + np.kron(Sz, Sz)
        return H.reshape((multipl,) * 4)

    def IsingInteraction(multipl=2):
        """Returns Ising interaction between two sites.

        This is given by:
            1/2 * (S_1^+  + S_1^- + S_2^+ + S_2^-) + S_1^z S_2^z

        Interaction is given in a dense matrix:
            Σ H_{1', 2', 1, 2} |1'〉|2'〉〈1|〈2|
        """
        Sp, Sm, Sz = OGDMRG.S_operators(multipl)
        unity = np.eye(Sp.shape[0])
        H = 0.5 * (
            np.kron(Sp, unity) + np.kron(Sm, unity) +
            np.kron(unity, Sp) + np.kron(unity, Sm)
        ) + 3.9 * np.kron(Sz, Sz)

        return H.reshape((multipl,) * 4)

    def Heff(self, x):
        """Executing the Effective Hamiltonian on the two-site object `x`.

        The Effective Hamiltonian exists out of:
            * Interactions between left environment and left site
            * Interactions between right environment and right site
            * Interactions between the left and right site

        Returns H_A * x.
        """
        x = x.reshape(self.M * self.p, -1)
        # Interactions between left environment and left site
        result = self.HA.reshape(self.M * self.p, -1) @ x
        # Interactions between right environment and right site
        result += x @ self.HA.reshape(self.M * self.p, -1).T
        # Interactions between left and right site
        x = x.reshape(self.M, self.p, self.M, self.p)
        result = result.reshape(self.M, self.p, self.M, self.p)
        result += np.einsum('xyij,lirj->lxry', self.NN_interaction, x)

        return result.ravel()

    def renormalize_basis(self, A2, D, tol=1e-10):
        """Renormalize the basis.
        """
        from numpy.linalg import svd, qr

        u, s, v = svd(A2.reshape((self.M * self.p, self.M * self.p)))
        svd_diff = (s[:-1] - s[1:]) / s[:-1]

        # Array with Trues every time this singular value is different than the
        # previous one (up to a tolerance)
        new_sval = np.concatenate(
            ([0], np.where(svd_diff > tol)[0] + 1, [len(s)])
        )
        # Truncating renormalized basis
        #
        # The kept basis states can be larger than te one specified by the
        # user, we just try not to cut up degenerate singular values.
        lastmultiplet = -1 if new_sval[-1] < D else np.argmax(new_sval >= D)

        # If adding the full multiplet makes the bond dimension too large,
        # discard the last multiplet
        if new_sval[lastmultiplet] > 1.2 * D:
            lastmultiplet -= 1
        D = new_sval[lastmultiplet]
        u = u[:, :D]

        # Trying to fix the guage
        for begin, end in zip(new_sval[:lastmultiplet], new_sval[1:]):
            U = qr(u[:, begin:end].T, mode='r').T

            # First nonzero element in each column
            fnz = U[np.argmin(np.isclose(U, 0), axis=0), range(U.shape[1])]
            U = U * (1 - 2 * (fnz < 0))[None, :]
            u[:, begin:end] = U

        self.A = u.reshape((self.M, self.p, -1))
        self.s = s[:D]
        return s[D:] @ s[D:]

    def update_Heff(self):
        """Update the effective Hamiltonian for the new renormalized basis.
        """
        Mp = self.A.shape[0] * self.A.shape[1]

        A = self.A.reshape(Mp, -1)
        tH = A.T @ self.HA.reshape(Mp, Mp) @ A
        self.HA = np.kron(tH, np.eye(self.p))
        self.HA = self.HA.reshape(self.M, self.p, self.M, self.p)

        A = self.A.reshape(-1, self.p * self.M)
        B = (A.T @ A).reshape(self.p, self.M, self.p, self.M)
        self.HA += np.einsum('ibjc,ikjl->bkcl', B, self.NN_interaction)

    def kernel(self, D=16, max_iter=100, verbosity=2):
        """Executing of the DMRG algorithm.

        Args:
            D: The bond dimension to use for DMRG. The algorithm can choose
            a bond dimension larger than the one specified to avoid truncating
            between renormalized states degenerate in their singular values.

            max_iter: The maximal iterations to use in the DMRG algorithm.

            verbosity: 0: Don't print anything.
                       1: Print results for the optimization.
                       2: Print intermediate result at every even chain length.
        """
        from scipy.sparse.linalg import eigsh, LinearOperator

        for i in range(max_iter):
            H = LinearOperator(((self.M * self.p) ** 2,) * 2,
                               matvec=lambda x: self.Heff(x))
            # Diagonalize
            w, v = eigsh(H, k=1, which='SA')
            # Renormalize the basis
            trunc = self.renormalize_basis(v[:, 0], D)
            # Update the effective Hamiltonian
            self.update_Heff()

            # Energy difference between this and the previous even-length chain
            E = (w[0] - self.Etot) / 2
            ΔE, self.E, self.Etot = self.E - E, E, w[0]

            if verbosity >= 2 and i > 10:
                print(f"it {i}:\tM: {self.M},\tE: {self.E:.12f},\t"
                      f"ΔE: {ΔE:.3g},\ttrunc: {trunc:.3g},\t"
                      f"A_diff {self.A_diff}"
                      )
            else:
                print(f"it {i}:\tM: {self.M},\tE: {self.E:.12f},\t"
                      f"ΔE: {ΔE:.3g},\ttrunc: {trunc:.3g},\t"
                      )
        if verbosity >= 1:
            print(f"M: {self.M},\tE: {self.E:.12f},\t"
                  f"ΔE: {ΔE:.3g},\ttrunc: {trunc:.3g}")

        return self.E


if __name__ == '__main__':
    from sys import argv
    if len(argv) > 1:
        D = [int(d) for d in argv[1:]]
    else:
        D = [16]

    # ogdmrg = OGDMRG(NN_interaction="Ising")
    ogdmrg = OGDMRG()
    for d in D:
        ogdmrg.kernel(D=d, max_iter=1000)
