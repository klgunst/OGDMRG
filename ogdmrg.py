import numpy as np


def four_site(NN):
    """Transforms the two site interaction to an equivalent four-site
    interaction such that we can do `two site` optimization which is actually
    four sites in a time.
    """
    pd = NN.shape[0]
    NN = NN.reshape(pd * pd, -1)
    NN2 = 0.5 * np.kron(NN, np.eye(pd ** 2)).reshape((pd ** 2,) * 4)
    NN2 += 0.5 * np.kron(np.eye(pd ** 2), NN).reshape((pd ** 2,) * 4)
    NNtemp = np.kron(np.eye(pd), NN).reshape((pd ** 3,) * 2)
    return (NN2 + np.kron(NNtemp, np.eye(pd)).reshape((pd ** 2,) * 4)) / 2


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
    Sp, Sm, Sz = S_operators(multipl)
    H = 0.5 * (np.kron(Sp, Sm) + np.kron(Sm, Sp)) + np.kron(Sz, Sz)
    return H.reshape((multipl,) * 4)


def IsingInteraction(multipl=2, J=4):
    """Returns Ising interaction between two sites.

        This is given by:
            1/2 * (S_1^+  + S_1^- + S_2^+ + S_2^-) + S_1^z S_2^z

        Interaction is given in a dense matrix:
            Σ H_{1', 2', 1, 2} |1'〉|2'〉〈1|〈2|
    """
    Sp, Sm, Sz = S_operators(multipl)
    unity = np.eye(Sp.shape[0])
    H = 0.5 * (
        np.kron(Sp, unity) + np.kron(Sm, unity) +
        np.kron(unity, Sp) + np.kron(unity, Sm)
    ) + J * np.kron(Sz, Sz)

    return H.reshape((multipl,) * 4)


class OGDMRG:
    """
    Attributes:
        NN_interaction: The Nearest neighbour interaction for the hamiltonian
        HL: The current effective Hamiltonian
        E: The current energy per site
        Etot: The current total energy of the system
    """
    def __init__(self, NN_interaction=None):
        """Initializes the OGDMRG object.

        Args:
            NN_interaction: The nearest neighbour interaction.

            If None assume Heisenberg interaction.

            This can also be set to a tensor representing the NN interaction.

            For more information how the passed NN interaction should be
            structured, see the HeisenbergInteraction function.
        """
        if NN_interaction is None:
            self.NN_interaction = HeisenbergInteraction()
        else:
            self.NN_interaction = NN_interaction

        self.HL = np.zeros((self.M, self.p, self.M, self.p))
        self.HR = np.zeros((self.p, self.M, self.p, self.M))
        self.E = 0
        self.Etot = 0

    @property
    def M(self):
        """The current bond dimension used for DMRG.

        It is equal to the last dimension of the current A tensor.
        """
        try:
            return self.A.shape[-1]
        except AttributeError:
            # No A given yet
            return 1

    @property
    def p(self):
        """The dimension of the local physical basis.
        """
        assert len(self.NN_interaction.shape) == 4
        return self.NN_interaction.shape[0]

    @property
    def A(self):
        """The current A-tensor (left canonical tensor).

        Internally it is stored as a deque of a certain length. The current
        A-tensor is the first element of the deque, previous A-tensors are the
        other elements of the deque.
        """
        return self._A_deque[0]

    @A.setter
    def A(self, A):
        from collections import deque
        # A deque is also initialized so to keep track of previous A's
        try:
            self._A_deque.appendleft(A)
        except AttributeError:
            # The deque is not initialized yet
            self._A_deque = deque(maxlen=3)
            self._A_deque.appendleft(A)

    @property
    def A_diff(self):
        """The difference between the current A and the one two iterations ago.
        """
        A1 = self._A_deque[0].reshape(-1, self._A_deque[0].shape[-1])
        A2 = self._A_deque[-1].reshape(-1, self._A_deque[-1].shape[-1])
        XX = np.diag(self._s_deque[0]) @ A1.T @ A2 @ np.diag(self._s_deque[-1])
        s_diag = np.diag(self._s_deque[0] * self._s_deque[-1])
        return np.linalg.norm(XX - s_diag)

    @property
    def B(self):
        """The current B-tensor (right canonical tensor).

        Internally it is stored as a deque of a certain length. The current
        B-tensor is the first element of the deque, previous B-tensors are the
        other elements of the deque.
        """
        return self._B_deque[0]

    @B.setter
    def B(self, B):
        from collections import deque
        # B deque is also initialized so to keep track of previous B's
        try:
            self._B_deque.appendleft(B)
        except AttributeError:
            # The deque is not initialized yet
            self._B_deque = deque(maxlen=3)
            self._B_deque.appendleft(B)

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
            self._s_deque = deque(maxlen=3)
            self._s_deque.appendleft(s)

    def Heff(self, x):
        """Executing the Effective Hamiltonian on the two-site object `x`.

        The Effective Hamiltonian exists out of:
            * Interactions between left environment and left site
            * Interactions between right environment and right site
            * Interactions between the left and right site

        Returns H_A * x.
        """
        x = x.reshape(self.M * self.p, -1)
        # Interactions between left environment and leftmost site
        result = (self.HL.reshape(self.M * self.p, -1) @ x).ravel()

        # Interactions between right environment and rightmost site
        x = x.reshape(-1, self.M * self.p)
        result += (x @ self.HR.reshape(self.p * self.M, -1).T).ravel()

        # Interactions between current sites
        for i in range(self.nrsites - 1):
            newshape = (
                self.M * self.p ** i,
                self.p, self.p,
                self.p ** (self.nrsites - i - 2) * self.M
            )
            result = result.reshape(newshape)
            x = x.reshape(newshape)
            result += np.einsum('xyij,lijr->lxyr', self.NN_interaction, x)
        return result.ravel()

    def renormalize_basis(self, A2, D, tol=1e-10):
        """Renormalize the basis.
        """
        from numpy.linalg import svd, qr
        hsites = self.nrsites // 2
        u, s, v = svd(A2.reshape((self.M * self.p ** hsites,) * 2))
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
        v = v[:D, :]

        # Trying to fix the guage
        for begin, end in zip(new_sval[:lastmultiplet], new_sval[1:]):
            Q, U = qr(u[:, begin:end].T)
            U = U.T

            # First nonzero element in each column
            fnz = U[np.argmin(np.isclose(U, 0), axis=0), range(U.shape[1])]
            u[:, begin:end] = U * (1 - 2 * (fnz < 0))[None, :]
            Q = Q * (1 - 2 * (fnz < 0))[None, :]

            v[begin:end, :] = Q.T @ v[begin:end, :]

        self.B = v.reshape(D, *(self.p,) * hsites, self.M)
        self.A = u.reshape(self.M, *(self.p,) * hsites, D)
        self.s = s[:D]

        return s[D:] @ s[D:]

    def update_Heff(self):
        """Update the effective Hamiltonian for the new renormalized basis.
        """
        # Left environment update
        oldM = self.A.shape[0]
        hsites = self.nrsites // 2
        ptot = self.p ** hsites
        Mp = oldM * self.p

        A = self.A.reshape(Mp, -1)
        tH = (self.HL.reshape(Mp, Mp) @ A).reshape(oldM * ptot, self.M)
        tH = self.A.reshape(oldM * ptot, self.M).T @ tH
        for i in range(hsites - 1):
            newshape = (
                oldM * self.p ** i,
                self.p, self.p,
                self.p ** (hsites - i - 2) * self.M
            )
            tA = np.einsum('xyij,lijr->lxyr', self.NN_interaction,
                           self.A.reshape(newshape))
            tH += self.A.reshape(-1, self.M).T @ tA.reshape(-1, self.M)
        self.HL = np.kron(tH, np.eye(self.p))
        self.HL = self.HL.reshape(self.M, self.p, self.M, self.p)

        A = self.A.reshape(-1, self.p * self.M)
        X = (A.T @ A).reshape(self.p, self.M, self.p, self.M)
        self.HL += np.einsum('ibjc,ikjl->bkcl', X, self.NN_interaction)

        # Right environment update
        B = self.B.reshape(-1, Mp)
        tH = (B @ self.HR.reshape(Mp, Mp).T).reshape(self.M, oldM * ptot)
        tH = tH @ self.B.reshape(self.M, oldM * ptot).T
        for i in range(hsites - 1):
            newshape = (
                self.M * self.p ** i,
                self.p, self.p,
                self.p ** (hsites - i - 2) * oldM
            )
            tB = np.einsum('xyij,lijr->lxyr', self.NN_interaction,
                           self.B.reshape(newshape))
            tH += self.B.reshape(self.M, -1) @ tB.reshape(self.M, -1).T
        self.HR = np.kron(np.eye(self.p), tH)
        self.HR = self.HR.reshape(self.p, self.M, self.p, self.M)

        B = self.B.reshape(self.M * self.p, -1)
        X = (B @ B.T).reshape(self.M, self.p, self.M, self.p)
        self.HR += np.einsum('bicj,kilj->kblc', X, self.NN_interaction)

    def kernel(self, D=16, sites=2, max_iter=100, tol=1e-6, verbosity=2):
        """Executing of the DMRG algorithm.

        Args:
            D: The bond dimension to use for DMRG. The algorithm can choose
            a bond dimension larger than the one specified to avoid truncating
            between renormalized states degenerate in their singular values.

            sites: The number of sites to add each step

            max_iter: The maximal iterations to use in the DMRG algorithm.

            tol: The Energy tolerance on which to abort the calculation

            verbosity: 0: Don't print anything.
                       1: Print results for the optimization.
                       2: Print intermediate result at every even chain length.
        """
        from scipy.sparse.linalg import eigsh, LinearOperator
        if sites % 2 != 0:
            raise ValueError('Number of sites needs to be even')

        self.nrsites = sites

        for i in range(max_iter):
            H = LinearOperator(
                (self.M * self.M * (self.p ** self.nrsites),) * 2,
                matvec=lambda x: self.Heff(x)
            )
            # Diagonalize
            w, v = eigsh(H, k=1, which='SA')
            # Renormalize the basis
            trunc = self.renormalize_basis(v[:, 0], D)
            # Update the effective Hamiltonian
            self.update_Heff()

            # Energy difference between this and the previous even-length chain
            E = (w[0] - self.Etot) / self.nrsites
            ΔE, self.E, self.Etot = self.E - E, E, w[0]

            if verbosity >= 2:
                try:
                    print(f"it {i}:\tM: {self.M},\tE: {self.E:.12f},\t"
                          f"ΔE: {ΔE:.3g},\ttrunc: {trunc:.3g},\t"
                          f"A_diff {self.A_diff}")
                except ValueError:
                    print(f"it {i}:\tM: {self.M},\tE: {self.E:.12f},\t"
                          f"ΔE: {ΔE:.3g},\ttrunc: {trunc:.3g}")

            if tol is not None and abs(ΔE) < tol:
                break
        if verbosity >= 1:
            print(f"its: {i},\tM: {self.M},\tE: {self.E:.12f},\t"
                  f"ΔE: {ΔE:.3g},\ttrunc: {trunc:.3g}")
            if i == max_iter:
                print("Convergence not reached.")

        return self.E


class VUMPS:
    """
    Attributes:
        NN_interaction: The Nearest neighbour interaction for the hamiltonian
    """
    def __init__(self, NN_interaction=None):
        """Initializes the VUMPS object.

        Args:
            NN_interaction: The nearest neighbour interaction.

            If None assume Heisenberg interaction.

            This can also be set to a tensor representing the NN interaction.

            For more information how the passed NN interaction should be
            structured, see the HeisenbergInteraction function.
        """
        if NN_interaction is None:
            self.NN_interaction = HeisenbergInteraction()
        else:
            self.NN_interaction = NN_interaction

    @property
    def p(self):
        """The dimension of the local physical basis.
        """
        assert len(self.NN_interaction.shape) == 4
        return self.NN_interaction.shape[0]

    def HAc(self, x):
        # left Heff
        result = (self.Hl @ x.reshape(self.M, -1)).ravel()
        # right Heff
        result += (x.reshape(-1, self.M) @ self.Hr.T).ravel()

        # first onsite
        LL = self.Al.reshape(-1, self.M) @ x.reshape(self.M, -1)
        LL = np.einsum('lijr,xyij->lxyr',
                       LL.reshape(self.M, self.p, self.p, self.M),
                       self.NN_interaction
                       )
        result += (self.Al.reshape(self.M * self.p, -1).T @
                   LL.reshape(self.M * self.p, -1)).ravel()

        # second onsite
        RR = x.reshape(-1, self.M) @ self.Ar.reshape(self.M, -1)
        RR = np.einsum('lijr,xyij->lxyr',
                       RR.reshape(self.M, self.p, self.p, self.M),
                       self.NN_interaction
                       )
        result += (RR.reshape(-1, self.M * self.p) @
                   self.Ar.reshape(-1, self.M * self.p).T).ravel()
        return result

    def Hc(self, x):
        x = x.reshape(self.M, self.M)
        # left Heff
        result = (self.Hl @ x).ravel()
        # right Heff
        result += (x @ self.Hr.T).ravel()

        # On site
        C1 = self.Al.reshape(-1, self.M) @ x @ self.Ar.reshape(self.M, -1)
        C2 = np.einsum('lijr,xyij->lxyr',
                       C1.reshape(self.M, self.p, self.p, self.M),
                       self.NN_interaction
                       )
        C3 = C2.reshape(-1, self.p * self.M) @ \
            self.Ar.reshape(-1, self.p * self.M).T
        result += (self.Al.reshape(self.M * self.p, -1).T @
                   C3.reshape(self.p * self.M, -1)).ravel()
        return result

    def MakeHeffLeft(self):
        from scipy.sparse.linalg import bicgstab, LinearOperator

        All = self.Al.reshape(-1, self.M) @ self.Al.reshape(self.M, -1)
        Hall = np.einsum('lijr,xyij->lxyr',
                         All.reshape(self.M, self.p, self.p, self.M),
                         self.NN_interaction
                         )
        hl = All.reshape(-1, self.M).T @ Hall.reshape(-1, self.M)
        e = np.trace(self.c.T @ hl @ self.c) * np.eye(self.M)

        def Afunc(x):
            x = x.reshape(self.M, self.M)
            res = (np.trace(self.c.T @ x @ self.c) * np.eye(self.M)).ravel()
            res += x.ravel()
            self.Al = self.Al.reshape(self.M, self.p, self.M)
            res -= np.einsum('ij,jpa,ipb->ba', x, self.Al, self.Al).ravel()
            return res

        A = LinearOperator((self.M * self.M,) * 2, matvec=Afunc)
        r, info = bicgstab(A, (hl - e).ravel())
        print(info)
        return r.reshape(self.M, self.M)

    def MakeHeffRight(self):
        from scipy.sparse.linalg import bicgstab, LinearOperator

        Arr = self.Ar.reshape(-1, self.M) @ self.Ar.reshape(self.M, -1)
        Harr = np.einsum('lijr,xyij->lxyr',
                         Arr.reshape(self.M, self.p, self.p, self.M),
                         self.NN_interaction
                         )
        hr = Arr.reshape(self.M, -1) @ Harr.reshape(self.M, -1).T
        e = np.trace(self.c @ hr @ self.c.T) * np.eye(self.M)

        def Afunc(x):
            x = x.reshape(self.M, self.M)
            res = (np.trace(self.c @ x @ self.c.T) * np.eye(self.M)).ravel()
            res += x.ravel()
            self.Ar = self.Ar.reshape(self.M, self.p, self.M)
            res -= np.einsum('ij,apj,bpi->ba', x, self.Ar, self.Ar).ravel()
            return res

        A = LinearOperator((self.M * self.M,) * 2, matvec=Afunc)
        r, info = bicgstab(A, (hr - e).ravel())
        print(info)
        return r.reshape(self.M, self.M)

    def kernel(self, D=16, max_iter=100, tol=1e-6, verbosity=2):
        from scipy.sparse.linalg import eigsh, LinearOperator
        from scipy.linalg import polar
        from numpy.random import rand
        self.M = D

        Ac = rand(self.M, self.p, self.M)

        ual, pal = polar(Ac.reshape(-1, self.M), side='right')
        self.c = pal

        uar, par = polar(Ac.reshape(self.M, -1), side='left')
        ucl, pcl = polar(self.c, side='right')
        ucr, pcr = polar(self.c, side='left')
        self.Al, self.Ar = ual @ ucl.T, ucr.T @ uar

        for i in range(max_iter):
            self.Hl = self.MakeHeffLeft()
            self.Hr = self.MakeHeffRight()

            HAc = LinearOperator(
                (self.M * self.M * self.p,) * 2,
                matvec=lambda x: self.HAc(x)
            )
            Hc = LinearOperator(
                (self.M * self.M,) * 2, matvec=lambda x: self.Hc(x)
            )
            # Solve the two eigenvalues problem for Ac and c
            w1, v1 = eigsh(HAc, k=1, which='SA')
            w2, v2 = eigsh(Hc, k=1, which='SA')
            print(f'HAc: {w1[0] / 2}, Hc: {w2[0]}')

            Ac = v1[:, 0].reshape(self.M, self.p, self.M)
            self.c = v2[:, 0].reshape(self.M, self.M)

            ual, pal = polar(Ac.reshape(-1, self.M), side='right')
            uar, par = polar(Ac.reshape(self.M, -1), side='left')
            ucl, pcl = polar(self.c, side='right')
            ucr, pcr = polar(self.c, side='left')
            self.Al, self.Ar = ual @ ucl.T, ucr.T @ uar


if __name__ == '__main__':
    from sys import argv
    if len(argv) > 1:
        D = [int(d) for d in argv[1:]]
    else:
        D = [16]

    ogdmrg = OGDMRG(NN_interaction=IsingInteraction())
    vumps = VUMPS(NN_interaction=IsingInteraction())
    vumps = VUMPS(NN_interaction=four_site(HeisenbergInteraction()))
    for d in D:
        vumps.kernel(D=d, max_iter=1000)
        exit(0)
        ogdmrg.kernel(D=d, max_iter=1000, sites=2, tol=None)
