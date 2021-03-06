import numpy as np
from numpy.random import rand
from numpy.linalg import norm
from scipy.linalg import polar, svd
from scipy.sparse.linalg import bicgstab, eigs, eigsh, LinearOperator
from numpy import einsum
import unittest
from numpy.testing import assert_allclose

from pyscf.lib import davidson


def transfer_eig(A, B):
    assert len(A.shape)
    assert A.size == B.size
    assert A.shape[0] == A.shape[-1]
    D = A.shape[0]

    def Transfer(x):
        xA = x.reshape(D, D) @ A.reshape(D, -1)
        return (B.reshape(-1, D).conj().T @ xA.reshape(-1, D)).ravel()

    LO = LinearOperator((D ** 2,) * 2, matvec=Transfer)
    w, v = eigs(LO, k=1, which='LM')
    return w[0], v[:, 0]


def canonicalize(Ar, c_in=None, tol=1e-14, dtype=np.float64):
    assert len(Ar.shape) == 3
    M = Ar.shape[-1]
    A = Ar.copy()
    c = np.eye(M)

    diff = 1
    iterations = 1
    while diff > tol:
        A = A.reshape(M, -1, M)
        _, v = transfer_eig(A, A)

        U, s, Vh = svd(v.reshape(M, M))
        sqrt_eps = np.sqrt(np.finfo(dtype).eps)
        s = np.array([max(np.sqrt(st), sqrt_eps) for st in s])
        s = s / norm(s)
        c1 = np.diag(s) @ Vh
        c1_inv = Vh.conj().T @ np.diag(1 / s)
        A = c1 @ A.reshape(M, -1)
        A = A.reshape(-1, M) @ c1_inv
        A = A / norm(A) * np.sqrt(M)

        c = c1 @ c
        c = c / norm(c)
        diff = norm(v.reshape(M, M) - np.eye(M) * v[0])
    assert np.isclose(
        norm(A.reshape(-1, M).conj().T @ A.reshape(-1, M) - np.eye(M)), 0
    )
    return A, c / norm(c), (iterations, diff)


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


def H_2site(NN_interaction, AA):
    """Executes the nearest neighbour interaction on a two-site tensor
    """
    assert len(AA.shape) == 4
    ppdim = AA.shape[1] * AA.shape[2]
    newshape = (AA.shape[0], ppdim, AA.shape[-1])
    AA = AA.reshape(newshape)
    NN = NN_interaction.reshape(ppdim, ppdim)
    return einsum('ij,ajb->aib', NN, AA)


class OGDMRG:
    """
    Attributes:
        NN_interaction: The Nearest neighbour interaction for the hamiltonian
        HL: The current effective Hamiltonian
        E: The current energy per site
        Etot: The current total energy of the system
    """
    def __init__(self, NN_interaction=None, compare_back=1):
        """Initializes the OGDMRG object.

        Args:
            NN_interaction: The nearest neighbour interaction.
            compare_back: With how many Al's back the current Al should be
            guage fixed.

            If None assume Heisenberg interaction.

            This can also be set to a tensor representing the NN interaction.

            For more information how the passed NN interaction should be
            structured, see the HeisenbergInteraction function.
        """
        if NN_interaction is None:
            self.NN_interaction = HeisenbergInteraction()
        else:
            self.NN_interaction = NN_interaction
        self.compare_back = compare_back

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
            self._A_deque = deque(maxlen=self.compare_back + 1)
            self._A_deque.appendleft(A)

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
            self._B_deque = deque(maxlen=self.compare_back + 1)
            self._B_deque.appendleft(B)

    @property
    def c(self):
        return self._c_deque[0]

    @c.setter
    def c(self, c):
        from collections import deque
        # A deque is also initialized so to keep track of previous A's
        try:
            self._c_deque.appendleft(c)
        except AttributeError:
            # The deque is not initialized yet
            self._c_deque = deque(maxlen=self.compare_back + 1)
            self._c_deque.appendleft(c)

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
            result += H_2site(self.NN_interaction, x).reshape(newshape)
        return result.ravel()

    def Heffdiagonal(self):
        """Returns diagonal for the Heff."""
        diag = np.zeros(self.M * self.M * (self.p ** self.nrsites))
        diag = diag.reshape(self.M * self.p, -1)
        diag += self.HL.reshape(self.M * self.p, -1).diagonal()[:, None]

        diag = diag.reshape(-1, self.M * self.p)
        diag += self.HR.reshape(-1, self.M * self.p).diagonal()[None, :]

        for i in range(self.nrsites - 1):
            newshape = (
                self.M * self.p ** i,
                self.p * self.p,
                self.p ** (self.nrsites - i - 2) * self.M
            )
            diag = diag.reshape(newshape)
            diag += self.NN_interaction.reshape(
                self.p * self.p, -1).diagonal()[None, :, None]
        return diag.ravel()

    def renormalize_basis(self, A2, D, tol=1e-13):
        """Renormalize the basis.
        """
        hsites = self.nrsites // 2
        u, s, v = svd(A2.reshape((self.M * self.p ** hsites,) * 2))
        svd_diff = (s[:-1] - s[1:])

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

        # Dimension of full multiplet
        Df = new_sval[lastmultiplet]
        # Make sure bond dimension does not explode
        D = D if Df > 20 + D else Df

        # fixing the guage
        # Only if the dimensions were constant during this and previous step
        try:
            Acomp = self._A_deque[self.compare_back - 1]
            Bcomp = self._B_deque[self.compare_back - 1]
            # New bond dimension needed is same as the previous one
            doGuagefix = D == Acomp.shape[-1] and self.M == Acomp.shape[0]
        except (AttributeError, IndexError):
            doGuagefix = False

        if doGuagefix:
            u = u[:, :D]
            v = v[:D, :]
            AAx = u.conj().reshape(-1, D).T @ Acomp.reshape(-1, D)
            u2, s2, v2 = svd(AAx, full_matrices=False)
            Q, uni1 = u2 @ v2, np.max(abs(s2 - 1))
            u = u @ Q

            BBx = Bcomp.reshape(D, -1) @ v.reshape(D, -1).conj().T
            u2, s2, v2 = svd(BBx, full_matrices=False)
            P, uni2 = u2 @ v2, np.max(abs(s2 - 1))
            v = P @ v

            c = Q.conj().T @ np.diag(s[:D] / norm(s[:D])) @ P.conj().T
            u = u.reshape(self.M, -1, D)
            v = v.reshape(D, -1, self.M)
            info = {
                'Q_error': uni1,
                'P_error': uni2,
                'Al TF': 1 - abs(transfer_eig(u, Acomp)[0]),
                'Ar TF': 1 - abs(transfer_eig(v, Bcomp)[0]),
                'ArAl TF': 1 - abs(transfer_eig(u, v)[0])
            }
            self.c = c
        else:
            u = u[:, :D]
            v = v[:D, :]
            self.c = np.diag(s[:D])
            info = None

        self.B = v.reshape(D, *(self.p,) * hsites, self.M)
        self.A = u.reshape(self.M, *(self.p,) * hsites, D)
        return s[D:] @ s[D:], info

    def update_Heff(self):
        """Update the both left and right effective Hamiltonian for the
        new renormalized basis.
        """
        self.HL = self.uHeff(self.A, self.NN_interaction, self.HL)
        self.HR = self.uHeff(
            self.B.conj().T, self.NN_interaction.conj().T, self.HR.T
        ).T

    def uHeff(self, A, NN, Hc):
        """Update the effective Hamiltonian for the new renormalized basis
        """
        oldM = self.A.shape[0]
        hsites = self.nrsites // 2
        ptot = self.p ** hsites
        Mp = oldM * self.p

        A = A.reshape(Mp, -1)
        tH = (Hc.reshape(Mp, Mp) @ A).reshape(oldM * ptot, self.M)
        tH = A.reshape(oldM * ptot, self.M).conj().T @ tH
        for i in range(hsites - 1):
            newshape = (
                oldM * self.p ** i,
                self.p, self.p,
                self.p ** (hsites - i - 2) * self.M
            )
            tA = H_2site(NN, A.reshape(newshape))
            tH += A.reshape(-1, self.M).conj().T @ tA.reshape(-1, self.M)
        Hc = np.kron(tH, np.eye(self.p))
        Hc = Hc.reshape(self.M, self.p, self.M, self.p)

        A = A.reshape(-1, self.p * self.M)
        X = (A.conj().T @ A).reshape(self.p, self.M, self.p, self.M)
        Hc += einsum('ibjc,ikjl->bkcl', X, NN)
        return Hc

    def kernel(self, D=16, sites=2, max_iter=100, tol=1e-15, verbosity=2):
        """Executing of the DMRG algorithm.

        Args:
            D: The bond dimension to use for DMRG. The algorithm can choose
            a bond dimension larger than the one specified to avoid truncating
            between renormalized states degenerate in their singular values.

            sites: The number of sites to add each step

            max_iter: The maximal iterations to use in the DMRG algorithm.

            tol: The tolerance on which to abort the calculation

            verbosity: 0: Don't print anything.
                       1: Print results for the optimization.
                       2: Print intermediate result at every even chain length.
        """
        if sites % 2 != 0:
            raise ValueError('Number of sites needs to be even')

        self.nrsites = sites

        AA = None
        for i in range(max_iter):
            # Diagonalize
            size = self.M * self.M * (self.p ** self.nrsites)
            if AA is None or AA.size != size:
                AA = rand(size)
                AA /= norm(AA)
            w, v = davidson(self.Heff, x0=AA, precond=self.Heffdiagonal())

            AA = v
            # Renormalize the basis
            E = (w - self.Etot) / self.nrsites
            ΔE, self.E, self.Etot = self.E - E, E, w

            trunc, info = self.renormalize_basis(AA, D)
            if info is not None and verbosity >= 3:
                print(info)

            error = info['ArAl TF'] if info is not None else None

            # Update the effective Hamiltonian
            self.update_Heff()

            # Energy difference between this and the previous even-length chain
            if verbosity >= 2:
                print(f"it {i}:\tM: {self.M},\tE: {self.E:.12f},\t"
                      f"ΔE: {ΔE:.3g},\ttrunc: {trunc:.3g}", end='')
                if error is None:
                    print()
                else:
                    print(f',\tError: {error}')
            try:
                if i > 10 and error < tol:
                    break
            except TypeError:
                # error or tol is None
                pass

        if verbosity >= 1:
            print(f"its: {i},\tM: {self.M},\tE: {self.E:.12f},\t"
                  f"ΔE: {ΔE:.3g},\ttrunc: {trunc:.3g}")
            if i == max_iter:
                print("Convergence not reached.")

        return self.E

    def vumps(self, **kwargs):
        """Returns a vumps instance.
        """


class VUMPS:
    """
    For doing VUMPS.

    VUMPS can use both real and complex tensors.

    Attributes:
        NN_interaction: The Nearest neighbour interaction for the hamiltonian
    """
    def __init__(self, NN_interaction=None, pure_real=False):
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
        self._dtype = np.float64 if pure_real else np.complex128

    @property
    def p(self):
        """The dimension of the local physical basis.
        """
        assert len(self.NN_interaction.shape) == 4
        return self.NN_interaction.shape[0]

    @property
    def Ar(self):
        """Right canonical tensor.

        Ar, Al and c are made properties to be sure they always have the
        expected shape.
        """
        return self._Ar.reshape(self.M, self.p, self.M)

    @Ar.setter
    def Ar(self, Ar):
        self._Ar = Ar
        self._Ac = None

    @property
    def Al(self):
        """Left canonical tensor.
        """
        return self._Al.reshape(self.M, self.p, self.M)

    @Al.setter
    def Al(self, Al):
        self._Al = Al
        self._Ac = None

    @property
    def c(self):
        """Central tensor.
        """
        try:
            return self._c.reshape(self.M, self.M)
        except AttributeError:
            return None

    @c.setter
    def c(self, c):
        self._c = c
        self._Ac = None

    @property
    def Ac(self):
        if self._Ac is None:
            self._Ac = self.c @ self.Ar.reshape(self.M, -1)
        return self._Ac.reshape(self.M, self.p, self.M)

    def current_energy_and_error(self):
        """Calculates the energy and estimated error of the current uMPS

        The energy is calculated as the expectation value of Hc for c.

        The error is calculated as ||HAc @ Ac - 2 * Hc @ c||_frobenius,
        which should be zero in the fixed point (i.e. when Ac, Al, Ar and c are
        consistent with each other and Ac and c are eigenstates of HAc and Hc).
        """
        HAcAc = self.HAc(self.Ac)
        Hcc = self.Hc(self.c)
        E = np.dot(self.c.ravel().conj(), Hcc.ravel()).real
        AlHcc = (self.Al.reshape(-1, self.M) @ Hcc.reshape(self.M, -1)).ravel()
        return E, norm(HAcAc - 2 * AlHcc) / (2 * abs(E))

    def H_2site(self, AA):
        return H_2site(self.NN_interaction,
                       AA.reshape(self.M, self.p, self.p, self.M))

    def HAc(self, x):
        # left Heff
        result = (self.Hl @ x.reshape(self.M, -1)).ravel()
        # right Heff
        result += (x.reshape(-1, self.M) @ self.Hr.T).ravel()

        # first onsite
        LL = self.Al.reshape(-1, self.M) @ x.reshape(self.M, -1)
        LL = self.H_2site(LL)

        result += (self.Al.reshape(self.M * self.p, -1).conj().T @
                   LL.reshape(self.M * self.p, -1)).ravel()

        # second onsite
        RR = x.reshape(-1, self.M) @ self.Ar.reshape(self.M, -1)
        RR = self.H_2site(RR)

        result += (RR.reshape(-1, self.M * self.p) @
                   self.Ar.reshape(-1, self.M * self.p).conj().T).ravel()
        return result

    def Hc(self, x):
        x = x.reshape(self.M, self.M)
        # left Heff
        result = (self.Hl @ x).ravel()
        # right Heff
        result += (x @ self.Hr.T).ravel()

        # On site
        C1 = self.Al.reshape(-1, self.M) @ x @ self.Ar.reshape(self.M, -1)
        C1 = self.H_2site(C1)

        C3 = C1.reshape(-1, self.p * self.M) @ \
            self.Ar.reshape(-1, self.p * self.M).conj().T
        result += (self.Al.reshape(self.M * self.p, -1).conj().T @
                   C3.reshape(self.p * self.M, -1)).ravel()
        return result

    def MakeHeff(self, A, c, tol=1e-14):
        def P_NullSpace(x):
            """Projecting x on the nullspace of 1 - T
            """
            x = x.reshape(self.M, self.M)
            return np.trace(c.conj().T @ x @ c) * np.eye(self.M)

        def Transfer(x):
            """Doing (1 - (T - P)) @ x
            """
            x = x.reshape(self.M, self.M)
            res = x.ravel().copy()
            res += P_NullSpace(x).ravel()
            temp = x @ A.reshape(self.M, -1)
            res -= (A.reshape(-1, self.M).conj().T @
                    temp.reshape(-1, self.M)).ravel()

            return res

        AA = A.reshape(-1, self.M) @ A.reshape(self.M, -1)
        HAA = self.H_2site(AA)

        h = AA.reshape(-1, self.M).conj().T @ HAA.reshape(-1, self.M)

        LO = LinearOperator((self.M * self.M,) * 2,
                            matvec=Transfer,
                            dtype=self._dtype
                            )

        r, info = bicgstab(LO, (h - P_NullSpace(h)).ravel(), tol=tol,
                           atol='legacy')

        # return (1 - P) @ result
        r = r.reshape(self.M, self.M)
        return r - P_NullSpace(r), info

    def MakeHl(self, tol):
        result, info = self.MakeHeff(self.Al, self.c, tol)
        if info != 0:
            print(f'Making left environment gave {info} as exit code')
        return result, info

    def MakeHr(self, tol):
        self.NN_interaction = self.NN_interaction.conj().transpose()
        result, info = \
            self.MakeHeff(self.Ar.conj().transpose(), self.c.conj().T, tol)
        self.NN_interaction = self.NN_interaction.conj().transpose()
        if info != 0:
            print(f'Making right environment gave {info} as exit code')
        return result.conj(), info

    def set_uMPS(self, Ac, c, canon=True, tol=1e-14):
        if canon:
            uar = polar(Ac.reshape(self.M, -1), side='left')[0]
            ucr = polar(c.reshape(self.M, self.M), side='left')[0]
            self.Ar = ucr.conj().T @ uar
            self.Al, self.c, info = canonicalize(self.Ar, c, tol, self._dtype)
        else:
            self.c = c
            uar = polar(Ac.reshape(self.M, -1), side='left')[0]
            ucr = polar(c.reshape(self.M, self.M), side='left')[0]
            ual = polar(Ac.reshape(-1, self.M), side='right')[0]
            ucl = polar(c.reshape(self.M, self.M), side='right')[0]
            self.Al, self.Ar = ual @ ucl.conj().T, ucr.conj().T @ uar
            info = [0]
        return info

    def kernel(self, D=16, max_iter=1000, tol=1e-10, verbosity=2, canon=True,
               Ac=None, c=None):
        def print_info(i, vumps, ctol, w1, w2, canon_info):
            print(
                f'it: {i},\t'
                f'E: {vumps.energy:.16g},\t'
                f'Error: {vumps.error:.3g},\t'
                f'tol: {ctol:.3g},\t'
                f'HAc: {w1:.6g},\t'
                f'Hc: {w2:.6g},\t'
                f'c_its: {canon_info}'
            )

        self.M = D
        # Random initial Ac and c guess
        if self._dtype == np.complex128:
            Ac = rand(self.M, self.p, self.M) + \
                rand(self.M, self.p, self.M) * 1j
            c = rand(self.M, self.M) + rand(self.M, self.M) * 1j
        else:
            Ac = rand(self.M, self.p, self.M)
            c = rand(self.M, self.M)
        Ac, c = Ac / norm(Ac), c / norm(c)

        ctol, self.error = 1e-3, 1
        canon_info = self.set_uMPS(Ac, c, canon, tol=ctol)
        self.Hl, _ = self.MakeHl(tol=ctol)
        self.Hr, _ = self.MakeHr(tol=ctol)

        for i in range(max_iter):
            ctol = max(min(1e-3, 1e-3 * self.error), 1e-15)
            etol = ctol ** 2 if ctol ** 2 > 1e-16 else 0

            HAc = LinearOperator(
                (self.M * self.M * self.p,) * 2,
                matvec=lambda x: self.HAc(x),
                dtype=self._dtype
            )
            Hc = LinearOperator(
                (self.M * self.M,) * 2,
                matvec=lambda x: self.Hc(x),
                dtype=self._dtype
            )

            # Solve the two eigenvalues problem for Ac and c
            w1, v1 = eigsh(HAc, v0=self.Ac.ravel(), k=1, which='SA', tol=etol)
            w2, v2 = eigsh(Hc, v0=self.c.ravel(), k=1, which='SA', tol=etol)

            canon_info = self.set_uMPS(v1[:, 0], v2[:, 0], canon, tol=ctol)
            self.Hl, _ = self.MakeHl(tol=ctol)
            self.Hr, _ = self.MakeHr(tol=ctol)
            self.energy, self.error = self.current_energy_and_error()

            if self.error < tol:
                break

            if verbosity >= 2:
                print_info(i, self, ctol, w1[0], w2[0], canon_info[0])

        if verbosity >= 1:
            print_info(i, self, ctol, w1[0], w2[0], canon_info[0])
        return self.energy


class TestDMRG(unittest.TestCase):
    verbosity = 0

    def test_DMRG_diagonal(self):
        ogdmrg = OGDMRG(NN_interaction=IsingInteraction())
        ogdmrg.kernel(D=16, max_iter=50, tol=1e-9, verbosity=self.verbosity)
        diag = ogdmrg.Heffdiagonal()

        def cdiag(i):
            x = np.zeros(len(diag))
            x[i] = 1.
            return ogdmrg.Heff(x)[i]

        calcdiag = np.array([cdiag(i) for i in range(len(diag))])
        assert_allclose(calcdiag, diag)

    def test_DMRG_criticalIsing(self):
        ogdmrg = OGDMRG(NN_interaction=IsingInteraction())
        E = ogdmrg.kernel(D=16, max_iter=1000, tol=1e-9,
                          verbosity=self.verbosity)
        assert_allclose(E, -1.273238747142841)

    def test_DMRG_Heisenberg(self):
        ogdmrg = OGDMRG(NN_interaction=HeisenbergInteraction())
        E = ogdmrg.kernel(D=16, max_iter=1000, sites=4, tol=1e-10,
                          verbosity=self.verbosity)
        assert_allclose(E, -0.4430946668996967)

    def test_VUMPS_Heisenberg(self):
        ogdmrg = VUMPS(NN_interaction=four_site(HeisenbergInteraction()))
        E = ogdmrg.kernel(D=16, max_iter=1000, tol=1e-10,
                          verbosity=self.verbosity)
        assert_allclose(E, -0.4431114747297753)

    def test_VUMPS_criticalIsing(self):
        ogdmrg = VUMPS(NN_interaction=IsingInteraction())
        E = ogdmrg.kernel(D=16, max_iter=1000, tol=1e-10,
                          verbosity=self.verbosity)
        assert_allclose(E, -1.273238977704085)


if __name__ == '__main__':
    unittest.main()
