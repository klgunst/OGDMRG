import numpy as np
from numpy.random import rand
from numpy.linalg import norm
from scipy.linalg import polar
from scipy.linalg import svd
from scipy.sparse.linalg import LinearOperator, eigs
from numpy import tensordot
import ogdmrg

import unittest
from numpy.testing import assert_allclose
from pyscf.lib import davidson


def qrpos(A, tol=1e-10):
    """Positive QR
    """
    from scipy.linalg import qr
    q, r = qr(A, mode='economic')

    for i in range(q.shape[1]):
        fel = np.argmax(abs(r[i, :]) > tol)
        phase = (r[i, fel] / abs(r[i, fel]))
        r[i, :] *= phase.conj()
        q[:, i] *= phase
    assert np.allclose(A, q @ r)
    return q, r


def usvd(A, tol=1e-13):
    """Trying to make SVD unique.
    """
    u, s, v = svd(A)

    svd_diff = (s[:-1] - s[1:])
    # Array with Trues every time this singular value is different than the
    # previous one (up to a tolerance)
    ns = np.concatenate(([0], np.where(svd_diff > tol)[0] + 1, [len(s)]))
    AA = u @ np.diag(s) @ v

    for begin, end in zip(ns, ns[1:]):
        q, r = qrpos(u[:, begin:end].conj().T)
        u[:, begin:end] = r.conj().T
        v[begin:end, :] = q.conj().T @ v[begin:end, :]

    assert np.allclose(u.conj().T @ u, np.eye(u.shape[1]))
    assert np.allclose(v @ v.conj().T, np.eye(v.shape[0]))

    assert np.allclose(AA, u @ np.diag(s) @ v)
    return u, s, v


def NN_to_MPO(NN, tol=1e-12):
    """Transforms a nearest neighbour interaction term (bra1, bra2, ket1, ket2)
    to its corresponding MPO (MPO1, ket1, MPO2, bra1).
    """
    NN = NN.transpose([0, 2, 1, 3])
    pdim = NN.shape[0]
    u, s, v = svd(NN.reshape(pdim * pdim, pdim * pdim))
    accept = s > tol
    u = u[:, accept].reshape(pdim, pdim, -1)
    v = (s[accept, None] * v[accept, :]).reshape(-1, pdim, pdim)
    MPOdim = sum(accept) + 2

    MPO = np.zeros((MPOdim, pdim, MPOdim, pdim))
    MPO[0, :, 0, :] = np.eye(pdim)
    MPO[-1, :, -1, :] = np.eye(pdim)
    MPO[0, :, 1:-1, :] = u.transpose([1, 2, 0])
    MPO[1:-1, :, -1, :] = v.transpose([0, 2, 1])
    return MPO


class Environment:
    """For the environments of DMRG. This object is indexeable.

    It automatically updates if needed.

    Attributes:
        maxlen: The maximal amount of environments stored.
        reverse: True if right environment else False.
    """
    def __init__(self, idmrg, reverse=False):
        """
        Attributes:
            idmrg: The IDMRG instance.
            reverse: True if it is a right environment else, false.
        """
        self._idmrg = idmrg
        self.reverse = reverse
        self.maxlen = idmrg.end_bond + 1
        self._envs = [None] * self.maxlen

    def __setitem__(self, index, value):
        self._envs[index] = value

        # Set also all next environments which were dependent of the previous
        # one to None
        if self.reverse:
            for i in range(index):
                self._envs[i] = None
        else:
            for i in range(index + 1, self.maxlen):
                self._envs[i] = None

    def __getitem__(self, index):
        if self._envs[index] is not None:
            return self._envs[index]
        else:
            # Environment not calculated yet, we will have to form it ourselves

            # The previous environment
            pbond_index = index + (1 if self.reverse else -1)
            site_index = index + (0 if self.reverse else -1)
            penv = self[pbond_index]
            A = self._idmrg.sites[site_index]
            rightCanon = self._idmrg.center_bond <= site_index

            if self.reverse != rightCanon:
                canontype = "Right" if rightCanon else "Left"
                raise IndexError(
                    f"Not able to create environment at {index}, "
                    f"site {index} is {canontype} canonicalized"
                )

            if self.reverse:
                env = tensordot(A.conj(), penv, [[2], [0]])
                env = tensordot(env, self._idmrg.MPO, [[1, 2], [3, 2]])
                self._envs[index] = tensordot(env, A, [[1, 3], [2, 1]])
            else:
                env = tensordot(A.conj(), penv, [[0], [0]])
                env = tensordot(env, self._idmrg.MPO, [[0, 2], [3, 0]])
                self._envs[index] = tensordot(env, A, [[1, 2], [0, 1]])

            return self._envs[index]


class IDMRG:
    """General unit cel iDMRG.

    Attributes:
        cell_size: The size of the unit cell.
        center_bond: The current central bond in the two unit cells.
        sites: The MPS's for the current two central unit cells.
        c: The center tensor.
        kind: Type of MPO inputted. Either 'h' for a Hamiltonian or 'pf' for a
        partition function.
    """

    def __init__(self, MPO, kind='h', cell_size=1):
        """Initializer for the IDMRG class.

        Environments will be set to zero and the initial tensors are randomly
        initialized.

        Attributes:
            MPO: The MPO for the problem.
            kind: Type of MPO inputted. Either 'h' for a Hamiltonian or 'pf'
            for a partition function.
            cell_size: The size of one unit cell.
        """
        if kind != 'h' and kind != 'pf':
            raise ValueError(
                f'Wrong kind inputted ({kind}), should be either "h" or "pf".'
            )

        self.cell_size = cell_size
        self.kind = kind

        self.MPO = MPO
        self.MPOdim = MPO.shape[0]
        self.pdim = MPO.shape[1]
        assert MPO.shape[3] == self.pdim
        assert MPO.shape[2] == self.MPOdim

        # We don't initialize the sites
        self.sites, self.center_bond = None, None
        self._prev_sites = None

        self.LEnvironment = Environment(self, reverse=False)
        self.REnvironment = Environment(self, reverse=True)

    @property
    def energy(self):
        """Energy per unit cell

        For `self.kind == 'h'` it is the energy.
        For `self.kind == 'pf'` it is -β * (Helmholtz free energy).
        """
        if self.kind == 'h':
            return self.eigenval / (2 * self.cell_size)
        elif self.kind == 'pf':
            return np.log(self.eigenval) / (2 * self.cell_size)
        else:
            raise ValueError("invalid `self.kind`")

    @property
    def sign(self):
        """Quick and dirty fix such that I can use 'SA' for eigsh for 'LA'
        also.
        """
        return -1 if self.kind == 'pf' else 1

    @property
    def Lcel(self):
        """The tensors of the left unit cell.
        """
        return self.sites[:self.cell_size]

    @property
    def pLcel(self):
        """The tensors of the previous left unit cell.
        """
        try:
            return self._prev_sites[:self.cell_size]
        except TypeError:
            return None

    @property
    def Rcel(self):
        """The tensors of the right unit cell.
        """
        return self.sites[self.cell_size:]

    @property
    def pRcel(self):
        """The tensors of the previous right unit cell.
        """
        try:
            return self._prev_sites[self.cell_size:]
        except TypeError:
            return None

    @property
    def end_bond(self):
        """The index of the most right bond of the two unit cells that are
        updated.
        """
        return self.cell_size * 2

    def transfer_eig(self, A_array, B_array):
        """Calculates the largest eigenvalue of the mixed transfer matrix of
        the cell A_array wit the cell B_array.
        """
        if A_array is None or B_array is None:
            return 0

        # A_array and B_array have tensors of different shape
        if any([A.shape != B.shape for A, B in zip(A_array, B_array)]):
            return 0

        def transfer(A, B, x):
            x = x.reshape(A.shape[0], B.shape[0])
            x = tensordot(x, A, [[0], [0]])
            return tensordot(x, B.conj(), [[0, 1], [0, 1]]).ravel()

        def fullTF(x):
            for A, B in zip(A_array, B_array):
                x = transfer(A, B, x)
            return x

        LO = LinearOperator((A_array[0].shape[0] ** 2,) * 2, matvec=fullTF)
        w, v = eigs(LO, k=1, which='LM')
        return w[0]

    def random_init(self, D):
        """Random initialization of the MPS and initializiation of the
        environments.
        """
        Lcanon = [polar(rand(D * self.pdim, D) + rand(D * self.pdim, D) * 1j
                        )[0].reshape(D, self.pdim, D)
                  for i in range(self.cell_size)]
        Rcanon = [polar(rand(D, self.pdim * D) + rand(D, self.pdim * D) * 1j
                        )[0].reshape(D, self.pdim, D)
                  for i in range(self.cell_size)]

        self.sites = Lcanon + Rcanon
        self.c = rand(D, D) + rand(D, D) * 1j
        self.c = self.c / norm(self.c)
        self.center_bond = self.cell_size

        # Begin of environment
        LE = np.zeros((D, self.MPOdim, D))
        LE[:, 0, :] = 1.
        LE = LE / norm(LE)
        self.LEnvironment[0] = LE

        # End of environment
        RE = np.zeros((D, self.MPOdim, D))
        RE[:, -1, :] = 1.
        RE = RE / norm(RE)
        self.REnvironment[self.end_bond] = RE

    def kernel(self, D=16, two_site=False, max_iter=10, msweeps=1, verbosity=1,
               rotate=True):
        """Optimize the iDMRG.

        Args:
            D: The bond dimension of the MPS.
            two_site: True if you wan't do to two-site update, False if you
            want to do one-site update.
            max_iter: Maximal number of unitcells to optimize in this update.
            verbosity:
                0: Don't print anything
                1: Print Every macro iteration
                2: Print Every micro iteration
        """
        if self.sites is None:
            self.random_init(D)

        # Two new unit cells are inserted per iteration
        for i in range(max_iter):

            # Update the current unit cells
            for j in range(msweeps):
                self.optimizeUnitCells(D, two_site, verbosity, rotate)

            info = {
                'it': i,
                'energy': self.energy,
                'L_tf': 1 - abs(self.transfer_eig(self.Lcel, self.pLcel)),
                'R_tf': 1 - abs(self.transfer_eig(self.Rcel, self.pRcel)),
                'mixed_tf': 1 - abs(self.transfer_eig(self.Lcel, self.Rcel))
            }
            if verbosity >= 1:
                print(info)
            self.newUnitCells()

        print({
            'it': i,
            'energy': self.energy,
            'L_tf': 1 - abs(self.transfer_eig(self.Lcel, self.pLcel)),
            'R_tf': 1 - abs(self.transfer_eig(self.Rcel, self.pRcel)),
            'mixed_tf': 1 - abs(self.transfer_eig(self.Lcel, self.Rcel))
        })
        return self.energy

    def newUnitCells(self):
        """Insert new unit cells which are copies of the current unitcells.

        It also appropriately updates the environments.
        """
        # Update the environment, this is:
        #   Setting the left environment in the middle to base left.
        #   Setting the right environment in the middle to base right.
        #   Deleting the other environments.
        #   Adding appropriate shifts or rescaling
        assert self.center_bond == self.cell_size
        if self.kind == 'h':
            shift = self.eigenval / 2
            self.LEnvironment[0] = self.LEnvironment[self.center_bond]
            D = self.LEnvironment[0].shape[0]
            self.LEnvironment[0][:, -1, :] -= np.eye(D) * shift
            self.REnvironment[self.end_bond] = \
                self.REnvironment[self.center_bond]
            self.REnvironment[self.end_bond][:, 0, :] -= np.eye(D) * shift
        elif self.kind == 'pf':
            scale = 1. / np.sqrt(self.eigenval)
            self.LEnvironment[0] = scale * self.LEnvironment[self.center_bond]
            self.REnvironment[self.end_bond] = scale * \
                self.REnvironment[self.center_bond]

        # Pushing left and right cells to the previous ones
        self._prev_sites = [s.copy() for s in self.sites]

        # Padding with zeros
        if self.LEnvironment[0].shape[0] != self.sites[0].shape[0]:
            orig = self.sites[0]
            newD = self.LEnvironment[0].shape[0]
            padded = np.zeros((newD, *orig.shape[1:]), dtype=complex)
            padded[:orig.shape[0], :, :] = orig
            self.sites[0] = padded

        if self.REnvironment[self.end_bond].shape[0] != \
                self.sites[-1].shape[-1]:
            orig = self.sites[-1]
            newD = self.REnvironment[self.end_bond].shape[0]
            padded = np.zeros((*orig.shape[:-1], newD), dtype=complex)
            padded[:, :, :orig.shape[-1]] = orig
            self.sites[-1] = padded

    def _sweep(self, two_site):
        """Sweeps through the bonds of the two central unit cells starting from
        the center, first going to the right, then to the left, and then back
        to the center.

        Args:
            two_site: True if you are doing two site, False for one site opt.

        Returns:
            A bunch of info
        """
        assert self.center_bond == self.cell_size

        edge = 0 if two_site else 1
        for i in range(self.cell_size, self.end_bond - 1 + edge):
            yield {
                'center_at': 'left',
                'sites': (i, i + 1) if two_site else (i,),
                'side': 'right' if i != self.end_bond - 1 else 'left'
            }

        for i in range(self.end_bond - 1, 1 - edge, -1):
            yield {
                'center_at': 'right',
                'sites': (i - 2, i - 1) if two_site else (i - 1,),
                'side': 'left' if i != 1 else 'right'
            }
        for i in range(1, self.cell_size):
            yield {
                'center_at': 'left',
                'sites': (i, i + 1) if two_site else (i,),
                'side': 'right' if i != self.end_bond - 1 else 'left'
            }

        if two_site and self.cell_size == 1:
            yield {
                'center_at': 'center',
                'sites': (0, 1),
            }

    def make_center_site(self, two_site, info):
        """Makes the center site (either two-site or one site).
        """
        if info['center_at'] == 'center':
            A1 = self.sites[info['sites'][0]]
            A2 = self.sites[info['sites'][1]]
            AA = tensordot(A1, self.c, axes=1)
            AA = tensordot(AA, A2, axes=1)
            info['AAshape'] = AA.shape
            return AA

        if two_site:
            A1 = self.sites[info['sites'][0]]
            A2 = self.sites[info['sites'][1]]
            AA = tensordot(A1, A2, axes=1)
        else:
            AA = self.sites[info['sites'][0]]
        info['AAshape'] = AA.shape

        # Absorb the center site, which is either to the
        #   * left when moving right
        #   * right when moving left
        if info['center_at'] == 'left':
            return tensordot(self.c, AA, axes=1)
        elif info['center_at'] == 'right':
            return tensordot(AA, self.c, axes=1)
        else:
            ValueError(f'Invalid info["center_at"]: {info["center_at"]}')

    def rotate(self, A, site, side, rotate):
        """Rotates the optimized site.
        """
        if side != 'left' and side != 'right':
            ValueError(f'Invalid side: {side}, choose "left" or "right".')

        # No previous sites saved
        if self._prev_sites is None or not rotate:
            return A, np.eye(A.shape[0 if side == 'left' else -1]), None

        # The site will be canonicalized to the left while it is part of the
        # right unit cell or vice versa. So no rotation needed
        if (side == 'left') == (site < self.cell_size):
            return A, np.eye(A.shape[0 if side == 'left' else -1]), None

        pA = self._prev_sites[site]

        # Different shape with previous tensor.
        if pA.shape != A.shape:
            return A, np.eye(A.shape[0 if side == 'left' else -1]), None

        if side == 'left':
            AA = tensordot(pA, A.conj(), [[1, 2], [1, 2]])
        else:
            AA = tensordot(A.conj(), pA, [[0, 1], [0, 1]])
        u, s, v = svd(AA, full_matrices=False)

        Q, uni = u @ v, np.max(abs(s - 1))
        if side == 'left':
            A = tensordot(Q, A, [[1], [0]])
        else:
            A = tensordot(A, Q, [[2], [0]])

        return A, Q, uni

    def update_sites(self, AA, two_site, info, D, rotate=True):
        """Recanonicalizes the updated sites and moves the center.

        Canonicalization is done by minimizing the difference between this
        site and the equivalent site in the previous unit cell.
        """
        sites = info['sites']
        print_info = {}
        if two_site:
            assert len(sites) == 2
            newshape = (info['AAshape'][0] * info['AAshape'][1], -1)
            u, s, v = usvd(AA.reshape(newshape))
            D = min(len(s), D)

            # Fill in left and right site and center site
            A1 = u[:, :D].reshape(-1, self.pdim, D)
            A2 = v[:D, :].reshape(D, self.pdim, -1)
            self.c = np.diag(s[:D] / norm(s[:D]))

            # fix the guage
            self.sites[sites[0]], Q, print_info['Qunity'] = \
                self.rotate(A1, sites[0], 'right', rotate)

            self.sites[sites[1]], P, print_info['Punity'] = \
                self.rotate(A2, sites[1], 'left', rotate)

            self.c = Q.conj().T @ self.c @ P.conj().T

            # Setting new center bond
            self.center_bond = sites[1]
            self.LEnvironment[self.center_bond] = None
            self.REnvironment[self.center_bond] = None
            print_info['trunc'] = s[D:] @ s[D:]
        else:
            assert len(sites) == 1
            if info['side'] == "right":
                newshape = (-1, info['AAshape'][2])
            elif info['side'] == "left":
                newshape = (info['AAshape'][0], -1)
            else:
                ValueError(f'Invalid info["side"]: {info["side"]}')

            if info['side'] == 'left':
                newshape = (info['AAshape'][0],  -1)
            else:
                newshape = (-1, info['AAshape'][-1])
            A, self.c = polar(AA.reshape(newshape), side=info['side'])
            A = A.reshape(info['AAshape'])

            # fix the guage
            self.sites[sites[0]], Q, print_info['Qunity'] = \
                self.rotate(A, sites[0], info['side'], rotate)

            if info['side'] == "right":
                self.c = Q.conj().T @ self.c
                self.center_bond = sites[0] + 1
                self.LEnvironment[self.center_bond] = None
            elif info['side'] == "left":
                self.c = self.c @ Q.conj().T
                self.center_bond = sites[0]
                self.REnvironment[self.center_bond] = None
            else:
                ValueError(f'Invalid info["side"]: {info["side"]}')
        return print_info

    def Heff(self, AA, two_site, info):
        """Executes a matvec.
        """
        AA = AA.reshape(info['AAshape'])
        LE = self.LEnvironment[info['sites'][0]]
        RE = self.REnvironment[info['sites'][-1] + 1]

        result = tensordot(LE, AA, axes=1)
        result = tensordot(result, self.MPO, axes=[[1, 2], [0, 1]])
        if two_site:
            result = tensordot(result, self.MPO, axes=[[1, 3], [1, 0]])
        result = tensordot(result, RE, axes=[[1, 2 + two_site], [2, 1]])
        return self.sign * result.ravel()

    def heff_diagonal(self, two_site, info):
        """diagonal of the matvec
        """
        LE = self.LEnvironment[info['sites'][0]]
        RE = self.REnvironment[info['sites'][-1] + 1]

        LE_diag = np.diagonal(LE, axis1=0, axis2=2)
        RE_diag = np.diagonal(RE, axis1=0, axis2=2)
        MPO_diag = np.diagonal(self.MPO, axis1=1, axis2=3)

        diag = tensordot(LE_diag, MPO_diag, axes=[[0], [0]])
        if two_site:
            diag = tensordot(diag, MPO_diag, axes=[[1], [0]])
        return self.sign * \
            tensordot(diag, RE_diag, axes=[[1 + two_site], [0]]).ravel()

    def optimizeUnitCells(self, D, two_site, verbosity, rotate):
        """Optimizes the two unit cells.
        """
        for info in self._sweep(two_site):
            # Create the big site
            AA = self.make_center_site(two_site, info)
            diagonal = self.heff_diagonal(two_site, info)

            w, v = davidson(lambda x: self.Heff(x, two_site, info),
                            x0=AA.ravel(), precond=diagonal)
            self.eigenval = w * self.sign

            print_info = {
                'sites': info['sites'],
                'fidelity': 1 - abs(np.dot(AA.ravel().conj(), v)),
                'energy': self.energy,
            }
            print_info = {**print_info,
                          **self.update_sites(v, two_site, info, D, rotate)}
            if verbosity >= 2:
                print(print_info)
        if verbosity >= 2:
            print()

    def transform_LR_guage(self, Q, P):
        """Go from ... Al Al Al Al c Ar Ar Ar Ar Ar ...
        to ... Al Q' Q Al Q' Q Al Q' Q c P P' Ar P P' Ar ...

        Q and P are inserted at unit cell devisions.

        * Al_new = Q Al Q'
        * Ar_new = P' Ar P
        * c_new = Q c P
        """
        self.c = Q @ self.c @ P
        self.sites[0] = tensordot(Q, self.sites[0], axes=1)
        self.sites[-1] = tensordot(self.sites[-1], P, axes=1)
        self.sites[self.cell_size - 1] = \
            tensordot(self.sites[self.cell_size - 1], Q.conj().T, axes=1)
        self.sites[self.cell_size] = \
            tensordot(P.conj().T, self.sites[self.cell_size], axes=1)

        self._prev_sites[0] = tensordot(Q, self._prev_sites[0], axes=1)
        self._prev_sites[-1] = tensordot(self._prev_sites[-1], P, axes=1)
        self._prev_sites[self.cell_size - 1] = \
            tensordot(self._prev_sites[self.cell_size - 1], Q.conj().T, axes=1)
        self._prev_sites[self.cell_size] = \
            tensordot(P.conj().T, self._prev_sites[self.cell_size], axes=1)

        self.LEnvironment[0] = \
            tensordot(tensordot(Q.T, self.LEnvironment[0], axes=[[0], [0]]),
                      Q.conj().T, axes=[[2], [0]])
        self.REnvironment[self.end_bond] = \
            tensordot(tensordot(P.T, self.REnvironment[self.end_bond],
                                axes=1), P.conj().T, axes=[[2], [1]])


class TestDMRG(unittest.TestCase):
    verbosity = 0

    def test_DMRG_diagonal(self):
        idmrg = IDMRG(NN_to_MPO(ogdmrg.HeisenbergInteraction()), cell_size=2)
        idmrg.kernel(D=16, max_iter=10, msweeps=1, verbosity=self.verbosity)

        two_site = True
        info = next(idmrg._sweep(two_site))
        info['AAshape'] = idmrg.make_center_site(two_site, info).shape
        diag = idmrg.heff_diagonal(two_site, info)

        def cdiag(i):
            x = np.zeros(len(diag))
            x[i] = 1.
            return idmrg.Heff(x, two_site, info)[i]
        calcdiag = np.array([cdiag(i) for i in range(len(diag))])
        assert_allclose(calcdiag, diag)


if __name__ == '__main__':
    unittest.main()
