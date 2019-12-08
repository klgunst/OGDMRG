import numpy as np
from numpy.random import rand
from numpy.linalg import norm
from scipy.linalg import polar, svd
from scipy.sparse.linalg import eigsh, LinearOperator, eigs
from numpy import einsum
import ogdmrg


def NN_to_MPO(NN, tol=1e-12):
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
    """For the environments of DMRG.
    """
    def __init__(self, idmrg, reverse=False):
        """
        Attributes:
            idmrg: The IDMRG instance
            reverse: True if it is a right environment else, false.
        """
        self._idmrg = idmrg
        self.reverse = reverse
        self.maxlen = idmrg.end_bond + 1
        self._envs = [None] * self.maxlen

    def __setitem__(self, index, value):
        self._envs[index] = value
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
                contract = 'anb,dib,cja,minj->cmd'
            else:
                contract = 'amb,bid,ajc,minj->cnd'

            self._envs[index] = einsum(contract, penv, A, A.conj(),
                                       self._idmrg.MPO)
            return self._envs[index]


class IDMRG:
    """General unit cel iDMRG.

    Attributes:
        cell_size: The size of the unit cell
        center_bond: The current central bond in the two unit cells.
        sites: The MPS's for the current two central unit cells.
        c: The center tensor.
    """

    def __init__(self, MPO, cell_size=1):
        """Initializer for the IDMRG class.

        Environments will be set to zero and the initial tensors are randomly
        initialized.

        Attributes:
            MPO: The MPO for the problem
            cell_size: The size of one unit cell.
        """
        self.cell_size = cell_size

        self.MPO = MPO
        self.MPOdim = MPO.shape[0]
        self.pdim = MPO.shape[1]
        assert MPO.shape[3] == self.pdim
        assert MPO.shape[2] == self.MPOdim

        # We don't initialize the sites
        self.sites, self.center_bond = None, None
        self._previous_sites = None
        self.previous_E = 0

        self.LEnvironment = Environment(self, reverse=False)
        self.REnvironment = Environment(self, reverse=True)

    @property
    def energy(self):
        """Energy per unit cell
        """
        return (self.current_E - self.previous_E) / 2 / self.cell_size

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
            return self._previous_sites[:self.cell_size]
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
            return self._previous_sites[self.cell_size:]
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

        def transfer(A, B, x):
            x = x.reshape(A.shape[0], B.shape[0])
            x = np.tensordot(x, A, [[0], [0]])
            return np.tensordot(x, B.conj(), [[0, 1], [0, 1]]).ravel()

        def fullTF(x):
            for A, B in zip(A_array, B_array):
                x = transfer(A, B, x)
            return x

        LO = LinearOperator((A_array[0].shape[0] ** 2,) * 2, matvec=fullTF)
        w, v = eigs(LO, k=1, which='LM')
        return w[0]

    def kernel(self, D=16, two_site=False, max_iter=10, which='SA',
               msweeps=1, verbosity=1):
        """Optimize the iDMRG.

        Args:
            D: The bond dimension of the MPS.
            two_site: True if you wan't do to two-site update, False if you
            want to do one-site update.
            max_iter: Maximal number of unitcells to optimize in this update.
            which: Which eigenvalue do you want to find?
            Same arguments as in `scipy.sparse.linalg.eigsh` are allowed.
        """
        if self.sites is None:
            self.sites = [
                polar(rand(D * self.pdim, D))[0].reshape(D, self.pdim, D)
                for i in range(self.cell_size)
            ] + [
                polar(rand(D, self.pdim * D))[0].reshape(D, self.pdim, D)
                for i in range(self.cell_size)
            ]
            self.c = rand(D, D)
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

        # Two new unit cells are inserted per iteration
        for i in range(max_iter):
            # Update the current unit cells
            for j in range(msweeps):
                self.optimizeUnitCells(D, two_site, which, verbosity)

            info = {
                'it': i,
                'L_tf': 1 - abs(self.transfer_eig(self.Lcel, self.pLcel)),
                'R_tf': 1 - abs(self.transfer_eig(self.Rcel, self.pRcel)),
                'mixed_tf': 1 - abs(self.transfer_eig(self.Lcel, self.Rcel))
            }
            if verbosity >= 2:
                print(info)
            if verbosity >= 3:
                print("Inserting cells\n")
            self.newUnitCells()

        return self.energy

    def newUnitCells(self):
        """Insert new unit cells which are copies of the current unitcells.

        It also appropriately updates the environments.
        """
        # Update the environment, this is:
        #   Setting the left environment in the middle to base left.
        #   Setting the right environment in the middle to base right.
        #   Deleting the other environments.
        assert self.center_bond == self.cell_size
        self.previous_E = self.current_E
        self.LEnvironment[0] = self.LEnvironment[self.center_bond]
        self.REnvironment[self.end_bond] = self.REnvironment[self.center_bond]

        # Pushing left and right cells to the previous ones
        self._previous_sites = tuple([s.copy() for s in self.sites])

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
            AA = np.tensordot(A1, self.c, axes=1)
            AA = np.tensordot(AA, A2, axes=1)
            info['AAshape'] = AA.shape
            return AA

        if two_site:
            A1 = self.sites[info['sites'][0]]
            A2 = self.sites[info['sites'][1]]
            AA = np.tensordot(A1, A2, axes=1)
        else:
            AA = self.sites[info['sites'][0]]
        info['AAshape'] = AA.shape

        # Absorb the center site, which is either to the
        #   * left when moving right
        #   * right when moving left
        if info['center_at'] == 'left':
            return np.tensordot(self.c, AA, axes=1)
        elif info['center_at'] == 'right':
            return np.tensordot(AA, self.c, axes=1)
        else:
            ValueError(f'Invalid info["center_at"]: {info["center_at"]}')

    def rotate(self, A, site, side, rotate):
        """Rotates the optimized site
        """
        # rotate = False
        if side != 'left' and side != 'right':
            ValueError(f'Invalid side: {side}, choose "left" or "right".')

        # No previous sites saved
        if self._previous_sites is None or not rotate:
            return A, np.eye(A.shape[0 if side == 'left' else -1]), None

        # The site will be canonicalized to the left while it is part of the
        # right unit cell or vice versa.
        if (side == 'left') == (site < self.cell_size):
            return A, np.eye(A.shape[0 if side == 'left' else -1]), None

        pA = self._previous_sites[site]
        if side == 'left':
            AA = np.tensordot(pA.conj(), A, [[1, 2], [1, 2]])
        else:
            AA = np.tensordot(A, pA.conj(), [[0, 1], [0, 1]])
        u, s, v = svd(AA, full_matrices=False)

        Q, uni = u @ v, np.max(abs(s - 1))
        if side == 'left':
            A = np.tensordot(Q, A, [[1], [0]])
        else:
            A = np.tensordot(A, Q, [[2], [0]])

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
            u, s, v = svd(AA.reshape(newshape))

            # Fill in left and right site and center site
            A1 = u[:, :D].reshape(-1, self.pdim, D)
            A2 = v[:D, :].reshape(D, self.pdim, -1)
            self.c = np.diag(s[:D] / norm(s[:D]))

            # fix the guage
            self.sites[sites[0]], Q, print_info['Qunity'] = \
                self.rotate(A1, sites[0], 'right', rotate)

            self.sites[sites[1]], P, print_info['Punity'] = \
                self.rotate(A2, sites[1], 'left', rotate)

            self.sites[sites[0]] = self.sites[sites[0]].reshape(
                info['AAshape'][0], info['AAshape'][1], -1
            )
            self.sites[sites[1]] = self.sites[sites[1]].reshape(
                -1, info['AAshape'][-2], info['AAshape'][-1]
            )
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
                newshape = (info['AAshape'][0] * info['AAshape'][1], -1)
            A, self.c = polar(AA.reshape(newshape), side=info['side'])

            # fix the guage
            self.sites[sites[0]], Q, print_info['Qunity'] = \
                self.rotate(A, sites[0], info['side'], rotate)

            self.sites[sites[0]] = \
                self.sites[sites[0]].reshape(info['AAshape'])

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
        AA = AA.reshape(info['AAshape'])
        lbond = info['sites'][0]
        rbond = info['sites'][-1] + 1

        result = np.tensordot(self.LEnvironment[lbond], AA, axes=1)
        result = np.tensordot(result, self.MPO, axes=[[1, 2], [0, 1]])
        if two_site:
            result = np.tensordot(result, self.MPO, axes=[[1, 3], [1, 0]])
        result = np.tensordot(
            result, self.REnvironment[rbond], axes=[[1, 2 + two_site], [2, 1]]
        )
        return result.ravel()

    def optimizeUnitCells(self, D, two_site, which, verbosity):
        """Optimizes the two unit cells.
        """
        for info in self._sweep(two_site):
            # Create the big site
            AA = self.make_center_site(two_site, info)
            H = LinearOperator(
                (AA.size,) * 2, matvec=lambda x: self.Heff(x, two_site, info)
            )
            w, v = eigsh(H, k=1, v0=AA.ravel(), which=which)
            self.current_E = w[0]
            print_info = {
                'sites': info['sites'],
                'fidelity': 1 - abs(np.dot(AA.ravel().conj(), v[:, 0])),
                'energy': self.energy
            }
            print_info = {**print_info,
                          **self.update_sites(v[:, 0], two_site, info, D)}
            if verbosity >= 3:
                print(print_info)


if __name__ == '__main__':
    idmrg = IDMRG(NN_to_MPO(ogdmrg.HeisenbergInteraction()), cell_size=2)
    idmrg.kernel(D=16, two_site=True, max_iter=200, msweeps=1, verbosity=3)
