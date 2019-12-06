import numpy as np
from numpy.random import rand
from numpy.linalg import norm
from scipy.linalg import polar, svd
from scipy.sparse.linalg import eigs, eigsh, LinearOperator
from numpy import einsum


def NN_to_MPO(NN):
    pass


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
            canon = self._idmrg.canon[site_index]

            if (self.reverse and canon != "Right") or \
                    (not self.reverse and canon != "Left"):
                raise IndexError(
                    f"Not able to create environment at {index}, "
                    f"site {index} is {canon} canonicalized"
                )
            if self.reverse:
                contract = 'amb,djb,cia,mijn->cnd'
            else:
                contract = 'amb,bjd,aic,mijn->cnd'

            self._envs[index] = einsum(contract, penv, A, A.conj(),
                                       self._idmrg.MPO)
            return self._envs[index]


class IDMRG:
    """General unit cel iDMRG.

    Attributes:
        cell_size: The size of the unit cell
        central_bond: The current central bond in the two unit cells.
        sites: The MPS's for the current two central unit cells.

        Be aware if you change this that you correctly canonicalize the
        corresponding sites or change the `self.canon` object.
        canon: "Left" "Central" "Right" for every site in `self.sites`.
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

        self._MPO = MPO
        self.MPOdim = MPO.shape[0]
        self.pdim = MPO.shape[1]
        assert MPO.shape[2] == self.pdim
        assert MPO.shape[3] == self.MPOdim

        # We don't initialize the sites
        self.sites, self.canon = None, None
        self._pLcel, self._pRcel = None, None

        self.LEnvironment = Environment(self, reverse=False)
        self.REnvironment = Environment(self, reverse=True)

    @property
    def MPO(self):
        return self._MPO.reshape(self.MPOdim, self.pdim,
                                 self.pdim, self.MPOdim)

    @property
    def Lcel(self):
        """The tensors of the left unit cell
        """
        return self.sites[:self.cell_size]

    @property
    def Rcel(self):
        """The tensors of the left unit cell
        """
        return self.sites[self.cell_size:]

    @property
    def end_bond(self):
        """The index of the most right bond of the two unit cells that are
        updated.
        """
        return self.cell_size * 2

    def kernel(self, D=16, two_site=False, max_iter=100, which='SA',
               msweeps=1):
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
                polar(rand(D, self.pdim * D))[0].reshape(D, self.pdim * D)
                for i in range(self.cell_size)
            ]
            self.canon = ["Left"] * self.cell_size + ["Right"] * self.cell_size
            self.central_bond = self.cell_size

            # Begin of environment
            self.LEnvironment[0] = np.zeros((D, self.MPOdim, D))
            self.LEnvironment[0][:, 0, :] = 1.

            # End of environment
            self.REnvironment[self.end_bond] = np.zeros((D, self.MPOdim, D))
            self.REnvironment[0][:, -1, :] = 1.

        # Two new unit cells are inserted per iteration
        for i in range(max_iter):

            # Update the current unit cells
            for j in range(msweeps):
                self.optimizeUnitCells(D, two_site, which)

            # Insert the two new unit cells
            self.newUnitCells()

        return self.eigenvalue

    def newUnitCells(self):
        """Insert new unit cells which are copies of the current unitcells.

        It also appropriately updates the environments.
        """
        # Update the environment, this is:
        #   Setting the left environment in the middle to base left.
        #   Setting the right environment in the middle to base right.
        #   Deleting the other environments.
        assert self.central_bond == self.cell_size
        self.LEnvironment[0] = self.LEnvironment[self.central_bond]
        self.REnvironment[self.end_bond] = self.REnvironment[self.central_bond]

        # Pushing left and right cells to the previous ones
        self._pLcel = tuple([L.copy() for L in self.Lcel])
        self._pRcel = tuple([R.copy() for R in self.Rcel])

    def optimizeUnitCells(self, D, two_site, which):
        """Optimizes the two unit cells.
        """
        assert not two_site
        for site in 
