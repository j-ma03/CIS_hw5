import numpy as np
from numpy.typing import NDArray
from typing import Union
from tqdm import tqdm

"""
Code created by Edmund Sumpena and Jayden Ma
"""

class PointCloudRegistration():
    """
    Class that performs point-cloud to point-cloud registration.
    """
    def __init__(
        self,
        eps: float = 1e-3,
        max_epochs: Union[int, None] = 100,
        verbose: bool = True
    ) -> None:
        # Minimum acceptable registration maximum residual error
        self.eps = eps

        # Maximum number of epochs before terminating registration algorithm
        self.max_epochs = max_epochs

        # Toggle progress bar and print statements
        self.verbose = verbose

    """
    Performs registration between a batch of two point clouds (each containing a set 
    of corresponding x, y, z coordinates) given as BxNx3 or BxNx4 matrices. Computes 
    transformation from pt_cloud_a to pt_cloud_b with a direct iterative approach.

    Based on Dr. Russell Tayor's lecture slides: https://ciis.lcsr.jhu.edu/lib/exe/fetch.php?media=courses:455-655:lectures:rigid3d3dcalculations.pdf
    """
    def register(
        self,
        pt_cloud_a: NDArray[np.float32],
        pt_cloud_b: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        
        # Point cloud inputs must be in format BxNx3 (B = batch size, N = number of points)
        if len(pt_cloud_a.shape) != 3 or not 3 <= pt_cloud_a.shape[2] <= 4 or \
            len(pt_cloud_b.shape) != 3 or not 3 <= pt_cloud_b.shape[2] <= 4:
            raise Exception(f'Point clouds must have shape BxNx3 or BxNx4!')
        
        # Point clouds must have matching shapes
        if pt_cloud_a.shape != pt_cloud_b.shape:
            raise Exception(f'Array shape of point clouds do not match!')
        
        # Pad point cloud matrix to create homogenous coordinates if necessary
        if pt_cloud_a.shape[2] == 3:
            homog = np.ones((*pt_cloud_a.shape[:-1], 1))
            pt_cloud_a = np.concatenate([pt_cloud_a, homog], axis=2)
        
        # Use identity matrix as initial guess for the registration transformation
        F = np.eye(4)
        
        if self.verbose:
            print('Performing 3D point cloud registration...')

        # Perform direct iterative registration algorithm
        for _ in (pbar := tqdm(range(self.max_epochs), bar_format="{n_fmt}ep [{elapsed}<{remaining}, {rate_fmt}{postfix}]", disable=(not self.verbose))):
            # Store error averaged across the entire batch
            err_epoch = np.zeros((3,))

            for j in range(pt_cloud_a.shape[0]):
                # Compute transformation using the current transformation prediction
                predicted = F @ pt_cloud_a[j].T

                # Calculate transformation that minimizes error and update F
                del_F, err = self._register_cloud(predicted[:3].T, pt_cloud_b[j,:,:3])
                F = del_F @ F
                err_epoch += err

                # Display error on progress bar
                pbar.set_postfix(err=err.max())
                pbar.update(0)

            # Compute average mean squared error across the entire batch
            err_epoch /= pt_cloud_a.shape[0]

            # Stop algorithm if maximum squared residual error is below threshold
            if err_epoch.max() < self.eps:
                break
        
        if self.verbose:
            print('Done.')

        return F, err

    """
    Performs registration between two point clouds (set of corresponding x, y, z 
    coordinates) given as Nx3 matrices. Computes transformation from pt_cloud_a 
    to pt_cloud_b with least squares method.

    Based on Dr. Russell Tayor's lecture slides: https://ciis.lcsr.jhu.edu/lib/exe/fetch.php?media=courses:455-655:lectures:rigid3d3dcalculations.pdf
    """
    def _register_cloud(
        self,
        pt_cloud_a: NDArray[np.float32],
        pt_cloud_b: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        
        # Point cloud inputs must be in format Nx3
        if len(pt_cloud_a.shape) != 2 or not pt_cloud_a.shape[1] == 3 or \
            len(pt_cloud_b.shape) != 2 or not pt_cloud_b.shape[1] == 3:
            raise Exception(f'Point clouds must have shape Nx3!')

        # Point clouds must have matching shapes
        if pt_cloud_a.shape != pt_cloud_b.shape:
            raise Exception(f'Array shape of point clouds do not match!')

        # Construct M and b matrices to solve for alpha and epsilon
        N = pt_cloud_a.shape[0]
        M = np.zeros((3 * N, 6))
        b = np.zeros(3 * N)

        for i in range(N):
            # Create skew-symmetric matrix with vectors from point cloud a
            M[3*i:3*i+3,:3] = self.skew_symmetric(-pt_cloud_a[i])
            M[3*i:3*i+3,3:] = np.eye(3)
            
            # Create difference matrix
            b[3*i:3*i+3] = pt_cloud_b[i] - pt_cloud_a[i]

        # Solve for alpha and epsilon using the least-squares method
        alpha_eps = np.linalg.lstsq(M, b, rcond=None)[0]

        # Use alpha and epsilon to construct an affine transformation matrix
        F = np.eye(4)
        F[:3,:3] = self.rotation_matrix(alpha_eps[:3])
        F[:3,3] = alpha_eps[3:]

        # Pad point cloud matrix to create homogenous coordinates if necessary
        if pt_cloud_a.shape[1] == 3:
            homog = np.ones((pt_cloud_a.shape[0], 1))
            pt_cloud_a = np.concatenate([pt_cloud_a, homog], axis=1)
            pt_cloud_b = np.concatenate([pt_cloud_b, homog], axis=1)

        # Compute mean absolute error for (x, y, z)
        err = np.mean(np.abs(pt_cloud_b.T - F @ pt_cloud_a.T), axis=1)
        
        return F, err[0:3]
    
    """
    Creates a skew-symmetric matrix of a 3D vector.
    """
    def skew_symmetric(self, vec: NDArray[np.float32]) -> NDArray[np.float32]:
        return np.array([
            [0, -vec[2], vec[1]],
            [vec[2], 0, -vec[0]],
            [-vec[1], vec[0], 0]
        ])
    
    """
    Creates a rotation matrix from a 3D vector of angles.
    """
    def rotation_matrix(self, vec: NDArray[np.float32]) -> NDArray[np.float32]:
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(vec[0]), -np.sin(vec[0])],
            [0, np.sin(vec[0]), np.cos(vec[0])]
        ])

        Ry = np.array([
            [np.cos(vec[1]), 0, np.sin(vec[1])],
            [0, 1, 0],
            [-np.sin(vec[1]), 0, np.cos(vec[1])]
        ])

        Rz = np.array([
            [np.cos(vec[2]), -np.sin(vec[2]), 0],
            [np.sin(vec[2]), np.cos(vec[2]), 0],
            [0, 0, 1]
        ])

        return Rx @ Ry @ Rz

