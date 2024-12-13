import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Any
from tqdm import tqdm
from utils.icp import IterativeClosestPoint, Matching
from utils.meshgrid import Meshgrid, Triangle
import warnings

"""
Code created by Edmund Sumpena and Jayden Ma
"""

class DeformableICP(IterativeClosestPoint):
    def __init__(
        self,
        max_iter: int = 200,
        match_mode: Matching = Matching.VECTORIZED_LINEAR,
        gamma: float = 0.96,
        early_stopping: int = 10
    ) -> None:
        super().__init__(max_iter, match_mode, gamma, early_stopping)

    def __call__(
        self,
        pt_cloud: NDArray[np.float32],
        meshgrid: Meshgrid,
        modes: NDArray[np.float32]
    ):
        """
        Runs the full deformable ICP algorithm given a point cloud,
        meshgrid, and modes.
        """

        # List storing point cloud and meshgrid closest point
        # matching error
        match_score = [np.inf]

        # Point cloud should be an Nx3 matrix of (x, y, z) coordinates
        if len(pt_cloud.shape) != 2 or pt_cloud.shape[1] != 3:
            raise ValueError('Point cloud should be an Nx3 matrix containing 3D coordinates!')

        # Input meshgrid should be a Meshgrid object
        if not isinstance(meshgrid, Meshgrid):
            raise Exception(f'Expected input meshgrid should be of type Meshgrid but got \'{meshgrid.__class__.__name__}\'!')
        
        # The matrix of modes should be a MxVx3 matrix
        if len(modes.shape) != 3 or modes.shape[2] != 3:
            raise ValueError('Matrix of modes should be an MxVx3 matrix!')
        
        # Initialize the best rigidly-transformed point cloud
        # and its corresponding transformation matrix
        pt_cloud_best = pt_cloud[:,:3].copy()
        F_best = np.eye(4)
        
        # Initialize λ vector as a zero vector (just look at mean initially)
        # λ_best = np.zeros(modes.shape[0])
        λ_best = np.random.randn(modes.shape[0])
        λ_best[0] = 1.0
        
        for i in range(self.max_iter):
            print(f'Iteration {i+1}:')
            
            ####
            ## Step 1: Perform rigid-body ICP until convergence
            ####

            best_pt_cloud, closest_pt, dist, F_i = super().__call__(
                pt_cloud_best, meshgrid
            )
            F_best = F_best @ F_i

            ####
            ## Step 2: Deform the meshgrid until convergence
            ####

            meshgrid, λ_best = self.deform_mesh(
                best_pt_cloud, meshgrid, modes, λ_best
            )

            ####
            ## Step 3: Compute error and check for terminatation condition
            ####

            _, _, epsilon = self._residual_error(pt_cloud, closest_pt)
            match_score.append(epsilon)

            # Compute the ratio reflecting the change in point cloud
            # to meshgrid matching score from the previous iteration
            match_ratio = match_score[-1] / match_score[-2]
            print(f'\nOverall Match Error: {epsilon:0.3f}')
            print(f'Overall Match Ratio: {match_ratio:0.3f}\n')

            # Check if the match ratio fails the termination condition
            if match_ratio >= self.gamma:
                print(f'Match ratio >= gamma ({self.gamma:0.3f}). Terminating algorithm.\n')
                break

        # Return the best point cloud, closest distance to mesh,
        # and best λ vector
        return best_pt_cloud, closest_pt, dist, F_best, λ_best

    def deform_mesh(
        self,
        pt_cloud: NDArray[np.float32],
        meshgrid: Meshgrid,
        modes: NDArray[np.float32],
        λ: NDArray[np.float32]
    ) -> Tuple[Meshgrid, NDArray[np.float32]]:
        """
        Returns the new deformed meshgrid and mode weights λ.
        """
        # List storing point cloud and meshgrid closest point
        # matching error
        match_score = [np.inf]

        # Stores the maximum closest distance threshold for point cloud
        # and meshgrid closest point to be considered a candidate
        dist_thresh = np.inf
        early_stop_count = 0    # Number of times failed termination condition

        # Store the best λ (mode weights) and deformed mesh
        λ_best = λ.copy()
        mesh_best = meshgrid.copy()

        for _ in (pbar := tqdm(range(self.max_iter), bar_format="Deforming Mesh: {n_fmt}it [{elapsed}<{remaining}, {rate_fmt}{postfix}]")):
            # Find closest points and distances
            closest_pt, dist, closest_tri = super().match(pt_cloud, meshgrid)

            # Find candidates where distance to closest point is
            # less than the maximum threshold
            candidates = np.where(dist < dist_thresh)[0]

            # Compute mode coordinates based on current prediction
            mode_coords = self._compute_mode_coordinates(
                closest_pt[candidates], closest_tri[candidates],
                modes, λ
            )

            # Predict new mode weights λ to improve deformation
            λ_new = self._get_deformable_transf(
                pt_cloud[candidates], modes, mode_coords
            )

            # Update mode weights λ for modes 1 - M
            λ[1:] = λ_new
            assert λ[0] == 1, f'Expected λ for the 0th mode to be 1, but got {λ[0]}!'

            # Apply deformation on the triangle mesh vertices
            meshgrid = self._deform_mesh(meshgrid, modes, λ)

            # compute sigma = residual error between A and B
            # compute epsilon max = maximum residual error between A and B             
            # compute epsilon = residual error between A and B; append to match score
            _, _, epsilon = self._residual_error(pt_cloud, closest_pt)
            match_score.append(epsilon)

            # Update distance threshold to three-times the similarity
            # score between the point cloud and closest meshgrid points
            dist_thresh = 3 * match_score[-1]
            
            # Compute the ratio reflecting the change in point cloud
            # to meshgrid matching score from the previous iteration
            match_ratio = match_score[-1] / match_score[-2]
            
            # Display match ratio on progress bar
            pbar.set_postfix(match=match_score[-1], prev_match_ratio=match_ratio)
            pbar.update(0)

            # Check if the match ratio fails the termination condition
            if match_ratio >= self.gamma:
                early_stop_count += 1   # Increment counter
            else:
                λ_best = λ.copy()   # New best deformation found
                mesh_best = meshgrid.copy()
                early_stop_count = 0    # Reset counter

            # Stop algorithm if failed termination condition
            # too many times
            if early_stop_count >= self.early_stopping:
                break

        return mesh_best, λ_best
    
    def _compute_mode_coordinates(
        self,
        closest_pt: NDArray[np.float32],
        closest_tri: NDArray[Any],
        modes: NDArray[np.float32],
        λ: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """
        Takes in a set of closest points as an Nx3 matrix, their 
        corresponding triangles as a N-dimensional array, a matrix
        of modes as a MxVx3, and array of weights (λ) as a 
        N-dimensional vector. Compute barycentric coordinates on 
        the triangle in terms of mode coordinates.
        """

        # Check that matrix of closest points is Nx3
        if len(closest_pt.shape) != 2 or closest_pt.shape[1] != 3:
            raise ValueError('Closest points should be an Nx3 matrix!')
        
        # Check that closest triangles is an N-dimensional array
        if len(closest_tri.shape) != 1:
            raise ValueError('Closest triangles should be an N-dimension array!')
        
        # Check that number of closest points match the 
        # number of closest triangles
        if closest_tri.shape[0] != closest_pt.shape[0]:
            raise ValueError(f'Expected array of closest triangles to be length {closest_pt.shape[0]} '\
                             f'to match the closest points matrix, but got {len(closest_tri)}!')
        
        # Check that the matrix of modes is a MxVx3 matrix
        if len(modes.shape) != 3 or modes.shape[2] != 3:
            raise ValueError('Matrix of modes should be an MxVx3 matrix!')
        
        # Check that λ is a M-dimensional vector
        if len(λ.shape) != 1:
            raise ValueError('Your λ should be an M-dimension vector!')
        
        # Check that the number of modes match the number of λ values
        if λ.shape[0] != modes.shape[0]:
            raise ValueError(f'Expected vector of λ to be {modes.shape[0]} '\
                             f'to match the modes matrix, but got {λ.shape[0]}!')
        
        # Check that λ of mode 0 (the mean) is 1
        if λ[0] != 1:
            warnings.warn('Your λ of the 0th mode (mean) should be 1. '\
                          'Automatically setting λ[0] to 1.')
            λ[0] = 1.0

        # Create an NxMx3 matrix of mode coordinates
        Q = np.zeros((closest_pt.shape[0], modes.shape[0], 3))

        for i in range(len(closest_tri)):
            triangle: Triangle = closest_tri[i]

            # Extract triangle vertex indices
            s = triangle.idx1
            t = triangle.idx2
            u = triangle.idx3

            # Compute mode coordinates of the deformed mesh
            mode_coords = modes[:,[s,t,u]] * λ.reshape(-1, 1, 1)
            mode_coords = np.sum(mode_coords, axis=0).T
            mode_coords = np.concatenate([mode_coords, [[1, 1, 1]]], axis=0)

            # Find barycentric coordinates by solving a least squares problem
            barycentric = np.linalg.lstsq(mode_coords, [*closest_pt[i], 1], rcond=None)[0]
            barycentric = barycentric.reshape(1, 3, 1)

            # Store mode coordinates in a NxMx3 matrix
            Q[i] = np.sum(modes[:,[s,t,u]] * barycentric, axis=1)

        return Q

    def _get_deformable_transf(
        self,
        pt_cloud: NDArray[np.float32], # transformed point cloud s_k
        modes: NDArray[np.float32], # modes q_m,k
        mode_coords: NDArray[np.float32] # mode coordinates q_m,k
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
        """
        Computes the rigid body transformation for a point cloud
        and the mode weights λ given a meshgrid and modes.
        """
        # Point cloud should be an Nx3 matrix of (x, y, z) coordinates
        if len(pt_cloud.shape) != 2 or pt_cloud.shape[1] != 3:
            raise ValueError('Point cloud should be an Nx3 matrix containing 3D coordinates!')
        
        # Check that the matrix of modes is a MxVx3 matrix
        if len(modes.shape) != 3 or modes.shape[2] != 3:
            raise ValueError('Matrix of modes should be an MxVx3 matrix!')

        # number of points in the point cloud
        n = pt_cloud.shape[0]

        # Initialize A and b matrices
        A = np.zeros((3 * n, modes.shape[0] - 1))
        b = (pt_cloud - mode_coords[:,0]).flatten()  # b_k = s_k - c_k

        for i in range(n): # populate A and b
            # each 3xn row in A = [q_1,k ... q_m,k], repeat for each point in the point cloud
            A[3*i:3*i+3] = mode_coords[i,1:].T

        # Solve least squares problem A * x = b
        x = np.linalg.lstsq(A, b, rcond=None)[0]

        # Return lambda vector (mode weights)
        return x
    
    def _deform_mesh(
        self,
        meshgrid: Meshgrid,
        modes: NDArray[np.float32],
        λ: NDArray[np.float32]
    ) -> None:
        
        # Check that the matrix of modes is a MxVx3 matrix
        if len(modes.shape) != 3 or modes.shape[2] != 3:
            raise ValueError('Matrix of modes should be an MxVx3 matrix!')
        
        # Check that λ is a M-dimensional vector
        if len(λ.shape) != 1:
            raise ValueError('Your λ should be an M-dimension vector!')
        
        # Check that the number of modes match the number of λ values
        if λ.shape[0] != modes.shape[0]:
            raise ValueError(f'Expected vector of λ to be {modes.shape[0]} '\
                             f'to match the modes matrix, but got {λ.shape[0]}!')
        
        # Check that λ of mode 0 (the mean) is 1
        if λ[0] != 1:
            warnings.warn('Your λ of the 0th mode (mean) should be 1. '\
                          'Automatically setting λ[0] to 1.')
            λ[0] = 1.0
        
        for i in range(len(meshgrid.triangles)):
            triangle: Triangle = meshgrid.triangles[i]

            # Extract triangle vertex indices
            s = triangle.idx1
            t = triangle.idx2
            u = triangle.idx3

            # Compute mode coordinates of the deformed mesh
            mode_coords = modes[:,[s,t,u]] * λ.reshape(-1, 1, 1)
            mode_coords = np.sum(mode_coords, axis=0)

            # Update the triangle with the deformed vertices
            meshgrid.triangles[i].set_vertices(
                mode_coords[0],
                mode_coords[1],
                mode_coords[2]
            )

        return meshgrid
    
    def _residual_error(
        self, 
        pt_cloud: NDArray[np.float32],
        closest_pt: NDArray[np.float32]
    )-> Tuple[float, float, float]:
        """
        Computes the residual error terms between the transformed point cloud and meshgrid
        closest points.
        """

        num_points = pt_cloud.shape[0] # NumElts(E)

        res_error = closest_pt - pt_cloud # e_k = b_k - F * a_k
        res_error_squared = np.sum(res_error * res_error, axis=1) # e_k * e_k

        # compute sigma = residual error between A and B
        # compute epsilon max = maximum residual error between A and B             
        # compute epsilon = residual error between A and B
        sigma = np.sqrt(np.sum(res_error_squared)) / num_points
        epsilon_max = np.sqrt(np.max(res_error_squared))
        epsilon = np.sum(np.sqrt(res_error_squared)) / num_points

        return sigma, epsilon_max, epsilon
    
    def _homogenize(self, pt_cloud: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Converts a point cloud stored in an Nx3 matrix of (x, y, z) points 
        into homogeneous coordinates.
        """

        # Point cloud inputs must be in format BxNx3 (B = batch size, N = number of points)
        if len(pt_cloud.shape) != 2 or not 3 <= pt_cloud.shape[1] <= 4:
            raise Exception(f'Point clouds must have shape Nx3 or Nx4!')

        # Pad point cloud matrix to create homogenous coordinates if necessary
        if pt_cloud.shape[1] == 3:
            homog = np.ones((pt_cloud.shape[0], 1))
            return np.concatenate([pt_cloud, homog], axis=1)
        else:
            # Point cloud is 
            return pt_cloud
        

    def _skew_symmetric(self, vec: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Creates a skew-symmetric matrix of a 3D vector.
        """

        if len(vec.shape) != 1 or vec.shape[0] != 3:
            raise ValueError('Input should be a 3D vector of (x, y, z)!')

        return np.array([
            [0, -vec[2], vec[1]],
            [vec[2], 0, -vec[0]],
            [-vec[1], vec[0], 0]
        ])
    
    def _rotation_matrix(self, vec: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Creates a rotation matrix from a 3D vector of angles.
        """
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

        
    