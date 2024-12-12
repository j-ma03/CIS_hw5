import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Any
from enum import Enum
from tqdm import tqdm
from utils.coordinate_calibration import PointCloudRegistration
from utils.icp import IterativeClosestPoint
from utils.meshgrid import Meshgrid, Triangle
from utils.octree import Octree
import warnings

"""
Code created by Edmund Sumpena and Jayden Ma
"""

class Matching(Enum):
    SIMPLE_LINEAR = 1
    VECTORIZED_LINEAR = 2
    SIMPLE_OCTREE = 3
    VECTORIZED_OCTREE = 4

class DeformableICP(IterativeClosestPoint):
    def __init__(
        self,
        max_iter: int = 200,
        match_mode: Matching = Matching.VECTORIZED_LINEAR,
        gamma: float = 0.95,
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

        # Point cloud should be an Nx3 matrix of (x, y, z) coordinates
        if len(pt_cloud.shape) != 2 or pt_cloud.shape[1] != 3:
            raise ValueError('Point cloud should be an Nx3 matrix containing 3D coordinates!')

        # Input meshgrid should be a Meshgrid object
        if not isinstance(meshgrid, Meshgrid):
            raise Exception(f'Expected input meshgrid should be of type Meshgrid but got \'{meshgrid.__class__.__name__}\'!')
        
        # The matrix of modes should be a MxVx3 matrix
        if len(modes.shape) != 3 or modes.shape[2] != 3:
            raise ValueError('Matrix of modes should be an MxVx3 matrix!')
        
        ####
        ## Step 1: Perform rigid-body ICP until convergence
        ####

        print('\nPerforming rigid-body ICP...')

        best_pt_cloud, closest_pt, dist, F_best = super().__call__(
            pt_cloud, meshgrid
        )

        print('Done.\n')


        ####
        ## Step 2: Refine transformation with Deformable ICP until convergence
        ####

        # List storing point cloud and meshgrid closest point
        # matching error
        match_score = [np.inf]

        # Stores the maximum closest distance threshold for point cloud
        # and meshgrid closest point to be considered a candidate
        dist_thresh = np.inf

        # Initialize λ vector as a zero vector (just look at mean initially)
        λ = np.zeros(modes.shape[0])
        λ[0] = 1.0
        λ_best = λ.copy()

        # Stores the current and best transformation from
        # the point cloud to meshgrid
        F: NDArray = F_best.copy()
        early_stop_count = 0    # Number of times failed termination condition

        # Stores the best point cloud computed at each iteration
        pt_cloud_i = best_pt_cloud[:,:3].copy()
        pt_cloud = self._homogenize(pt_cloud)

        print('\nPerforming deformable ICP...')
        for _ in (pbar := tqdm(range(self.max_iter), bar_format="{n_fmt}it [{elapsed}<{remaining}, {rate_fmt}{postfix}]")):
            # Find closest points and distances
            closest_pt, dist, closest_tri = self.match(pt_cloud_i, meshgrid)

            # Find candidates where distance to closest point is
            # less than the maximum threshold
            candidates = np.where(dist < dist_thresh)[0]

            # Compute mode coordinates based on current prediction
            mode_coords = self._compute_mode_coordinates(
                closest_pt[candidates], closest_tri[candidates],
                modes, λ
            )

            # Compute updated rigid transformation (rotation ɑ & translation ε)
            # and mode weights λ
            alpha, eps, λ_new = self._get_deformable_transf(
                pt_cloud_i[candidates], closest_pt[candidates],
                modes, mode_coords
            )

            # Update mode weights λ for modes 1 - M
            λ[1:] = λ_new
            assert λ[0] == 1, f'Expected λ for the 0th mode to be 1, but got {λ[0]}.'

            # Compute new transformation matrix from alpha and epsilon
            F_i = np.eye(4)
            F_i[:3,:3] = self._rotation_matrix(alpha)
            F_i[3,:3] = eps

            # Update point cloud with transformation
            pt_cloud_i = self._homogenize(pt_cloud_i)
            pt_cloud_i = (F_i @ pt_cloud_i.T)[:3].T
            F = F @ F_i

            # Apply deformation on the triangle mesh vertices
            self._deform_mesh(meshgrid, modes, λ)

            # compute sigma = residual error between A and B
            # compute epsilon max = maximum residual error between A and B             
            # compute epsilon = residual error between A and B; append to match score
            sigma, epsilon_max, epsilon = self._residual_error(pt_cloud_i, closest_pt)
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
                F_best = F.copy()   # New best transformation found
                λ_best = λ.copy()
                early_stop_count = 0    # Reset counter

            # Stop algorithm if failed termination condition
            # too many times
            if early_stop_count >= self.early_stopping:
                break

        print('Done.')

        # Compute best point cloud and closest distance to mesh
        best_pt_cloud = (F_best @ pt_cloud.T).T[:,:3]
        closest_pt, dist, _ = self.match(best_pt_cloud[:,:3], meshgrid)

        return best_pt_cloud, closest_pt, dist, F_best, λ_best

    def match(
        self,
        pt_cloud: NDArray[np.float32],
        meshgrid: Meshgrid
    ) -> Tuple[NDArray[np.float32], ...]:
        """
        Finds the closest point and distance from points on a point cloud 
        to a triangle meshgrid.
        """

        # Point cloud should be an Nx3 matrix of (x, y, z) coordinates
        if len(pt_cloud.shape) != 2 or pt_cloud.shape[1] != 3:
            raise ValueError('Point cloud should be an Nx3 matrix containing 3D coordinates!')

        # Input meshgrid should be a Meshgrid object
        if not isinstance(meshgrid, Meshgrid):
            raise Exception(f'Expected input meshgrid should be of type Meshgrid but got \'{meshgrid.__class__.__name__}\'!')

        ####
        ## Search algorithms to find closest points
        ####

        # Performs a simple linear search for closest points
        if self.match_mode == Matching.SIMPLE_LINEAR:
            return self._simple_linear_match(pt_cloud, meshgrid)
        
        # Performs a faster vectorized linear search for closest points
        elif self.match_mode == Matching.VECTORIZED_LINEAR:
            return self._vectorized_linear_match(pt_cloud, meshgrid)
        
        # Performs a simple iterative search for closest points using Octrees
        elif self.match_mode == Matching.SIMPLE_OCTREE:
            return self._simple_octree_match(pt_cloud, meshgrid)
        
        # Performs a simple iterative search for closest points using Octrees
        elif self.match_mode == Matching.VECTORIZED_OCTREE:
            return self._vectorized_octree_match(pt_cloud, meshgrid)


    def test(
        self,
        pt_cloud: NDArray[np.float32],
        meshgrid: Meshgrid,
        modes: NDArray[np.float32]
    ) -> None:
        λ = np.random.rand(modes.shape[0])
        λ[0] = 1.0

        closest_pt, dist, closest_tri = self.match(pt_cloud, meshgrid)

        Q = self._compute_mode_coordinates( # q_m,k
            closest_pt, closest_tri, modes, λ
        )

        F_reg = np.eye(4)
        homog_cloud = self._homogenize(pt_cloud[:,:3])
        S = (F_reg @ homog_cloud.T)[:3].T

        # n = pt_cloud.shape[0]

        # A = np.zeros((3 * n, 6 + modes.shape[0]))
        # b = (S - closest_pt).flatten()    # b_k = s_k - c_k

        # for i in range(n): # populate A and b
        #     # TODO Construct least squares matrix
        #     skew_s_k = self.skew_symmetric(S[i]) # skew(s_k)
        #     # get q_k for each mode
        #     A[3*i:3*i+3, :3] = skew_s_k
        #     A[3*i:3*i+3, 3:6] = -np.eye(3)
        #     A[3*i:3*i+3, 6:] = Q[i].reshape(3, -1)

        # print(A.shape)
        # print(b.shape)

        # # TODO Solve least squares problem A * x = b
        # # ɑ, ε, λ
        # x = np.linalg.lstsq(A, b, rcond=None)[0]

        # print(x.shape)

        ɑ, ε, λ_new = self._get_deformable_transf(S, closest_pt, modes, Q)
        
        # return ⍺, ε, λ_new
        return λ, Q, ɑ, ε, λ_new
    
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

            # Find barycentric coordinates by solving a least squares problem
            barycentric = np.linalg.lstsq(mode_coords, closest_pt[i], rcond=None)[0]
            barycentric = barycentric.reshape(1, 3, 1)

            # Mode coordinates as a NxMx3
            Q[i] = np.sum(modes[:,[s,t,u]] * barycentric, axis=1)

        return Q
    
    def _get_deformable_transf(
        self,
        transf_pt_cloud: NDArray[np.float32], # transformed point cloud s_k
        closest_pt: NDArray[np.float32], # closest point on mesh c_k
        modes: NDArray[np.float32], # modes q_m,k
        mode_coords: NDArray[np.float32] # mode coordinates q_m,k
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
        """
        Computes the rigid body transformation for a point cloud
        and the mode weights λ given a meshgrid and modes.
        """
        # Point cloud should be an Nx3 matrix of (x, y, z) coordinates
        if len(transf_pt_cloud.shape) != 2 or transf_pt_cloud.shape[1] != 3:
            raise ValueError('Point cloud should be an Nx3 matrix containing 3D coordinates!')
        
        # Check that the matrix of modes is a MxVx3 matrix
        if len(modes.shape) != 3 or modes.shape[2] != 3:
            raise ValueError('Matrix of modes should be an MxVx3 matrix!')

        # number of points in the point cloud
        n = transf_pt_cloud.shape[0]

        # Initialize A and b matrices
        A = np.zeros((3 * n, 6 + modes.shape[0] - 1))
        b = (transf_pt_cloud - closest_pt).flatten()    # b_k = s_k - c_k

        for i in range(n): # populate A and b
            # each 3xn row in A = [skew(s_k) -I q_1,k ... q_m,k], repeat for each point in the point cloud
            A[3*i:3*i+3, :3] = self._skew_symmetric(transf_pt_cloud[i])
            A[3*i:3*i+3, 3:6] = -np.eye(3)
            A[3*i:3*i+3, 6:] = mode_coords[i,1:].T

        # Solve least squares problem A * x = b
        x = np.linalg.lstsq(A, b, rcond=None)[0]
        
        # Return alpha, epsilon, and lambdas
        return x[:3], x[3:6], x[6:]
    
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
            meshgrid.triangles[i].v1 = mode_coords[0]
            meshgrid.triangles[i].v2 = mode_coords[1]
            meshgrid.triangles[i].v3 = mode_coords[2]
    
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

        
    