import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Any
from enum import Enum
from tqdm import tqdm
from utils.coordinate_calibration import PointCloudRegistration
from utils.meshgrid import Meshgrid
from utils.octree import Octree

"""
Code created by Edmund Sumpena and Jayden Ma
"""

class Matching(Enum):
    SIMPLE_LINEAR = 1
    VECTORIZED_LINEAR = 2
    SIMPLE_OCTREE = 3
    VECTORIZED_OCTREE = 4

class IterativeClosestPoint():
    def __init__(
        self,
        max_iter: int = 200,
        match_mode: Matching = Matching.VECTORIZED_LINEAR,
        gamma: float = 0.95,
        early_stopping: int = 10
    ) -> None:
        # Define maximum number of ICP iterations
        self.max_iter: int = max_iter

        # Define the algorithm used to find closest points
        self.match_mode: Matching = match_mode

        # Define termination threshold
        self.gamma: float = gamma

        # Define early stopping counter defining the maximum
        # number of iterations
        self.early_stopping: int = early_stopping

    def __call__(
        self,
        pt_cloud: NDArray[np.float32],
        meshgrid: Meshgrid
    ):
        """
        Runs the full ICP algorithm given a point cloud and meshgrid.
        """

        # Point cloud should be an Nx3 matrix of (x, y, z) coordinates
        if len(pt_cloud.shape) != 2 or pt_cloud.shape[1] != 3:
            raise ValueError('Point cloud should be an Nx3 matrix containing 3D coordinates!')

        # Input meshgrid should be a Meshgrid object
        if not isinstance(meshgrid, Meshgrid):
            raise Exception(f'Expected input meshgrid should be of type Meshgrid but got \'{meshgrid.__class__.__name__}\'!')

        # List storing point cloud and meshgrid closest point
        # matching error
        match_score = [np.inf]

        # Stores the maximum closest distance threshold for point cloud
        # and meshgrid closest point to be considered a candidate
        dist_thresh = np.inf

        # Stores the current and best transformation from
        # the point cloud to meshgrid
        F: NDArray = np.eye(4)
        F_best: NDArray = np.eye(4)
        early_stop_count = 0    # Number of times failed termination condition

        # Initialize rigid registration helper class
        rigid_register = PointCloudRegistration(max_epochs=10, verbose=False)
        pt_cloud_i = pt_cloud[:,:3].copy()
        pt_cloud = self._homogenize(pt_cloud)

        for _ in (pbar := tqdm(range(self.max_iter), bar_format="{n_fmt}it [{elapsed}<{remaining}, {rate_fmt}{postfix}]")):
            # Find closest points and distances
            closest_pt, dist, _ = self.match(pt_cloud_i, meshgrid)

            # Find candidates where distance to closest point is
            # less than the maximum threshold
            candidates = np.where(dist < dist_thresh)[0]

            # Stop the algorithm if there are no candidates to consider
            if candidates.size == 0:
                break

            # Find the best rigid registration transformation and
            # update thre overall transformation
            F_i = rigid_register.register(
                pt_cloud_i[candidates][None,], closest_pt[candidates][None,]
            )[0]

            # Update point cloud with transformation
            pt_cloud_i = self._homogenize(pt_cloud_i)
            pt_cloud_i = (F_i @ pt_cloud_i.T)[:3].T
            F = F @ F_i

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
                early_stop_count = 0    # Reset counter

            # Stop algorithm if failed termination condition
            # too many times
            if early_stop_count >= self.early_stopping:
                break

        # Compute best point cloud and closest distance to mesh
        best_pt_cloud = (F_best @ pt_cloud.T).T[:,:3]
        closest_pt, dist, closest_tri = self.match(best_pt_cloud[:,:3], meshgrid)

        return best_pt_cloud, closest_pt, dist, closest_tri, F_best

    def match(
        self,
        pt_cloud: NDArray[np.float32],
        meshgrid: Meshgrid
    ):
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

    def _simple_linear_match(
        self,
        pt_cloud: NDArray[np.float32],
        meshgrid: Meshgrid
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
        """
        Implementation of a simple linear search for the closest point 
        containing loops over all data points and Triangles. Finds the 
        closest point and distance to the meshgrid.
        """

        # Populate a matrix of closest distances to the meshgrid
        min_dist = np.empty(pt_cloud.shape[0])
        min_dist.fill(np.inf)

        # Populate a matrix of closest points on the meshgrid
        closest_pt = np.zeros_like(pt_cloud)

        # Populate a list of triangles for each point
        closest_tri = np.empty(pt_cloud.shape[0], dtype=object)
        closest_tri.fill(None)

        # Iterate through all the points and triangles in the meshgrid
        for i, point in enumerate(pt_cloud):
            for triangle in meshgrid:
                # Extract the bounding box of the triangle
                box = triangle.box()

                # Extend the bounding box by a margin determined by the
                # current minimum distance from each point
                box.enlarge(min_dist[i])

                # Check if there are any candidates to consider
                if box.contains(point[None,]):
                    # Compute closest distance on the triangle for all candidates
                    dist, pt = triangle.closest_distance_to(point[None,])

                    # Find candidates where distance to triangle is less than
                    # the previously recorded minimum distance
                    if dist[0] < min_dist[i]:
                        # Update the closest point and minimum distance
                        closest_pt[i] = pt[0]
                        min_dist[i] = dist[0]
                        closest_tri[i] = triangle

        return closest_pt, min_dist, closest_tri
    
    def _vectorized_linear_match(
        self,
        pt_cloud: NDArray[np.float32],
        meshgrid: Meshgrid
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
        """
        Implementation of a fast vectorized linear search for the closest point
        containing a only single loop over all Triangles in the meshgrid.
        Closest point and distance to the meshgrid for all data points are updated
        at once for each Triangle.
        """

        # Populate a list of closest distances to the meshgrid
        min_dist = np.empty(pt_cloud.shape[0])
        min_dist.fill(np.inf)

        # Populate a matrix of closest points on the meshgrid
        closest_pt = np.zeros_like(pt_cloud)

        # Populate a list of triangles for each point
        closest_tri = np.empty(pt_cloud.shape[0], dtype=object)
        closest_tri.fill(None)

        # Iterate through all the triangles in the meshgrid
        for triangle in meshgrid:
            # Extract the bounding box of the triangle
            box = triangle.box()

            # Extend the bounding box by a margin determined by the
            # current minimum distance from each point
            expanded_min = box.min_xyz.reshape(1, 3) - min_dist.reshape(-1, 1)
            expanded_max = box.max_xyz.reshape(1, 3) + min_dist.reshape(-1, 1)

            # Identify candidate points within the bounding box
            candidates = np.all((expanded_min <= pt_cloud) & \
                                (pt_cloud <= expanded_max), axis=1)

            # Check if there are any candidates to consider
            if candidates.any():
                # Compute closest distance on the triangle for all candidates
                candidate_points = pt_cloud[candidates]
                dist, pt = triangle.closest_distance_to(candidate_points)

                # Find candidates where distance to triangle is less than
                # the previously recorded minimum distance
                closer_mask = dist < min_dist[candidates]

                # Select indices where new distances are closer from candidate indices
                indices = np.where(candidates)[0]
                closer_indices = indices[closer_mask]

                # Update the closest point and minimum distance
                min_dist[closer_indices] = dist[closer_mask]
                closest_pt[closer_indices] = pt[closer_mask]
                closest_tri[closer_indices] = triangle

        return closest_pt, min_dist, closest_tri
    
    def _simple_octree_match(
        self,
        pt_cloud: NDArray[np.float32],
        meshgrid: Meshgrid
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
        """
        Implementation of a simple iterative Octree search to find the
        closest point using a loop over all data points and elements of 
        the tree. Find the closest point and distance to the meshgrid for
        all data points.
        """

        # Populate a matrix of closest points on the meshgrid
        closest_pt = np.zeros_like(pt_cloud)
        
        # Populate a matrix of closest distances to the meshgrid
        min_dist = np.empty(pt_cloud.shape[0])
        min_dist.fill(np.inf)

        # Populate a list of triangles for each point
        closest_tri = np.empty(pt_cloud.shape[0], dtype=object)
        closest_tri.fill(None)

        # Create a new tree containing triangles from the meshgrid
        tree = Octree(meshgrid.triangles)

        # Check if tree is empty
        if tree.num_elements == 0:
            return closest_pt, min_dist
        
        # Search Octree for every point
        for i, pt in enumerate(pt_cloud):
            closest_pt[i], min_dist[i], closest_tri[i] = self._simple_search_octree(
                pt, tree, closest_pt[i], min_dist[i], closest_tri[i]
            )

        return closest_pt, min_dist
    
    def _simple_search_octree(
        self,
        point: NDArray[np.float32],
        tree: Octree,
        closest_pt: NDArray[np.float32],
        min_dist: float,
        closest_tri: NDArray[Any]
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
        """
        Helper function that recursively searches an Octree to
        find the closest point and distance to the meshgrid, which
        are stored as elements of the Octree.
        """
        
        # Point should be a 3D vector
        if len(point.shape) != 1 or point.shape[0] != 3:
            raise ValueError(f'Point should be a 3D vector of (x, y, z) coordinates!')
        
        # Check if tree is empty
        if tree.num_elements == 0:
            return closest_pt, min_dist
        
        # Get the bounding box of the tree
        box = tree.box()
        box.enlarge(min_dist)

        # Stop if there are no candidates to consider
        if not box.contains(point[None,]):
            return closest_pt, min_dist, closest_tri
        
        # Iterate through all elements of the subtree if node is a child
        if not tree.have_subtrees:
            for triangle in tree.elements:
                # Compute closest distance on the triangle for all candidates
                dist, pt = triangle.closest_distance_to(point[None,])

                if dist[0] < min_dist:
                    # Update the closest point and minimum distance
                    closest_pt = pt[0]
                    min_dist = dist[0]
                    closest_tri = triangle

            return closest_pt, min_dist, closest_tri

        # Recursively process all subtrees
        for subtree in tree:
            closest_pt, min_dist, closest_tri = self._simple_search_octree(
                point, subtree, closest_pt, min_dist, closest_tri
            )

        return closest_pt, min_dist, closest_tri
    
    def _vectorized_octree_match(
        self,
        pt_cloud: NDArray[np.float32],
        meshgrid: Meshgrid,
        tree: Octree = None,
        closest_pt: NDArray[np.float32] = None,
        min_dist: NDArray[np.float32] = None, 
        closest_tri: NDArray[Any] = None
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
        """
        Implementation of a vectorized Octree search to find the
        closest point with a single loop over the elements of 
        the tree. Find the closest point and distance to the meshgrid 
        for all data points.
        """

        # Populate a matrix of closest points on the meshgrid (if necessary)
        if closest_pt is None:
            closest_pt = np.zeros_like(pt_cloud)
        
        # Populate a matrix of closest distances to the meshgrid (if necessary)
        if min_dist is None:
            min_dist = np.empty(pt_cloud.shape[0])
            min_dist.fill(np.inf)

        # Populate a list of triangles for each point (if necessary)
        if closest_tri is None:
            closest_tri = np.empty(pt_cloud.shape[0], dtype=object)
            closest_tri.fill(None)

        # Create a new tree containing triangles from the meshgrid (if necessary)
        if tree is None:
            tree = Octree(meshgrid.triangles)

        # Check if tree is empty
        if tree.num_elements == 0:
            return closest_pt, min_dist, closest_tri
        
        # Get the bounding box of the tree
        box = tree.box()
        
        # Extend the bounding box by a margin determined by the
        # current minimum distance from each point
        expanded_min = box.min_xyz.reshape(1, 3) - min_dist.reshape(-1, 1)
        expanded_max = box.max_xyz.reshape(1, 3) + min_dist.reshape(-1, 1)

        # Identify candidate points within the bounding box
        candidates = np.all((expanded_min <= pt_cloud) & \
                            (pt_cloud <= expanded_max), axis=1)
        
        # Stop if there are no candidates to consider
        if not candidates.any():
            return closest_pt, min_dist

        closest_pt = closest_pt.copy()
        min_dist = min_dist.copy()
        
        # Iterate through all elements of the subtree if node is a child
        if not tree.have_subtrees:
            for triangle in tree.elements:
                # Compute closest distance on the triangle for all candidates
                candidate_points = pt_cloud[candidates]
                dist, pt = triangle.closest_distance_to(candidate_points)

                # Find candidates where distance to triangle is less than
                # the previously recorded minimum distance
                closer_mask = dist < min_dist[candidates]

                # Select indices where new distances are closer from candidate indices
                indices = np.where(candidates)[0]
                closer_indices = indices[closer_mask]

                # Update the closest point and minimum distance
                closest_pt[closer_indices] = pt[closer_mask]
                min_dist[closer_indices] = dist[closer_mask]
                closest_tri[closer_indices] = triangle

            return closest_pt, min_dist, closest_tri
        
        # Recursively process all subtrees
        for subtree in tree:
            closest_pt, min_dist, closest_tri = self._vectorized_octree_match(
                pt_cloud, meshgrid, subtree, closest_pt, min_dist, closest_tri
            )

        return closest_pt, min_dist, closest_tri
    
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

        
    