import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple, Iterator, Union

"""
Code created by Edmund Sumpena and Jayden Ma
"""

class BoundingBox():
    def __init__(
        self,
        min_xyz: NDArray[np.float32],
        max_xyz: NDArray[np.float32]
    ) -> None:
        """
        Stores a bounding box by its minimum and maximum (x, y, z) corners.
        """

        # Top-left coordinate should be a 3D vector of (x, y, z) coordinates
        if len(min_xyz.shape) != 1 or min_xyz.shape[0] != 3:
            raise ValueError('Minimum corner coordinate should be a 3D vector of (x, y, z) coordinates!')
        
        # Bottom-left coordinate should be a 3D vector of (x, y, z) coordinates
        if len(max_xyz.shape) != 1 or max_xyz.shape[0] != 3:
            raise ValueError('Maximum corner coordinate should be a 3D vector of (x, y, z) coordinates!')

        # Stores the top left and bottom right corners of the bounding box
        self.min_xyz: NDArray[np.float32] = min_xyz
        self.max_xyz: NDArray[np.float32] = max_xyz

    def contains(self, points: NDArray[np.float32]) -> NDArray[np.bool8]:
        """
        Determines whether points, given as an Nx3 matrix, are contained within 
        or on the edge of the bounding box.
        """

        # Input should be an Nx3 matrix of points
        if len(points.shape) != 2 or points.shape[1] != 3:
            raise ValueError('Points should be an Nx3 matrix!')
        
        # Finds which points are between the minimum and maximum values
        # for each coordinate axis
        bounds = (self.min_xyz <= points) & (points <= self.max_xyz)
        bounds = np.all(bounds, axis=1)
        
        return bounds
    
    def enlarge(self, growth: Union[float, NDArray[np.float32]]) -> None:
        """
        Increases the size of the bounding box by a constant factor.
        """
        
        if isinstance(growth, np.ndarray):
            # If growth is vector, then it should be a 3D vector
            if self.min_xyz.shape != growth.shape or growth.shape[0] != 3 or \
                len(growth.shape) != 1:
                raise ValueError(f'Growth factor should be 3D vector but got shape {growth.shape}!')
            
        elif not isinstance(growth, float):
            # Growth should be either float or vector
            raise ValueError('Growth factor should be a float or 3D vector!')

        self.min_xyz -= growth
        self.max_xyz += growth

    def overlaps(self, bounding_box: 'BoundingBox') -> bool:
        """
        Determines if the bounding box overlaps with another bounding box.
        """

        # Checks if the x-axis, y-axis, and z-axis overlaps
        if (self.min_xyz < bounding_box.max_xyz).any() or \
            (bounding_box.min_xyz < self.max_xyz).any():
            # One of the axes do not overlap
            return False
        
        # All axes overlap
        return True

class Triangle():
    def __init__(
        self,
        v1: NDArray[np.float32],
        v2: NDArray[np.float32],
        v3: NDArray[np.float32],
        idx1: Union[float, None] = None,
        idx2: Union[float, None] = None,
        idx3: Union[float, None] = None,
    ) -> None:
        """
        Stores a triangle as a set of vertices.
        """
        
        # Set triangle vertices
        self.set_vertices(v1, v2, v3, idx1, idx2, idx3)

    def set_vertices(
        self,
        v1: NDArray[np.float32],
        v2: NDArray[np.float32],
        v3: NDArray[np.float32],
        idx1: Union[float, None] = None,
        idx2: Union[float, None] = None,
        idx3: Union[float, None] = None,
    ) -> None:
        """
        Sets the vertices of the triangle.
        """

        # Triangle vertices should be a 3D vector of (x, y, z) coordinates
        if len(v1.shape) != 1 or v1.shape[0] != 3:
            raise ValueError(f'Expected v1 to be a 3D vector of (x, y, z) coordinates but got {v1.shape}.')
        
        if len(v2.shape) != 1 or v2.shape[0] != 3:
            raise ValueError(f'Expected v2 to be a 3D vector of (x, y, z) coordinates but got {v2.shape}.')
        
        if len(v3.shape) != 1 or v3.shape[0] != 3:
            raise ValueError(f'Expected v3 to be a 3D vector of (x, y, z) coordinates but got {v3.shape}.')
        
        # Save triangle vertices
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3

        # Save indexes of triangle vertices
        if idx1 is not None:
            self.idx1 = idx1
        
        if idx2 is not None:
            self.idx2 = idx2

        if idx3 is not None:
            self.idx3 = idx3

    def center(self) -> NDArray[np.float32]:
        """
        Finds the center point of the triangle based on its vertices.
        """

        return (self.v1 + self.v2 + self.v3) / 3.0

    def box(self) -> BoundingBox:
        """
        Computes a bounding box for the triangle based on its vertices.
        """

        # Stack vertices into a 3x3 matrix
        vertices = np.stack([self.v1, self.v2, self.v3], axis=0)

        # Compute coordinates of the top-left and bottom-right corners 
        # of the bounding box
        top_left = np.min(vertices, axis=0)
        bottom_right = np.max(vertices, axis=0)

        return BoundingBox(top_left, bottom_right)

    def closest_distance_to(
        self,
        points: NDArray[np.float32]
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
        """
        Computes the closest distance from each point in an Nx3 matrix
        to the triangle by solving a constrained least-squares problem.
        Returns an N-dimensional vector of the closest distance to 
        the triangle and an Nx3 matrix of the closest point on the triangle.
        """

        # Input should be an Nx3 matrix of points
        if len(points.shape) != 2 or points.shape[1] != 3:
            raise ValueError('Points should be an Nx3 matrix!')

        # Construct 3x2 matrix of edge vectors in Barycentric form
        # Approach based on Dr. Taylor's slides on finding closest points:
        # https://ciis.lcsr.jhu.edu/lib/exe/fetch.php?media=courses:455-655:lectures:finding_point-pairs.pdf
        edge1 = self.v2 - self.v1
        edge2 = self.v3 - self.v1
        A = np.stack([edge1, edge2], axis=1)

        # Compute vector b for all points (point - v1)
        c_points = (points - self.v1).T

        # Solve the least squares problem and get Nx3 matrix of v, λ, μ
        b = np.linalg.lstsq(A, c_points, rcond=None)[0]
        b = np.concatenate([(1. - b[0] - b[1]).reshape(1, -1), b], axis=0)

        # Construct 3x3 matrix of vertices
        M = np.array([
            self.v1,
            self.v2,
            self.v3
        ]).T

        # Find the point projected onto the plane of the triangle
        closest_pt = (M @ b).T[:,:3]

        # Find point that is outside the λ, μ, v constraints
        λ_constraint = b[0] < 0
        μ_constraint = b[1] < 0
        v_constraint = b[2] < 0

        # Handle the boundary cases where λ < 0
        if λ_constraint.any():
            closest_pt[λ_constraint] = self._project_on_seg(
                points[λ_constraint].reshape(-1, 3), self.v2, self.v3
            )

        # Handle the boundary cases where μ < 0
        if μ_constraint.any():
            closest_pt[μ_constraint] = self._project_on_seg(
                points[μ_constraint].reshape(-1, 3),self.v3, self.v1
            )
        
        # Handle the boundary cases where v < 0
        if v_constraint.any():
            closest_pt[v_constraint] = self._project_on_seg(
                points[v_constraint].reshape(-1, 3), self.v1, self.v2
            )

        # Calculate distances from points to their closest points on the triangle
        distances = np.linalg.norm(closest_pt - points, axis=1)

        return distances, closest_pt

    def _project_on_seg(
        self,
        c: NDArray[np.float32],
        p: NDArray[np.float32],
        q: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """
        Given an set of points c as an Nx3 matrix, find the projection
        of the closest point onto the line segment pq, where p and q are
        3D vectors of (x, y, z) coordinates.
        """

        # c should be an Nx3 matrix of points
        if len(c.shape) != 2 or c.shape[1] != 3:
            raise ValueError('c should be an Nx3 matrix!')
        
        # p should be a 3D vector
        if len(p.shape) != 1 or p.shape[0] != 3:
            raise ValueError(f'p should be a 3D vector of (x, y, z) coordinates!')
        
        # q should be a 3D vector
        if len(q.shape) != 1 or q.shape[0] != 3:
            raise ValueError(f'q should be a 3D vector of (x, y, z) coordinates!')

        # Compute the degree to which segment pc aligns with pc*,
        # where c* are the points c projected onto pq
        λ = c.reshape(-1, 3) - p.reshape(1, 3)
        λ = λ @ (q - p).reshape(3, 1)

        # Normalize by the degree which the segment pq aligns with pq
        λ /= np.dot(q - p, q - p)
        λ = np.maximum(0.0, np.minimum(λ, 1.0))

        # Compute c*, the projection of c onto pq
        c_star = p + λ * (q - p)

        return c_star
    
    def __repr__(self) -> str:
        """
        Define Triangle object as a string representation
        """

        # Print out triangle vertices as string
        return str((self.v1, self.v2, self.v3))
        
class Meshgrid():
    def __init__(
        self,
        vertices: NDArray[np.float32],
        triangle_indices: NDArray[np.float32]
    ) -> None:
        """
        Stores a meshgrid as a set of vertices and Triangles.
        """
        
        # Vertices should be an Nx3 matrix of (x, y, z) coordinates
        if len(vertices.shape) != 2 or vertices.shape[1] != 3:
            raise ValueError('Vertices should be provided as an Nx3 matrix!')
        
        # Triangle indices should be a Tx3 matrix of indices cooresponding
        # to the three vertices of the triangle
        if len(triangle_indices.shape) != 2 or triangle_indices.shape[1] != 3:
            raise ValueError('Triangle indices should be provided as a Tx3 matrix!')

        # Save all vertices in meshgrid as an Nx3 matrix
        self.vertices: NDArray[np.float32] = vertices

        # Save triangle vertex indices as a Tx3 matrix
        self.triangle_indices: NDArray[np.float32] = triangle_indices

        # Save triangles as a list of Triangle objects
        self.triangles: List[Triangle] = []

        # Construct Triangles and add them to the list
        for i in range(self.triangle_indices.shape[0]):
            # Extract the vertices as (x, y, z) coordinates
            idx1, idx2, idx3 = self.triangle_indices[i]
            v1, v2, v3 = self.vertices[self.triangle_indices[i]]

            self.triangles.append(
                Triangle(v1, v2, v3, idx1, idx2, idx3)
            )

    def __iter__(self) -> Iterator[Triangle]:
        """
        Creates an iterator for accessing all the triangles in
        the meshgrid.
        """

        # Returns iterator from list of Triangles
        return self.triangles.__iter__()
