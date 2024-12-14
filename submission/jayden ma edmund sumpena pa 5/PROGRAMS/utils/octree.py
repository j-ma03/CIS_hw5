import numpy as np
from numpy.typing import NDArray
from typing import List, Dict, Iterator
from utils.meshgrid import Triangle, BoundingBox

"""
Code created by Edmund Sumpena and Jayden Ma
"""

class Octree:
    def __init__(
        self,
        elements: List[Triangle],
        min_count: int = 1,
        min_diag: float = 0.1
    ) -> None:
        """
        Octree class is based on Dr. Taylor's lecture on finding
        point pairs:

        https://ciis.lcsr.jhu.edu/lib/exe/fetch.php?media=courses:455-655:lectures:finding_point-pairs.pdf
        """
        
        # Save Triangle elements
        self.elements: List[Triangle] = elements
        self.num_elements: int = len(elements)

        # Compute centers and find minimum and maximum corners
        self.min_xyz: NDArray[np.float32] = []
        self.max_xyz: NDArray[np.float32] = []

        for i in range(len(self.elements)):
            box = self.elements[i].box()
            self.min_xyz.append(box.min_xyz)
            self.max_xyz.append(box.max_xyz)

        self.min_xyz = np.array(self.min_xyz)
        self.max_xyz = np.array(self.max_xyz)

        # Save bounding box of the triangle centers and compute 
        self.bb: BoundingBox = self.box()
        self.cb: BoundingBox = self.centroid_box()
        self.center: NDArray[np.float32] = (self.cb.min_xyz + self.cb.max_xyz) / 2.0
        self.have_subtrees: bool = False
        self.subtrees: List[Octree] = None

        # Build subtrees if necessary
        self.construct_subtrees(min_count, min_diag)

    def centroid_box(self) -> BoundingBox:
        """
        Computes the bounding box of the Triangle center points.
        """

        # Compute centers and find minimum and maximum corners
        centers = np.array([ele.center() for ele in self.elements])

        # No bounding box if there are no elements in tree
        if centers.size == 0:
            return None
        
        box = BoundingBox(np.min(centers, axis=0), np.max(centers, axis=0))

        return box
    
    def box(self) -> BoundingBox:
        """
        Computes the bounding box of the Triangle inner and outer corners.
        """

        # No bounding box if there are no elements in tree
        if self.min_xyz.size == 0 or self.max_xyz.size == 0:
            return None
        
        box = BoundingBox(np.min(self.min_xyz, axis=0), np.max(self.max_xyz, axis=0))

        return box
    
    def construct_subtrees(
        self,
        min_count: int,
        min_diag: float
    ) -> None:
        """
        Construct subtrees recursively by splitting elements into eight regions.
        """

        # Stop constructing subtrees if too few elements or the 
        # split diagonal is too small
        if self.num_elements <= min_count or \
            np.linalg.norm(self.bb.min_xyz - self.bb.max_xyz) <= min_diag:
            self.have_subtrees = False
            return
        
        # Add subtrees
        self.have_subtrees = True
        
        # Split and sort the things into 8 subtrees based on the center
        partitions = self.split_sort(self.center)

        # Construct subtrees for each partition
        self.subtrees = [ Octree(partitions[k]) for k in partitions if len(partitions[k]) > 0 ]
    
    def split_sort(self, splitting_point: NDArray[np.float32]) -> Dict[List[bool], int]:
        """
        Split the elements into octants based on comparison with the splitting point.
        This method updates the nnn, npn, etc., counts for each octant.
        """

        # Create empty partion as dictionary
        partitions = self.create_partitions()

        # Iterate through all elements 
        for ele in self.elements:
            # Extract center point of the element
            center = ele.center()

            # Add element to the respective parititons
            octant = (center >= splitting_point).tolist()
            partitions[tuple(octant)].append(ele)

        return partitions

    def create_partitions(self) -> Dict[List[bool], List[Triangle]]:
        """
        Creates eight partitions for the Octree as a Dictionary, which 
        contains the octants represented by a 3D list of booleans.
        """

        partitions = {}

        # Create the 8 quadrants represented by a 3D list of booleans
        # where True represents + and False represents -
        for x in [True, False]:
            for y in [True, False]:
                for z in [True, False]:
                    partitions[(x, y, z)] = []

        return partitions
    
    def __iter__(self) -> Iterator['Octree']:
        """
        Creates an iterator for accessing all the subtrees in
        the Octree.
        """

        # Returns iterator from list of subtrees
        return self.subtrees.__iter__()

