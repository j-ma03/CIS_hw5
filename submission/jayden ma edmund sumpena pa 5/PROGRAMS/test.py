from typing import Tuple
import unittest

from utils.dataloader import *
# from utils.icp import *
from utils.meshgrid import BoundingBox, Triangle, Meshgrid
from utils.deformable_icp import DeformableICP
import numpy as np

EPS = 0
DATA_DIR = './pa345_data'
OUTPUT_DIR = '../OUTPUT'

class TestClosestPoint(unittest.TestCase):  
    """
    tests the closest distance to a triangle from a point
    """  
    def test_closest_point_above_plane(self):
        # generate 3 random points that make up a triangle
        # generate random point p that lies on the plane of the triangle -- ground truth point
        # project a vector p' from p of random distance perpendicular to the plane of the triangle
        # find experimental closest point on triangle by calling closest_distance_to(p')
        # compare the two
        # assert that experimental closest point and p are the same

        v1 = np.random.rand(3)
        v2 = np.random.rand(3)
        v3 = np.random.rand(3)
        triangle = Triangle(v1, v2, v3) # create a triangle 

        # generate random point that lies on the plane of the triangle
        r1, r2 = np.random.rand(2)
        if r1 + r2 > 1:
            r1, r2 = 1 - r1, 1 - r2
        p = (1 - r1 - r2) * v1 + r1 * v2 + r2 * v3

        # get unit vector perpendicular to the plane of the triangle
        n = np.cross(v2 - v1, v3 - v1)
        n = n / np.linalg.norm(n)
        # project a vector p' from p of random distance perpendicular to the plane of the triangle
        distance = np.random.rand()
        p_prime = p + distance * n
        p_prime = p_prime.reshape(1, 3)

        # find experimental closest point on triangle by calling closest_distance_to(p')
        closest_dist, closest_point = triangle.closest_distance_to(p_prime)
        closest_point = closest_point.reshape(3)

        # compare the points
        error_closest_point = np.abs(closest_point - p)
        print(error_closest_point)
        self.assertTrue(np.isclose(error_closest_point[0], EPS))
        self.assertTrue(np.isclose(error_closest_point[1], EPS))
        self.assertTrue(np.isclose(error_closest_point[2], EPS))

        # compare the distances
        error_closest_dist = np.abs(closest_dist[0] - distance)
        self.assertTrue(np.isclose(error_closest_dist, EPS))

        pass

    def test_closest_point_outside_plane(self):
        # generate 3 random points that make up a triangle
        # generate random point p that lies outside the plane of the triangle -- ground truth point
        # project a vector p' from p of random distance perpendicular to the plane of the triangle
        # find experimental closest point on triangle by calling closest_distance_to(p')
        # compare the two
        # assert that experimental closest point and p are the same

        v1 = np.random.rand(3)
        v2 = np.random.rand(3)
        v3 = np.random.rand(3)
        triangle = Triangle(v1, v2, v3)

        # generate random point that lies on edge of the triangle
        # Randomly choose an edge: 0 for AB, 1 for BC, 2 for CA
        edge_choice = np.random.choice([0, 1, 2])
        
        # Generate a random value for t between 0 and 1
        t = np.random.rand()
        
        # Calculate the point on the chosen edge
        if edge_choice == 0:
            # Point on edge between v1 and v2
            p = (1 - t) * v1 + t * v2
            # get the unit vector perpendicular to edge AB
            edge_orthogonal = np.cross(v2 - v1, v3 - v1)
        elif edge_choice == 1:
            # Point on edge between v2 and v3
            p = (1 - t) * v2 + t * v3
            # get the unit vector perpendicular to edge BC
            edge_orthogonal = np.cross(v1 - v2, v3 - v2)            
        else:
            # Point on edge between v3 and v1
            p = (1 - t) * v3 + t * v1
            # get the unit vector perpendicular to edge CA
            edge_orthogonal = np.cross(v1 - v3, v2 - v3)

        # get unit vector perpendicular to the plane
        plane_orthogonal = np.cross(v2 - v1, v3 - v1)
        plane_orthogonal = plane_orthogonal / np.linalg.norm(plane_orthogonal)

        # project a vector p' from p of random distance perpendicular to the plane of the triangle
        distance_plane_orthogonal = np.random.rand()
        p_prime = p + distance_plane_orthogonal * plane_orthogonal

        # project a vector p'' from p' of random distance perpendicular to the edge of the triangle
        distance_edge_orthogonal = np.random.rand()
        p_prime = p_prime.reshape(1, 3)
        p_prime_prime = p_prime + distance_edge_orthogonal * edge_orthogonal

        # get distance from p'' to p
        distance = np.linalg.norm(p_prime_prime - p)        

        # find experimental closest point on triangle by calling closest_distance_to(p'')
        closest_dist, closest_point = triangle.closest_distance_to(p_prime_prime)
        closest_point = closest_point.reshape(3)

        # compare the points
        error_closest_point = np.abs(closest_point - p)
        print(error_closest_point)
        self.assertTrue(np.isclose(error_closest_point[0], EPS))
        self.assertTrue(np.isclose(error_closest_point[1], EPS))
        self.assertTrue(np.isclose(error_closest_point[2], EPS))

        # compare the distances
        error_closest_dist = np.abs(closest_dist[0] - distance)
        self.assertTrue(np.isclose(error_closest_dist, EPS))

        pass
        
class TestBoundingBoxContains(unittest.TestCase):
    """
    tests that the contains method of the BoundingBox class returns true for points inside 
    the bounding box and false for points outside the bounding box
    """
    def test_contains(self):
        for i in range(100):
            # define maximum and minimum values of a bounding box
            top_left = np.random.rand(3)
            bottom_right = np.random.rand(3)
            max_xyz = np.maximum(top_left, bottom_right)
            min_xyz = np.minimum(top_left, bottom_right)

            # create a bounding box
            box = BoundingBox(min_xyz, max_xyz)


            # create a random point t within the bounding box
            t_x = np.random.uniform(min_xyz[0], max_xyz[0])
            t_y = np.random.uniform(min_xyz[1], max_xyz[1])
            t_z = np.random.uniform(min_xyz[2], max_xyz[2])
            t = np.array([t_x, t_y, t_z])
            t = t.reshape(1, 3)

            # create a random point f outside the bounding box
            f = np.random.rand(3) + 2 * (top_left - bottom_right)
            f = f.reshape(1, 3)

            # assert that the bounding box contains t
            self.assertTrue(box.contains(t)[0])

            # assert that the bounding box does not contain f
            self.assertFalse(box.contains(f)[0])
        pass

    
class TestOutputAccuracy(unittest.TestCase):
    def test_output_accuracy(self):
        """
        tests that our own output values are within a certain threshold of the output values in the debug files
        """
        for SAMPLE_ID in ['A', 'B', 'C', 'D', 'E', 'F']:
            print(f'Testing sample {SAMPLE_ID}')
            matcher = FileOutputMatcher()
            error_s_k, error_c_k, error_norm = matcher(f'{OUTPUT_DIR}/pa4-{SAMPLE_ID}-Output.txt', f'{DATA_DIR}/PA4-{SAMPLE_ID}-Debug-Output.txt')

            # assert that the mean absolute error of d_k is within a certain threshold
            self.assertTrue(np.all(np.less_equal(error_s_k, [0.1, 0.1, 0.1])))

            # assert that the mean absolute error of c_k is within a certain threshold
            self.assertTrue(np.all(np.less_equal(error_c_k, [0.1, 0.1, 0.1])))

            # assert that the mean absolute error of ||d_k-c_k|| is within a certain threshold
            self.assertTrue(np.less_equal(error_norm, 0.1))

class TestModeWeights(unittest.TestCase):
    def test_mode_weights(self):
        """
        tests that _get_deformable_transf returns the correct mode weights
        """
        DATA_DIR = './pa345_data'
        SURFACE_DATA = f'{DATA_DIR}/Problem5MeshFile.sur'

        # initialize meshgrid, modes, and deformable icp
        surface_dl = Surfaceloader.read_file(SURFACE_DATA)
        meshgrid = Meshgrid(surface_dl.vertices, surface_dl.triangles)
        modes_dl = AtlasModesDataloader.read_file(f'{DATA_DIR}/Problem5Modes.txt')
        deform_icp = DeformableICP()


        # generate random values for the mode weights; ground truth
        mode_weights = np.random.rand(6)

        # use mode weights to deform the meshgrid
        deform_icp._apply_deformation(meshgrid, modes_dl.modes, mode_weights)

        # TODO sample points from the deformed meshgrid
        pt_cloud = None

        # get predicted mode weights
        closest_pt, dist, closest_tri = deform_icp.match(pt_cloud, meshgrid)
        pred_mode_weights = np.zeros(6)
        mode_coords = deform_icp._compute_mode_coordinates(closest_pt, closest_tri, modes_dl.modes, pred_mode_weights)
        pred_mode_weights = deform_icp._get_deformable_transf(pt_cloud, modes_dl.modes, mode_coords)

        # assert that the predicted mode weights are the same as the ground truth mode weights
        self.assertTrue(np.all(np.isclose(mode_weights, pred_mode_weights)))
        pass

class FileOutputMatcher():
    """
    Computes the mean absolute error between a predicted output file
    and a ground-truth output file.
    """
    def __call__(
        self,
        pred_file: str,
        gt_file: str
    ) -> Tuple[Tuple[float]]:
        dl1 = Dataloader.read_file(pred_file)
        dl2 = Dataloader.read_file(gt_file)

        raw_data1 = dl1.raw_data
        raw_data2 = dl2.raw_data

        # get the mean absolute error for d_x
        error_s_x = np.mean(np.abs(raw_data1[:, 0] - raw_data2[:, 0]))
        # get the mean absolute error for d_y
        error_s_y = np.mean(np.abs(raw_data1[:, 1] - raw_data2[:, 1]))
        # get the mean absolute error for d_z
        error_s_z = np.mean(np.abs(raw_data1[:, 2] - raw_data2[:, 2]))
        error_s_k = (error_s_x, error_s_y, error_s_z)

        # get the mean absolute error for c_x
        error_c_x = np.mean(np.abs(raw_data1[:, 3] - raw_data2[:, 3]))
        # get the mean absolute error for c_y
        error_c_y = np.mean(np.abs(raw_data1[:, 4] - raw_data2[:, 4]))
        # get the mean absolute error for c_z
        error_c_z = np.mean(np.abs(raw_data1[:, 5] - raw_data2[:, 5]))
        error_c_k = (error_c_x, error_c_y, error_c_z)

        # get the mean absolute error for ||d_k-c_k||
        error_norm = np.mean(np.abs(raw_data1[:, 6] - raw_data2[:, 6]))

        return error_s_k, error_c_k, error_norm

if __name__ == '__main__':
    unittest.main()