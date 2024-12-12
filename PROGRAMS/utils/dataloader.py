import numpy as np
from numpy.typing import NDArray
import pandas as pd
from typing import List, Any

"""
Code created by Edmund Sumpena and Jayden Ma
"""

class AtlasModesDataloader():
    """
    Atlas modes dataloader class that, in addition to the functionality of
    the base Dataloader, does the following:
        - extracts the number of modes in the atlas
        - extracts the number of vertices in the atlas
        - extracts the coordinates of vertices for each mode in the atlas
    """
    def __init__(
        self,
        N_modes: int,
        N_vertices: int,
        modes: NDArray[np.float32]
    ) -> None:
           
        # Store the number of modes and vertices
        self.N_modes: int = N_modes
        self.N_vertices: int = N_vertices

        # Store the coordinates of vertices for each mode
        self.modes: NDArray[np.float32] = modes

    # Construct an AtlasModesDataloader class given a data file
    @staticmethod
    def read_file(filename: str) -> 'AtlasModesDataloader':
        # Read the file line by line
        with open(filename, 'r') as file:
            lines = file.readlines()

        # Extract the number of vertices and modes from the first line
        first_line = lines[0].strip().split()
        N_vertices = int(first_line[1].split('=')[1])
        N_modes = int(first_line[2].split('=')[1]) + 1

        # Extract the coordinates of vertices for each mode
        modes = []
        for i in range(N_modes):
            mode_data = []
            for j in range(N_vertices):
            # Skip the first line of each set of coordinates
                line_index = i * (N_vertices + 1) + j + 2
                mode_data.append([float(coord) for coord in lines[line_index].strip().replace(',', '').split()])
            modes.append(np.array(mode_data, dtype=np.float32))
        modes = np.array(modes, dtype=np.float32)

        # Print the first and last element of modes
        print("First element of modes:", modes[0])
        print("Last element of modes:", modes[-1])

        return AtlasModesDataloader(N_modes, N_vertices, modes)

    # Reads the data file's metadata from dataframe
    @staticmethod
    def _read_metadata(df: pd.DataFrame) -> List[Any]:
        return list(df.columns.values)

    # Reads the data file's raw data from dataframe
    @staticmethod
    def _read_raw_data(df: pd.DataFrame) -> NDArray[np.float32]:
        raw_data = []

        # Extract all the coordinates from dataframe
        for _, row in df.iterrows():
            raw_data.append(row.values.flatten().tolist())

        return np.array(raw_data)

class Surfaceloader(): 
    """
    Surfaceloader class that performs the following functionalities:
        -  extracts vertices of the mesh in CT coordinates
        -  extracts the vertex and neighbor indices of each triangle in the mesh
        -  extracts the number of vertices in the mesh
        -  extracts the number of triangles in the mesh
    """
    def __init__(
        self,
        N_vertices: List[Any],
        N_triangles: List[Any],
        vertices: NDArray[np.float32],
        triangles: NDArray[np.int32]
    ) -> None:
        
        # Remove any leading or trailing spaces
        # for i in range(len(metadata)):
        #     metadata[i] = metadata[i].strip()

        # Store the number of vertices and triangles
        self.N_vertices: List[Any] = N_vertices
        self.N_triangles: List[Any] = N_triangles

        # Store tuples of (x, y, z) coordinates read from the file
        self.vertices: NDArray[np.float32] = vertices

        # Store the indices of the vertices that make up each triangle and its 3 neighbors
        self.triangles: NDArray[np.int32] = triangles

    # Construct a Surfaceloader class given a data file
    @staticmethod
    def read_file(filename: str) -> 'Surfaceloader':
        # Read the file line by line
        with open(filename, 'r') as file:
            lines = file.readlines()

        # get the number of vertices and triangles
        N_vertices = int(lines[0].strip())
        N_triangles = int(lines[N_vertices+1].strip())

        # Extract vertices from the file
        vertices = np.zeros((N_vertices, 3))
        for i in range(1, N_vertices + 1):
            vertices[i - 1] = list(map(float, lines[i].strip().split()))

        # Extract triangles from the file
        triangles = np.zeros((N_triangles, 6), dtype=int)
        for i in range(N_vertices + 1, N_vertices + N_triangles + 1):
            triangles[i - N_vertices - 1] = list(map(int, lines[i+1].strip().split()))
        
        # Remove the extra elements present in the raw data
        triangles = triangles[:,:3]

        return Surfaceloader(N_vertices, N_triangles, vertices, triangles)

    # Reads the data file's metadata from dataframe
    @staticmethod
    def _read_metadata(df: pd.DataFrame) -> List[Any]:
        return list(df.columns.values)

    # Reads (x, y, z) coordinates from each row
    @staticmethod
    def _read_raw_data(df: pd.DataFrame) -> NDArray[np.float32]:
        raw_data = []

        # Extract all the coordinates from dataframe
        for _, row in df.iterrows():
            raw_data.append(row.values.flatten().tolist()[0:3])

        return np.array(raw_data)

class Dataloader():
    """
    Base Dataloader class that performs some basic functionalities:
        -  Read the data files as a .csv
        -  Store basic file properties given on the first line of the data files
        -  Retrieve the individual coordinate points as an array of tuples
    """
    def __init__(
        self,
        metadata: List[Any],
        raw_data: NDArray[np.float32],
        N_A: int = 0,
        N_B: int = 0
    ) -> None:
        
        # Remove any leading or trailing spaces
        # for i in range(len(metadata)):
        #     metadata[i] = metadata[i].strip()

        # Stores metadata of the data file
        self.metadata: List[Any] = metadata

        # Store tuples of (x, y, z) coordinates read from the file
        self.raw_data: NDArray[np.float32] = raw_data

        # Store N_A and N_B parameters
        self.N_A: int = N_A
        self.N_B: int = N_B

    # Construct a Dataloader class given a data file
    @staticmethod
    def read_file(filename: str, delimiter: str = '', N_A: int = 0, N_B: int = 0) -> 'Dataloader':
        # Read the file line by line
        with open(filename, 'r') as file:
            lines = file.readlines()

        # Extract metadata from the first line
        if delimiter:
            metadata = lines[0].strip().split(delimiter)
        else:
            metadata = lines[0].strip().split()


        # Extract raw data from the remaining lines
        raw_data = []
        for line in lines[1:]:
            # Split the line by spaces and convert to floats
            if delimiter:
                coordinates = list(map(float, line.strip().split(delimiter)))
            else:
                coordinates = list(map(float, line.strip().split()))
            raw_data.append(coordinates)

        # Convert raw_data to a numpy array
        raw_data = np.array(raw_data, dtype=np.float32)

        return Dataloader(metadata, raw_data, N_A, N_B)

    # Reads the data file's metadata from dataframe
    @staticmethod
    def _read_metadata(df: pd.DataFrame) -> List[Any]:
        return list(df.columns.values)

    # Reads (x, y, z) coordinates from each row
    @staticmethod
    def _read_raw_data(df: pd.DataFrame) -> NDArray[np.float32]:
        raw_data = []

        # Extract all the coordinates from dataframe
        for _, row in df.iterrows():
            raw_data.append(row.values.flatten().tolist()[0:3])

        return np.array(raw_data)
    
class RigidBodyDataloader(Dataloader):
    """
    Rigid body dataloader class that, in addition to the functionality of
    the base Dataloader, does the following:
        - extracts a set of coordinates corresponding to the LED markers in body coordinates
        - extracts a coordinate corresponding to the tip in body coordinates
        - extracts N_markers, the number of LED markers in body coordinates
    """    
    def __init__(
    self,
    metadata: List[Any],
    raw_data: NDArray[np.float32]
    ) -> None:
        super().__init__(metadata, raw_data)

        # number of LED markers in body coordinates
        self.N_markers: int = int(metadata[0])

        # get the LED markers in body coordinates
        self.markers: NDArray[np.float32] = raw_data[:self.N_markers]

        # get the tip in body coordinates
        self.tip: NDArray[np.float32] = raw_data[self.N_markers]

    @staticmethod
    def read_file(filename: str) -> 'RigidBodyDataloader':
        dl = Dataloader.read_file(filename)
        return RigidBodyDataloader(dl.metadata, dl.raw_data)

class SampleReadingsDataloader(Dataloader):
    """
    Sample readings dataloader class that, in addition to the functionality of
    the base Dataloader, does the following:
        - extracts a set of coordinates corresponding to the LED markers of body A in body coordinates
        - extracts a set of coordinates corresponding to the LED markers of body B in body coordinates
        - extracts a set of coordinates corresponding to the dummy LED markers in body coordinates
        - extracts N_S, the total number of LEDs read per sample
        - extracts N_samps, the total number of samples
    """    
    def __init__(
    self,
    metadata: List[Any],
    raw_data: NDArray[np.float32],
    N_A: int,
    N_B: int
    ) -> None:
        super().__init__(metadata, raw_data, N_A, N_B)

        # number of LEDs read per sample
        self.N_S: int = int(metadata[0])

        # number of samples
        self.N_samps: int = int(metadata[1])

        # self.raw_data = np.zeros((self.N_samps, self.N_S, 3))
        self.body_A = np.zeros((self.N_samps, N_A, 3))
        self.body_B = np.zeros((self.N_samps, N_B, 3))
        self.dummy = np.zeros((self.N_samps, self.N_S - N_A - N_B, 3))

        # get the LED marker coordinates
        for sample in range(self.N_samps):
            start = sample * self.N_S
            end = (sample + 1) * self.N_S

            N_A_ind = start
            N_B_ind = start + N_A
            dummy_ind = start + N_A + N_B

            self.body_A[sample] = raw_data[N_A_ind:N_B_ind]
            self.body_B[sample] = raw_data[N_B_ind:dummy_ind]
            self.dummy[sample] = raw_data[dummy_ind:end]
            # self.raw_data[sample] = raw_data[start:end]

    @staticmethod
    def read_file(filename: str, delimiter: str = ',', N_A: int = 0, N_B: int = 0) -> 'RigidBodyDataloader':
        dl = Dataloader.read_file(filename, delimiter, N_A, N_B)
        return SampleReadingsDataloader(dl.metadata, dl.raw_data, N_A, N_B)