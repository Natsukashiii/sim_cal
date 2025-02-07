import numpy as np


def build_distance_matrix(similarity_matrix):
        """
        Converts a similarity matrix into a distance.py matrix.

        Args:
            similarity_matrix: The similarity matrix to convert. It can be a NumPy array
                               or other iterable structures (like tuples or lists).

        Returns:
            np.ndarray: The resulting distance.py matrix.
        """

        if not isinstance(similarity_matrix, np.ndarray):
                try:
                        similarity_matrix = np.array(similarity_matrix)
                except Exception as e:
                        raise ValueError("Input must be convertible to a NumPy array.") from e

        # Ensure the matrix is square
        if similarity_matrix.shape[0] != similarity_matrix.shape[1]:
                raise ValueError("Matrix must be square (same number of rows and columns).")

        # Compute the distance.py matrix as (1 - similarity)
        distance_matrix = 1 - similarity_matrix


        # Ensure the diagonal is 0 (no distance.py from a point to itself)
        np.fill_diagonal(distance_matrix, 0)

        return distance_matrix
