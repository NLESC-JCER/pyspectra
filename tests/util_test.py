"""Helper functions to tests."""

import numpy as np


def norm(vs: np.array) -> float:
    """Compute the norm of a vector."""
    return np.sqrt(np.dot(vs, vs))


def create_random_matrix(size: int) -> np.array:
    """Create a numpy random matrix."""
    return np.random.normal(size=size ** 2).reshape(size, size)


def create_symmetic_matrix(size: int) -> np.array:
    """Create a numpy symmetric matrix."""
    xs = create_random_matrix(size)
    return xs + xs.T


def check_eigenpairs(
        matrix: np.ndarray, eigenvalues: np.ndarray, eigenvectors: np.ndarray) -> bool:
    """Check that the eigenvalue equation holds."""
    for i, value in enumerate(eigenvalues):
        residue = np.dot(
            matrix, eigenvectors[:, i]) - value * eigenvectors[:, i]
        assert norm(residue) < 1e-8
