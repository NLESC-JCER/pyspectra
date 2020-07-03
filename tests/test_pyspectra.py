#!/usr/bin/env python
"""Tests for the pyspectra module."""
import numpy as np

from pyspectra import spectra_dense_interface
from typing import Callable, List, Tuple, TypeVar

T = TypeVar('T')


def norm(vs: np.array) -> float:
    """Compute the norm of a vector."""
    return np.sqrt(np.dot(vs, vs))


def create_symmetic_matrix(size: int) -> np.array:
    """Create a numpy symmetric matrix."""
    xs = np.random.normal(size=size ** 2).reshape(size, size)
    return xs + xs.T


def run_test(function: Callable[[T], np.array], args: List[T], selection_rules: Tuple[str]) -> None:
    """Call ``function`` with ``args``."""
    for selection in selection_rules:
        print(f"testing selection rule:{selection}")
        es, cs = function(*args, selection)
        for i, value in enumerate(es):
            residue = np.dot(args[0], cs[:, i]) - value * cs[:, i]
            assert norm(residue) < 1e-8


def test_dense_symmetric():
    """Test the interface to Spectra::SymEigsSolver."""
    # Number of rows/columns
    size = 100

    # Eigenpairs to compute
    pairs = 2
    search_space = pairs * 5
    mat = create_symmetic_matrix(size)

    # These are the only supported rules
    selection_rules = ("LargestMagn",
                       "LargestAlge",
                       "SmallestAlge",
                       "BothEnds"
                       )

    args = (mat, pairs, search_space)
    run_test(spectra_dense_interface.symmetric_eigensolver,
             args, selection_rules)


def test_dense_symmetric_shift():
    """Test the interface to Spectra::SymEigsShiftSolver."""
    # Number of rows/columns
    size = 100

    # Eigenpairs to compute
    pairs = 2
    search_space = pairs * 5
    mat = create_symmetic_matrix(size)

    # shift to use
    sigma = 1.0

    # These are the only supported rules
    selection_rules = ("LargestMagn",
                       "LargestAlge",
                       "SmallestAlge",
                       "BothEnds"
                       )

    args = (mat, pairs, search_space, sigma)
    run_test(spectra_dense_interface.symmetric_shift_eigensolver,
             args, selection_rules)
