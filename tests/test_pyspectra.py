#!/usr/bin/env python
"""Tests for the pyspectra module."""
import numpy as np

from pyspectra import spectra_dense_interface


def norm(vs: np.array) -> float:
    """Compute the norm of a vector."""
    return np.sqrt(np.dot(vs, vs))


def test_dense_symmetric():
    """Test the interface to Spectra::SymEigsSolver"""
    # Number of rows/columns
    size = 100

    # Eigenpairs to compute
    pairs = 2
    search_space = pairs * 5

    # create symmetric matrix
    xs = np.random.normal(size=size ** 2).reshape(size, size)
    mat = xs + xs.T

    # These are the only supported rules
    selection_rules = ("LargestMagn",
                       "LargestAlge",
                       "SmallestAlge",
                       "BothEnds"
                       )

    for selection in selection_rules:
        print(f"testing selection rule:{selection}")
        es, cs = spectra_dense_interface.symmetric_eigensolver(
            mat, pairs, search_space, selection)
        for i, value in enumerate(es):
            residue = np.dot(mat, cs[:, i]) - value * cs[:, i]
            assert norm(residue) < 1e-8