#!/usr/bin/env python
"""Tests for the pyspectra module."""
from typing import Callable, List, Tuple, TypeVar

import numpy as np
import pytest

from pyspectra import spectra_dense_interface

from .util_test import (check_eigenpairs, create_random_matrix,
                        create_symmetic_matrix)

T = TypeVar('T')

# Constant for all the tests
SIZE = 100  # Matrix size
PAIRS = 2  # number of eigenpairs
SEARCH_SPACE = PAIRS * 5
SIGMA = 1.0
SIGMAR = 2.0  # Real shift
SIGMAI = 1.0  # Imag shift


def run_test(
        function: Callable[[T], np.ndarray], args: List[T], selection_rules: Tuple[str],
        is_symmetric: bool = True, is_generalized: bool = False) -> None:
    """Call ``function`` with ``args``."""
    for selection in selection_rules:
        print(f"testing selection rule:{selection}")
        es, cs = function(*args, selection)
        fun_numpy = np.linalg.eigh if is_symmetric else np.linalg.eig
        if is_generalized:
            es_np, _cs_np = fun_numpy(args[0], args[1])
        else:
            es_np, _cs_np = fun_numpy(args[0])
        print(f"Expected eigenvalues:{es}")
        print(f"Computed eigenvalues:{es_np}")
        check_eigenpairs(args[0], es, cs)


def test_dense_general():
    """Test the interface to Spectra::GenEigsSolver."""
    mat = create_random_matrix(SIZE)

    # These are the only supported rules
    selection_rules = ("LargestMagn",
                       "LargestReal",
                       "LargestImag",
                       "SmallestReal",
                       "SmallestImag"
                       )

    args = (mat, PAIRS, SEARCH_SPACE)
    run_test(spectra_dense_interface.general_eigensolver,
             args, selection_rules, is_symmetric=False)


def test_dense_real_shift_general():
    """Test the interface to Spectra::GenEigsRealShiftSolver."""
    mat = create_random_matrix(SIZE)

    # These are the only supported rules
    selection_rules = ("LargestMagn",
                       "LargestReal",
                       "LargestImag",
                       "SmallestReal",
                       "SmallestImag"
                       )

    args = (mat, PAIRS, SEARCH_SPACE, SIGMA)
    run_test(spectra_dense_interface.general_real_shift_eigensolver,
             args, selection_rules)


def test_dense_complex_shift_general():
    """Test the interface to Spectra::GenEigsComplexShiftSolver."""
    mat = create_random_matrix(SIZE)

    # These are the only supported rules
    selection_rules = ("LargestMagn",
                       "LargestReal",
                       "LargestImag",
                       "SmallestReal",
                       "SmallestImag"
                       )

    args = (mat, PAIRS, SEARCH_SPACE, SIGMAR, SIGMAI)
    run_test(spectra_dense_interface.general_complex_shift_eigensolver,
             args, selection_rules)


def test_dense_symmetric():
    """Test the interface to Spectra::SymEigsSolver."""
    mat = create_symmetic_matrix(SIZE)

    # These are the only supported rules
    selection_rules = ("LargestMagn",
                       "LargestAlge",
                       "SmallestAlge",
                       "BothEnds"
                       )

    args = (mat, PAIRS, SEARCH_SPACE)
    run_test(spectra_dense_interface.symmetric_eigensolver,
             args, selection_rules)


def test_dense_symmetric_shift():
    """Test the interface to Spectra::SymEigsShiftSolver."""
    # Eigenpairs to compute
    mat = create_symmetic_matrix(SIZE)

    # These are the only supported rules
    selection_rules = ("LargestMagn",
                       "LargestAlge",
                       "SmallestAlge",
                       "BothEnds"
                       )

    args = (mat, PAIRS, SEARCH_SPACE, SIGMA)
    run_test(spectra_dense_interface.symmetric_shift_eigensolver,
             args, selection_rules)


# def test_dense_symmetric_generalized_shift():
#     """Test the interface to Spectra::SymEigsShiftSolver."""
#     # Eigenpairs to compute
#     mat_A = create_symmetic_matrix(SIZE)
#     mat_B = np.diag(np.abs(np.random.normal(size=SIZE)))

#     # These are the only supported rules
#     selection_rules = ("LargestMagn",
#                        "LargestAlge",
#                        "SmallestAlge",
#                        "BothEnds"
#                        )

#     args = (mat_A, mat_B, PAIRS, SEARCH_SPACE, SIGMA)
#     run_test(spectra_dense_interface.symmetric_generalized_shift_eigensolver,
#              args, selection_rules)


def test_unknown_selection_rule():
    """Check that an error is raise if a selection rule is unknown."""
    mat = create_symmetic_matrix(SIZE)
    selection_rules = ("something")

    args = (mat, PAIRS, SEARCH_SPACE)
    with pytest.raises(RuntimeError):
        run_test(spectra_dense_interface.symmetric_eigensolver,
                 args, selection_rules)
