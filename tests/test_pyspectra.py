"""Check the pytest wrappers."""

from typing import Callable, List, Optional, Tuple, TypeVar, Union

import numpy as np
import pytest

from pyspectra import eigensolver, eigensolverh

from .util_test import (check_eigenpairs, create_random_matrix,
                        create_symmetic_matrix)

T = TypeVar('T')

SIZE = 100  # Matrix size
SIGMA = 1.0 + 0.5j


def run_test(fun: Callable[[T], Tuple[np.ndarray, np.ndarray]],
             mat: np.ndarray, nvalues: int, rules: List[str],
             search_space: Optional[int],
             shift: Optional[Union[np.float, np.complex]],
             generalized: Optional[np.ndarray] = None) -> None:
    """Run a given configuration."""
    for r in rules:
        print(f"Testing selection_rule: {r}")
        if generalized is not None:
            es, cs = fun(mat, nvalues, selection_rule=r,
                         search_space=search_space, shift=shift,
                         mat_B=generalized)
        else:
            es, cs = fun(mat, nvalues, selection_rule=r,
                         search_space=search_space, shift=shift)
        check_eigenpairs(mat, es, cs)


def test_eigensolver():
    """Check the eigensolver interface."""
    mat = create_random_matrix(SIZE)
    rules = ("LargestMagn",
             "LargestReal",
             "LargestImag",
             "SmallestReal")

    nvalues = 2
    print("default search space and shift = None")
    run_test(eigensolver, mat, nvalues, rules,
             search_space=None, shift=None)

    search_space = nvalues * 5
    print("search_space = nvalues * 5 and shift = None")
    run_test(eigensolver, mat, nvalues, rules,
             search_space=search_space, shift=None)

    print("search_space = default and shift = 1.0")
    run_test(eigensolver, mat, nvalues, rules,
             search_space=None, shift=SIGMA.real)

    print("search_space = default and shift = 1.0 + 0.5j")
    run_test(eigensolver, mat, nvalues, rules,
             search_space=None, shift=SIGMA)

    print("search_space = nvalues *  5 and shift = 1.0 + 0.5j")
    run_test(eigensolver, mat, nvalues, rules,
             search_space=search_space, shift=SIGMA)


def test_eigensolverh():
    """Check the eigensolverh interface."""
    mat = create_symmetic_matrix(SIZE)
    rules = ("LargestMagn",
             "LargestAlge",
             "SmallestAlge",
             "BothEnds"
             )

    nvalues = 2
    print("default search space and shift = None")
    run_test(eigensolverh, mat, nvalues, rules,
             search_space=None, shift=None)

    print("search space = nvalues * 5 and shift = None")
    run_test(eigensolverh, mat, nvalues, rules,
             search_space=nvalues * 5, shift=None)

    print(f"default search space and shift = {SIGMA.real}")
    run_test(eigensolverh, mat, nvalues, rules,
             search_space=None, shift=SIGMA.real)


def test_invalid_argument():
    """Check that an error is raised if the arguments are invalid."""
    mat = create_symmetic_matrix(SIZE)
    nvalues = 2
    print("wrong name for selection rule")
    with pytest.raises(RuntimeError):
        eigensolverh(mat, nvalues, selection_rule="Boom")

    print("more eigenpairs requested than columns")
    with pytest.raises(RuntimeError):
        eigensolverh(mat, SIZE + 1)

    print("Shift is not an scalar")
    with pytest.raises(RuntimeError):
        eigensolverh(mat, nvalues, shift="1.0")
