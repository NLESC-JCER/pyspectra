"""PySpectra API.

API
---
.. autofunction:: eigensolver
.. autofunction:: eigensolverh
"""
from typing import Optional, Union

import numpy as np

import spectra_dense_interface

__all__ = ["eigensolver", "eigensolverh"]


def eigensolver(
        mat: np.ndarray, nvalues: int, selection_rule: str,
        search_space: Optional[int] = None,
        shift: Optional[Union[np.float, np.complex]] = None) -> (np.ndarray, np.ndarray):
    """
    Compute ``nvalues`` for matrix ``mat``.

    Parameters
    ----------
    mat
        Matrix to compute the eigenpairs
    nvalues
        Number of eigenpairs to compute
    search_space
        Size of the search space
    selection_rule
        Target of the spectrum to compute. Available values:
        LargestMagn, LargestReal, LargestImag, LargestAlge,
        SmallestMagn, SmallestReal, SmallestImag, SmallestAlge
        BothEnds
    shift
        scalar value of the shift

    Raises
    ------
    RunTimeError
        if the algorithm does not converge

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Eigenvalues and eigenvectors
    """
    pass


def eigensolverh(
        mat: np.ndarray, nvalues: int, selection_rule: str,
        search_space: Optional[int] = None, mat_B: np.ndarray = None,
        shift: Optional[Union[np.float, np.complex]] = None) -> (np.ndarray, np.ndarray):
    """
    Compute ``nvalues`` eigenvalues for the symmetric matrix ``mat``

    Parameters
    ----------
    mat
        Matrix to compute the eigenpairs
    nvalues
        Number of eigenpairs to compute
    search_space
        Size of the search space
    mat_B
        Solve the generalized eigenvalue problem
    selection_rule
        Target of the spectrum to compute. Available values:
        LargestMagn, LargestReal, LargestImag, LargestAlge,
        SmallestMagn, SmallestReal, SmallestImag, SmallestAlge
        BothEnds
    shift
        scalar value of the shift

    Raises
    ------
    RunTimeError
        if the algorithm does not converge

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Eigenvalues and eigenvectors
    """
    pass
