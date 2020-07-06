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

rules = {"LargestMagn",
         "LargestReal",
         "LargestImag",
         "LargestAlge",
         "SmallestReal",
         "SmallestMagn",
         "SmallestImag",
         "SmallestAlge",
         "BothEnds"}


def check_and_sanitize(
        mat: np.ndarray, nvalues: int, selection_rule: Optional[str],
        search_space: Optional[int],
        shift: Optional[Union[np.float, np.complex]]) -> (str, str):
    """Check that the values are correct and initialize missing values."""
    if nvalues > mat.shape[0]:
        raise RuntimeError(
            "The requested number of eigenpairs is larger than the matrix size")
    if selection_rule is not None and selection_rule not in rules:
        raise RuntimeError(f"unknown selection_rule:{selection_rule}")
    if selection_rule is None:
        selection_rule = "LargestMagn"
    if search_space is None:
        search_space = nvalues * 5

    if shift is not None:
        if not any(isinstance(shift, x) for x in (np.float, np.complex)):
            raise RuntimeError("Shift must be None, float or complex")

    return search_space, selection_rule


def eigensolver(
        mat: np.ndarray, nvalues: int, selection_rule: Optional[str] = None,
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
    search_space, selection_rule = check_and_sanitize(
        mat, nvalues, selection_rule, search_space, shift)

    if shift is None:
        return spectra_dense_interface.general_eigensolver(
            mat, nvalues, search_space, selection_rule)
    if isinstance(shift, np.float):
        return spectra_dense_interface.general_real_shift_eigensolver(
            mat, nvalues, search_space, shift, selection_rule)
    else:
        return spectra_dense_interface.general_complex_shift_eigensolver(
            mat, nvalues, search_space, shift.real, shift.imag, selection_rule)


def eigensolverh(
        mat: np.ndarray, nvalues: int, selection_rule: str,
        search_space: Optional[int] = None, mat_B: np.ndarray = None,
        shift: Optional[Union[np.float, np.complex]] = None) -> (np.ndarray, np.ndarray):
    """Compute ``nvalues`` eigenvalues for the symmetric matrix ``mat``.

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
    search_space, selection_rule = check_and_sanitize(
        mat, nvalues, selection_rule, search_space, shift)

    if shift is None:
        return spectra_dense_interface.symmetric_eigensolver(
            mat, nvalues, search_space, selection_rule)
    elif mat_B is None:
        return spectra_dense_interface.symmetric_shift_eigensolver(
            mat, nvalues, search_space, shift, selection_rule)
    else:
        return spectra_dense_interface.symmetric_generalized_shift_eigensolver(
            mat, mat_B, nvalues, search_space, shift, selection_rule)
