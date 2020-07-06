"""pyspectra API."""
import spectra_dense_interface

from .__version__ import __version__
from .pyspectra import eigensolver, eigensolverh

__author__ = "Netherlands eScience Center"
__email__ = 'f.zapata@esciencecenter.nl'


__all__ = ["__version__", "eigensolver",
           "eigensolverh", "spectra_dense_interface"]
