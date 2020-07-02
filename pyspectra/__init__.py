"""pyspectra API"""
import logging

import spectra_dense_interface
from spectra_dense_interface import symmetric_eigensolver

from .__version__ import __version__

logging.getLogger(__name__).addHandler(logging.NullHandler())

__author__ = "Netherlands eScience Center"
__email__ = 'f.zapata@esciencecenter.nl'


__all__ = ["spectra_dense_interface", "symmetric_eigensolver"]
