"""pyspectra API"""
import logging

import spectra_dense_interface
from spectra_dense_interface import (general_eigensolver,
                                     general_real_shift_eigensolver,
                                     symmetric_eigensolver,
                                     symmetric_shift_eigensolver)

from .__version__ import __version__

logging.getLogger(__name__).addHandler(logging.NullHandler())

__author__ = "Netherlands eScience Center"
__email__ = 'f.zapata@esciencecenter.nl'


__all__ = ["general_eigensolver", "general_real_shift_eigensolver",
           "spectra_dense_interface",
           "symmetric_eigensolver", "symmetric_shift_eigensolver"]
