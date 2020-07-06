[![Build Status](https://github.com/NLESC-JCER/pyspectra/workflows/build/badge.svg)](https://github.com/NLESC-JCER/pyspectra/actions)

# PySpectra

Python interface to the [C++ Spectra library](https://github.com/yixuan/spectra)

## General Eigensolvers
**PySpecta** offers two general interfaces to [Spectra](https://github.com/yixuan/spectra): **eigensolver** and **eigensolverh**. For general
and symmetric matrices respectively.

These two functions would invoke the most suitable method based on the information provided by the user.

### Example
Eigenpairs of a symmetric dense matrix
```py
import numpy as np
from pyspectra import eigensolver, eigensolverh

# matrix size
size = 100

# number of eigenpairs to compute
nvalues = 2

# Create random matrix
xs = np.random.normal(size=size ** 2).reshape(size, size)

# Create symmetric matrix
mat = xs + xs.T

# Compute two eigenpairs
eigenvalues, eigenvectors = eigensolverh(xs, nvalues)
symm_eigenvalues, symm_eigenvectors = eigensolverh(mat, nvalues)
```
**Note**:

  The [available selection_rules](https://github.com/NLESC-JCER/pyspectra/blob/master/include/Spectra/Util/SelectionRule.h) to compute a portion of the spectrum are:
  * LargestMagn
  * LargestReal
  * LargestImag
  * LargestAlge
  * SmallestMagn
  * SmallestReal
  * SmallestImag
  * SmallestAlge
  * BothEnds

## Eigensolvers Dense Interface
You can also call directly the dense interface. You would need
to import the following module:
```python
import numpy as np
from pyspectra import spectra_dense_interface
```
The following functions are available in the [spectra_dense_interface](https://github.com/NLESC-JCER/pyspectra/blob/master/pyspectra/interface/spectra_dense_interface.cc):
* ```py
  general_eigensolver(
    mat: np.ndarray, eigenpairs: int, basis_size: int, selection_rule: str)
    > (np.ndarray, np.ndarray)
  ```
* ```py
  general_real_shift_eigensolver(
    mat: np.ndarray, eigenpairs: int, basis_size: int, shift: float, selection_rule: str)
    -> (np.ndarray, np.ndarray)
  ```
* ```py
  general_complex_shift_eigensolver(
    mat: np.ndarray, eigenpairs: int, basis_size: int,
    shift_real: float, shift_imag: float, selection_rule: str)
    -> (np.ndarray, np.ndarray)
  ```
* ```py
  symmetric_eigensolver(
    mat: np.ndarray, eigenpairs: int, basis_size: int, selection_rule: str)
    -> (np.ndarray, np.ndarray)
  ```
* ```py
  symmetric_shift_eigensolver(
    mat: np.ndarray, eigenpairs: int, basis_size: int, shift: float, selection_rule: str)
    -> (np.ndarray, np.ndarray)
  ```
* ```py
  symmetric_generalized_shift_eigensolver(
    mat_A: np.ndarray, mat_B, eigenpairs: int, basis_size: int, shift: float,
    selection_rule: str)
    -> (np.ndarray, np.ndarray)
  ```

### Example
Eigenpairs of a symmetric dense matrix using shift
```py
import numpy as np
from pyspectra import spectra_dense_interface

size = 100
nvalues = 2 # eigenpairs to compute
search_space = nvalues * 2 # size of the search space
shift = 1.0

# Create random matrix
xs = np.random.normal(size=size ** 2).reshape(size, size)

# Create symmetric matrix
mat = xs + xs.T

# Compute the eigenpairs
symm_eigenvalues, symm_eigenvectors = \
  spectra_dense_interface.symmetric_eigensolver(
  mat, nvalues, search_space, shift)
```

**All functions return a tuple whith the resulting eigenvalues and eigenvectors.**


## Installation
To install pyspectra, do:
```bash
  git clone https://github.com//pyspectra.git
  cd pyspectra
  pip install .
```

Run tests (including coverage) with:

```bash
  pytest tests
```

## Contributing

If you want to contribute to the development of pyspectra,
have a look at the contribution guidelines **CONTRIBUTING.md**

## License

Copyright (c) 2020, Netherlands eScience Center

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.



## Credits
This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [NLeSC/python-template](https://github.com/NLeSC/python-template).
