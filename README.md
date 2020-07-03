[![Build Status](https://github.com/NLESC-JCER/pyspectra/workflows/build/badge.svg)](https://github.com/NLESC-JCER/pyspectra/actions)

# PySpectra

Python interface to the [C++ Spectra library](https://github.com/yixuan/spectra)

## API

### Dense interface
To call the dense interface you need to import the following module:
```python
import numpy as np
from pyspectra import spectra_dense_interface
```
The following functions are available in the [spectra_dense_interface](https://github.com/NLESC-JCER/pyspectra/blob/master/pyspectra/interface/spectra_dense_interface.cc):
* ```py
  general_eigensolver(mat: np.ndarray, eigenpairs: int, basis_size: int, selection_rule: str) -> (np.ndarray, np.ndarray)
  ```
* ```py
  general_real_shift_eigensolver(mat: np.ndarray, eigenpairs: int, basis_size: int, shift: float, selection_rule: str) -> np.ndarray
  ```
* ```py
  symmetric_eigensolver(mat: np.ndarray, eigenpairs: int, basis_size: int, selection_rule: str) -> np.ndarray
  ```
* ```py
  symmetric_shift_eigensolver(mat: np.ndarray, eigenpairs: int, basis_size: int, shift: float, selection_rule: str) -> np.ndarray
  ```

**All functions return a tuple whith the resulting eigenvalues and eigenvectors.**

**Note**:
  Check the [available selection rules](https://github.com/NLESC-JCER/pyspectra/blob/master/include/Spectra/Util/SelectionRule.h)


## Example
Eigenpairs of a symmetric dense matrix
```py
import numpy as np
from pyspectra import spectra_dense_interface 

# eigenpairs to compute
nvalues = 2

# size of the search space
search_space = nvalues * 3

# Compute the highest eigenvalues
selection_rule = "LargestAlge"

# Create symmetric matrix
xs = np.random.normal(size=100).reshape(10,10)
mat = xs + xs.T

# Compute the eigenpairs
eigenvalues, eigenvectors = spectra_dense_interface.symetric_eigensolver(mat, nvalues, search_space, selection_rule)
```

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
