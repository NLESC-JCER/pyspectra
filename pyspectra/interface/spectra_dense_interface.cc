/*
 * Copyright 2020 Netherlands eScience Center
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <utility>
#include <Eigen/Core>
#include <Eigen/Dense>

#include <Spectra/SymEigsSolver.h>
#include <Spectra/MatOp/DenseSymMatProd.h>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

using Matrix =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Vector = Eigen::VectorXd;
using Eigen::Index;

std::pair<Vector, Matrix> symeigssolver(const Matrix &mat, Index nvalues, std::string selection)
{
    // Construct matrix operation object using the wrapper class DenseSymMatProd
    Spectra::DenseSymMatProd<double> op(mat);

    //TODO: choose selection rule

    // Construct eigen solver object, requesting the largest three eigenvalues
    Spectra::SymEigsSolver<double, Spectra::LARGEST_ALGE, Spectra::DenseSymMatProd<double>> eigs(&op, nvalues, nvalues * 2);

    // Initialize and compute
    eigs.init();
    eigs.compute();

    // Retrieve results
    if (eigs.info() == Spectra::SUCCESSFUL)
    {
        return std::make_pair(eigs.eigenvalues(), eigs.eigenvectors());
    }
    else
    {
        throw std::runtime_error("The Spectra SymEigsSolver calculation has failed!");
    }
}

PYBIND11_MODULE(spectra_dense_interface, m)
{
    m.doc() = "Interface to the C++ spectra library, see: "
              "https://github.com/yixuan/spectra";

    m.def("symetric_eigensolver", &symeigssolver, py::return_value_policy::reference_internal);
}
