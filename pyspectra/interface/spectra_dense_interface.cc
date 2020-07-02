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

// using Matrix =
//     Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using Eigen::Index;

Spectra::SortRule string_to_sortrule(const std::string& name)
{
    std::unordered_map<std::string, Spectra::SortRule> rules = {
        {"LargestMagn", Spectra::SortRule::LargestMagn},
        {"LargestReal", Spectra::SortRule::LargestReal},
        {"LargestImag", Spectra::SortRule::LargestImag},
        {"LargestAlge", Spectra::SortRule::LargestAlge},
        {"SmallestMagn", Spectra::SortRule::SmallestMagn},
        {"SmallestReal", Spectra::SortRule::SmallestReal},
        {"SmallestImag", Spectra::SortRule::SmallestImag},
        {"SmallestAlge", Spectra::SortRule::SmallestAlge},
        {"BothEnds", Spectra::SortRule::BothEnds}};
    return rules.at(name);
}

std::pair<Vector, Matrix> symeigssolver(const Matrix& mat, Index nvalues, Index nvectors, const std::string& selection)
{
    // Construct matrix operation object using the wrapper class DenseSymMatProd
    Spectra::DenseSymMatProd<double> op(mat);

    // Construct eigen solver object, requesting the largest three eigenvalues
    Spectra::SymEigsSolver<double, Spectra::DenseSymMatProd<double>> eigs(op, nvalues, nvectors);

    // Initialize and compute
    eigs.init();
    // Compute using the user provided selection rule
    eigs.compute(Spectra::SortRule::LargestAlge);
    // eigs.compute(string_to_sortrule(selection));

    // Retrieve results
    if (eigs.info() == Spectra::CompInfo::Successful)
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

    m.def("symmetric_eigensolver", &symeigssolver, py::return_value_policy::reference_internal);
}
