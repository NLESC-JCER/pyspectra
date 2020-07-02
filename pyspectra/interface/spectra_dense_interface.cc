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
#include <Spectra/SymEigsShiftSolver.h>
#include <Spectra/MatOp/DenseSymMatProd.h>
#include <Spectra/MatOp/DenseSymShiftSolve.h>

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

/// \brief Run the computation and throw and error if it fails
template <typename Solver>
std::pair<Vector, Matrix> compute_and_check(Solver& eigs, const std::string& selection)
{
    // Initialize and compute
    eigs.init();
    // Compute using the user provided selection rule
    eigs.compute(string_to_sortrule(selection));

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

/// \brief Call the Spectra::DenseSymMatProd eigensolver
std::pair<Vector, Matrix> symeigssolver(const Matrix& mat, Index nvalues, Index nvectors, const std::string& selection)
{
    // Construct matrix operation object using the wrapper class DenseSymMatProd
    Spectra::DenseSymMatProd<double> op(mat);
    Spectra::SymEigsSolver<double, Spectra::DenseSymMatProd<double>> eigs(op, nvalues, nvectors);

    return compute_and_check(eigs, selection);
}

/// \brief Call the Spectra::SymEigsShiftSolver eigensolver
std::pair<Vector, Matrix> symeigsshiftsolver(const Matrix& mat, Index nvalues, Index nvectors, double sigma, const std::string& selection)
{
    // Construct matrix operation object using the wrapper class DenseSymMatProd
    Spectra::DenseSymShiftSolve<double> op(mat);
    Spectra::SymEigsShiftSolver<double, Spectra::DenseSymShiftSolve<double>> eigs(op, nvalues, nvectors, sigma);

    return compute_and_check(eigs, selection);
}

PYBIND11_MODULE(spectra_dense_interface, m)
{
    m.doc() = "Interface to the C++ spectra library, see: "
              "https://github.com/yixuan/spectra";

    m.def("symmetric_eigensolver", &symeigssolver, py::return_value_policy::reference_internal);

    m.def("symmetric_shift_eigensolver", &symeigsshiftsolver, py::return_value_policy::reference_internal);
}
