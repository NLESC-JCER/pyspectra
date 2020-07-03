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

#include <Spectra/GenEigsComplexShiftSolver.h>
#include <Spectra/GenEigsSolver.h>
#include <Spectra/GenEigsRealShiftSolver.h>
#include <Spectra/SymEigsSolver.h>
#include <Spectra/SymEigsShiftSolver.h>
#include <Spectra/MatOp/DenseGenComplexShiftSolve.h>
#include <Spectra/MatOp/DenseGenMatProd.h>
#include <Spectra/MatOp/DenseGenRealShiftSolve.h>
#include <Spectra/MatOp/DenseSymMatProd.h>
#include <Spectra/MatOp/DenseSymShiftSolve.h>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

using ComplexMatrix = Eigen::MatrixXcd;
using ComplexVector = Eigen::VectorXcd;
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
template <typename ResultVector, typename ResultMatrix, typename Solver>
std::pair<ResultVector, ResultMatrix> compute_and_check(Solver& eigs, const std::string& selection)
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

/// \brief Call the Spectra::GenEigsSolver eigensolver
std::pair<ComplexVector, ComplexMatrix> geneigssolver(
    const Matrix& mat, Index nvalues, Index nvectors, const std::string& selection)
{
    using DenseOp = Spectra::DenseGenMatProd<double>;

    // Construct matrix operation object using the wrapper class DenseSymMatProd
    Spectra::DenseGenMatProd<double> op(mat);
    Spectra::GenEigsSolver<double, DenseOp> eigs(op, nvalues, nvectors);
    return compute_and_check<ComplexVector, ComplexMatrix>(eigs, selection);
}

/// \brief Call the Spectra::GenEigsRealShiftSolver eigensolver
std::pair<ComplexVector, ComplexMatrix> geneigsrealshiftsolver(
    const Matrix& mat, Index nvalues, Index nvectors, double sigma, const std::string& selection)
{
    using DenseOp = Spectra::DenseGenRealShiftSolve<double>;
    DenseOp op(mat);
    Spectra::GenEigsRealShiftSolver<double, DenseOp> eigs(op, nvalues, nvectors, sigma);
    return compute_and_check<ComplexVector, ComplexMatrix>(eigs, selection);
}

/// \brief Call the Spectra::GenEigsComplexShiftSolver eigensolver
std::pair<ComplexVector, ComplexMatrix> geneigscomplexshiftsolver(
    const Matrix& mat, Index nvalues, Index nvectors, double sigmar, double sigmai, const std::string& selection)
{
    using DenseOp = Spectra::DenseGenComplexShiftSolve<double>;
    DenseOp op(mat);
    Spectra::GenEigsComplexShiftSolver<double, DenseOp> eigs(op, nvalues, nvectors, sigmar, sigmai);
    return compute_and_check<ComplexVector, ComplexMatrix>(eigs, selection);
}

/// \brief Call the Spectra::DenseSymMatProd eigensolver
std::pair<Vector, Matrix> symeigssolver(
    const Matrix& mat, Index nvalues, Index nvectors, const std::string& selection)
{
    using DenseSym = Spectra::DenseSymMatProd<double>;
    // Construct matrix operation object using the wrapper class DenseSymMatProd
    DenseSym op(mat);
    Spectra::SymEigsSolver<double, DenseSym> eigs(op, nvalues, nvectors);

    return compute_and_check<Vector, Matrix>(eigs, selection);
}

/// \brief Call the Spectra::SymEigsShiftSolver eigensolver
std::pair<Vector, Matrix> symeigsshiftsolver(
    const Matrix& mat, Index nvalues, Index nvectors, double sigma, const std::string& selection)
{
    using DenseSymShift = Spectra::DenseSymShiftSolve<double>;
    // Construct matrix operation object using the wrapper class DenseSymMatProd
    DenseSymShift op(mat);
    Spectra::SymEigsShiftSolver<double, DenseSymShift> eigs(op, nvalues, nvectors, sigma);

    return compute_and_check<Vector, Matrix>(eigs, selection);
}

PYBIND11_MODULE(spectra_dense_interface, m)
{
    m.doc() = "Interface to the C++ spectra library, see: "
              "https://github.com/yixuan/spectra";

    m.def("general_eigensolver", &geneigssolver, py::return_value_policy::reference_internal);

    m.def("general_real_shift_eigensolver", &geneigsrealshiftsolver, py::return_value_policy::reference_internal);

    m.def("general_complex_shift_eigensolver", &geneigscomplexshiftsolver, py::return_value_policy::reference_internal);

    m.def("symmetric_eigensolver", &symeigssolver, py::return_value_policy::reference_internal);

    m.def("symmetric_shift_eigensolver", &symeigsshiftsolver, py::return_value_policy::reference_internal);
}
