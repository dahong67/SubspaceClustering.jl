# Algorithm Types

"""
    Algorithm Types for Subspace Estimation
"""
module SubspaceEstimation

using ..SubspaceClustering
using TSVD: tsvd
using IncrementalSVD: isvd
using LinearAlgebra:svd

"""
    AbstractAlgorithm

Abstract type for Subspace basis estimation algorithms.
Concrete types `ConcreteAlgorithm <: AbstractAlgorithm` should implement
`estimate_subspace(Xk::AbstractMatrix, dk::Integer, algorithm::ConcreteAlgorithm)`
and return a basis matrix `U` of size `D×d` where `D` is the ambient dimension
and `d` is the subspace dimension.
"""
abstract type AbstractAlgorithm end

"""
    estimate_subspace(Xk, dk, algorithm)

Internal function to estimate the subspace basis matrix
using the algorithm `algorithm`.
"""
function estimate_subspace end

include("methods/tsvd.jl")
include("methods/isvd.jl")
include("methods/svd.jl")

end