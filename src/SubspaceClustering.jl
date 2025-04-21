module SubspaceClustering

# Imports
using ArnoldiMethod: partialschur
using Compat
using LinearAlgebra: norm, svd, transpose
using Logging: @info, @warn
using ProgressLogging: @progress
using Random: default_rng, AbstractRNG

# Exports
export KSS, KSS!
@compat public randsubspace

# Algorithms
include("algorithms/kss.jl")

# Utility functions
"""
    randsubspace(D::Int, d::Vector{Int}; rng::AbstractRNG=default_rng())

Generate random d-dimensional subspaces. 

# Arguments
- `D::Int`: Dimension of the feature space.
- `d::Int`: Dimensions of the subspace.
- `rng::AbstractRNG=default_rng()`: Default global random number generator (RNG) with AbstractRNG type.

# Returns
A matrix, each of size `(D, d)`.
"""

function randsubspace(rng::AbstractRNG, D::Int, d::Int)
    A = randn(rng, D, d)
    # Perform polar decomposition
    U, _, V = svd(A)
    return U * V'
end

end
