"""
Subspace clustering module. Provides algorithms for clustering data points by subspace.
"""
module SubspaceClustering

# Imports
using ArnoldiMethod: partialschur
using Base.Threads
using Clustering: KmeansResult, kmeans
using Compat: Compat, @compat
using LinearAlgebra: Diagonal, I, Symmetric, mul!, normalize, svd!
using Logging: @info, @warn
using ProgressLogging: @logprogress, @withprogress
using Random: AbstractRNG, default_rng, randn!, MersenneTwister
using SparseArrays: sparse
using Statistics: mean

# Exports
export KASResult, kas, KSSResult, kss, TSCResult, tsc, EKSSResult, ekss
@compat public randsubspace

# Algorithms
include("algorithms/kss.jl")
include("algorithms/kas.jl")
include("algorithms/tsc.jl")
include("algorithms/ekss.jl")

# Utility functions
include("utils/randsubspace.jl")

end
