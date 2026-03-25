"""
Subspace clustering module. Provides algorithms for clustering data points by subspace.
"""
module SubspaceClustering

# Imports
using ArnoldiMethod: partialschur, partialeigen
using Clustering: KmeansResult, kmeans
using Compat: Compat, @compat
using LinearAlgebra: Diagonal, I, Symmetric, mul!, normalize, normalize!, svd!
using Logging: @info, @warn
using ProgressLogging: @logprogress, @withprogress
using Random: AbstractRNG, default_rng, randn!
using SparseArrays: sparse
using Statistics: mean

# Exports
export KASResult, kas, KSSResult, kss, TSCResult, tsc
@compat public randsubspace

# Utility functions/macros
include("utils/randsubspace.jl")
include("utils/progresslogging.jl")

# Algorithms
include("algorithms/kss.jl")
include("algorithms/kas.jl")
include("algorithms/tsc.jl")

end
