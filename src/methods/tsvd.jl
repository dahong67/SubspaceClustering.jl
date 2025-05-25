# Subspace Basis Estimation using Truncated SVD

"""
    Truncated Singular Value Decomposition (TSVD)

Algorithm parameters:
- `maxiters::Int`: Maximum number of iterations for the TSVD algorithm. Default is 200.
"""
Base.@kwdef struct TrunSVD <:AbstractAlgorithm
    maxiters::Int = 200
end

function estimate_subspace(Xk::AbstractMatrix, dk::Integer, algorithm::SubspaceEstimation.TrunSVD)
    U, _, _ = tsvd(Xk, dk; maxiter = algorithm.maxiters)
    return U
end