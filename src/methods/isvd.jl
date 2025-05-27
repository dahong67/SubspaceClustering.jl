# Subspace Basis Estimation using Incremental SVD

"""
    Incremental SVD (TSVD)

"""
Base.@kwdef struct ISVD <:AbstractAlgorithm

end

function estimate_subspace(Xk::AbstractMatrix, dk::Integer, algorithm::SubspaceEstimation.ISVD)
    U, _ = isvd(Xk, dk)
    return U

end