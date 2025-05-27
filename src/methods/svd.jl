# Subspace Basis Estimation using SVD

"""
    Singular Value Decomposition (SVD)

The default subspace basis estimation algorithm used by `kss` is
set to SVD, which uses standard function from the Julia standard library.
"""
Base.@kwdef struct SVD <:AbstractAlgorithm

end

const DEFAULT_ALGORITHM = SVD()

function estimate_subspace(Xk::AbstractMatrix, dk::Integer, algorithm::SubspaceEstimation.SVD)
    U, _, _ = svd(Xk; full=true)
    return U[:, 1:dk] 
end
