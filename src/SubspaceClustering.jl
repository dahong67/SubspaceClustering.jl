"""
Subspace clustering module. Provides algorithms for clustering data points by subspace.
"""
module SubspaceClustering

# Imports
using ArnoldiMethod: partialschur
using Compat
using LinearAlgebra: mul!, norm, svd!
using Logging: @info, @warn
using ProgressLogging: @withprogress, @logprogress
using Random: AbstractRNG, default_rng, randn!

# Exports
export KSSResult, kss
@compat public randsubspace

# Algorithms
include("algorithms/kss.jl")

# Utility functions
"""
    randsubspace([rng=default_rng()], [T=Float64], D, d) where T<:Number

Generate a random `d`-dimensional subspace of `ℝᴰ` (if `T<:Real`) or of `ℂᴰ` (if `T<:Complex`)
and return a `D×d` orthonormal basis matrix with elements of type `T`
(`T` must be a floating point type).

See also [`randsubspace!`](@ref)
"""
randsubspace(rng::AbstractRNG, ::Type{T}, D::Integer, d::Integer) where {T<:AbstractFloat} =
    randsubspace!(rng, Array{T}(undef, D, d))
randsubspace(rng::AbstractRNG, ::Type{T}, D::Integer, d::Integer) where {T<:Complex{<:AbstractFloat}} =
    randsubspace!(rng, Array{T}(undef, D, d))
randsubspace(::Type{T}, D::Integer, d::Integer) where {T<:Number} =
    randsubspace(default_rng(), T, D, d)

"""
    randsubspace!([rng=default_rng()], U::AbstractMatrix{T}) where T<:Number

Set the `D×d` matrix `U` to be the basis matrix of
a randomly generated `d`-dimensional subspace of `ℝᴰ` or `ℂᴰ`.

See also [`randsubspace`](@ref)
"""
function randsubspace!(rng::AbstractRNG, U::AbstractMatrix{T}) where T<:Number
    # Check arguments
    size(U, 2) <= size(U, 1) || throw(
        ArgumentError(
            "Subspace dimension `d` cannot be greater than the ambient dimension `D`.",
        ),
    )

    # Generate random subspace
    randn!(rng, U)
    P, _, Q = svd!(U)
    return mul!(U, P, Q')
end
randsubspace!(U::AbstractMatrix) = randsubspace!(default_rng(), U)

end
