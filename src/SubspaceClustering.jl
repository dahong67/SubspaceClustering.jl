"""
Subspace clustering module. Provides algorithms for clustering data points by subspace.
"""
module SubspaceClustering

# Imports
using ArnoldiMethod: partialschur
using Compat
using LinearAlgebra: mul!, norm, svd!, svd
using Statistics: mean
using Logging: @info, @warn
using ProgressLogging: @withprogress, @logprogress
using Random: AbstractRNG, default_rng, randn!

# Exports
export KSSResult, kss, KASResult, kas
@compat public randsubspace, randaffinespace

# Algorithms
include("algorithms/kss.jl")
include("algorithms/kas.jl")

# Utility functions
"""
    randsubspace([rng=default_rng()], [T=Float64], D, d)

Generate a random `d`-dimensional subspace of `ℝᴰ`
and return a basis matrix with element type `T<:AbstractFloat`.

See also [`randsubspace!`](@ref)
"""
randsubspace(rng::AbstractRNG, ::Type{T}, D::Integer, d::Integer) where {T<:AbstractFloat} =
    randsubspace!(rng, Array{T}(undef, D, d))
randsubspace(::Type{T}, D::Integer, d::Integer) where {T<:AbstractFloat} =
    randsubspace(default_rng(), T, D, d)
randsubspace(rng::AbstractRNG, D::Integer, d::Integer) = randsubspace(rng, Float64, D, d)
randsubspace(D::Integer, d::Integer) = randsubspace(default_rng(), Float64, D, d)

"""
    randsubspace!([rng=default_rng()], U::AbstractMatrix)

Set the `D×d` matrix `U` to be the basis matrix of
a randomly generated `d`-dimensional subspace of `ℝᴰ`.

See also [`randsubspace`](@ref)
"""
function randsubspace!(rng::AbstractRNG, U::AbstractMatrix)
    # Check arguments
    eltype(U) <: AbstractFloat ||
        throw(ArgumentError("Basis matrix `U` must have real (floating point) elements."))
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

"""
    randaffinespace([rng=default_rng()], [T=Float64], D, d)

Generate a random `d`-dimensional subspace of `ℝᴰ` (if `T<:Real`) or of `ℂᴰ` (if `T<:Complex`)
and return a `D×d` orthonormal basis matrix and a bias vector of length `D` with elements of type `T`
(`T` must be a floating point type).

See also [`randaffinespace!`](@ref)
"""

randaffinespace(rng::AbstractRNG, ::Type{T}, D::Integer, d::Integer) where {T <: Union{AbstractFloat,Complex{<:AbstractFloat}}} =
    randaffinespace!(rng, Array{T}(undef, D, d), Array{T}(undef, D))
randaffinespace(::Type{T}, D::Integer, d::Integer) where {T <: Union{AbstractFloat,Complex{<:AbstractFloat}}} =
    randaffinespace(default_rng(), T, D, d)
randaffinespace(rng::AbstractRNG, D::Integer, d::Integer) = randaffinespace(rng, Float64, D, d)
randaffinespace(D::Integer, d::Integer) = randaffinespace(default_rng(), Float64, D, d)

"""
    randaffinespace!([rng=default_rng()], U::AbstractMatrix{<:Union{AbstractFloat,Complex{<:AbstractFloat}}})

Set the `D×d` matrix `U` to be the basis matrix and a vector of length `D` to be the bias vector `b` of
a randomly generated `d`-dimensional subspace of `ℝᴰ` or `ℂᴰ`.

See also [`randaffinespace`](@ref)
"""
function randaffinespace!(rng::AbstractRNG, U::AbstractMatrix{<:Union{AbstractFloat,Complex{<:AbstractFloat}}}, b::AbstractVector{<:Union{AbstractFloat,Complex{<:AbstractFloat}}})
    # Check arguments

    size(U, 2) <= size(U, 1) ||
        throw(ArgumentError("Subspace dimension `d` cannot be greater than the ambient dimension `D`."))
    
    length(b) == size(U, 1) ||
        throw(ArgumentError("Base vector `b` dimensions must match the dimensions of the basis matrix `U`."))

    # Generate random affine space
    randn!(rng, U)
    P, _, Q = svd!(U)
    mul!(U, P, Q')

    # Generate initial zero bias vector
    fill!(b, zero(eltype(b)))

    return U, b
end

randaffinespace!(U::AbstractMatrix, b::AbstractVector) = randaffinespace!(default_rng(), U, b)

end
