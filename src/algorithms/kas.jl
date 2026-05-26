## Algorithm: KAS

# Result type

"""
    KASResult{
        TUb<:Union{AbstractFloat,Complex{<:AbstractFloat}},
        TU<:AbstractVector{<:AbstractMatrix{TUb}},
        Tb<:AbstractVector{<:AbstractVector{TUb}},
        Tc<:AbstractVector{<:Integer},
        T<:Real}

The output of [`kas`](@ref).

# Fields
- `U::TU`: vector of affine space basis matrices `U[1],...,U[K]`
- `b::Tb`: vector of bias vectors `b[1],...,b[K]`
- `assignments::Tc`: vector of cluster assignments `assignments[1],...,assignments[N]`
- `iterations::Int`: number of iterations performed
- `totalcost::T`: final value of total cost function
- `counts::Vector{Int}`: vector of cluster sizes `counts[1],...,counts[K]`
- `converged::Bool`: final convergence status
"""
struct KASResult{
    TUb<:Union{AbstractFloat,Complex{<:AbstractFloat}},
    TU<:AbstractVector{<:AbstractMatrix{TUb}},
    Tb<:AbstractVector{<:AbstractVector{TUb}},
    Tc<:AbstractVector{<:Integer},
    T<:Real,
}
    U::TU
    b::Tb
    assignments::Tc
    iterations::Int
    totalcost::T
    counts::Vector{Int}
    converged::Bool
end

function show(io::IO, ::MIME"text/plain", result::KASResult)
    println(
        io,
        " KASResult ($(length(result.counts)) clusters, $(length(result.assignments)) cluster assignments)",
    )
    println(io)

    assignments_preview =
        length(result.assignments) > 10 ?
        string("[", join(result.assignments[1:10], ","), ", ...]") :
        string(result.assignments)

    println(io, " assignments       :   ", assignments_preview)
    println(io)
    println(io, " Additional Fields: ")
    println(io)
    println(io, " counts            :   ", result.counts)
    println(io, " iterations        :   ", result.iterations)
    println(io, " converged         :   ", result.converged)
    println(io, " U                 ::  ", typeof(result.U))
    println(io, " b                 ::  ", typeof(result.b))
    println(io, " totalcost         ::  ", typeof(result.totalcost))
end

# Main function

"""
    kas(X::AbstractMatrix{<:Number}, d::AbstractVector{<:Integer};
        maxiters = 100,
        rng = default_rng(),
        init = [(randsubspace(rng, float(eltype(X)), size(X, 1), di), zeros(float(eltype(X)), size(X, 1))) for di in d],
        showprogress = false)

Cluster the `N` data points in the `D×N` data matrix `X`
into `K` clusters via the **K**-**a**ffine-**s**paces (KAS) algorithm
with corresponding affine space dimensions `d[1],...,d[K]`.
Output is a [`KASResult`](@ref) containing the resulting
cluster assignments `assignments[1],...,assignments[N]`,
affine space basis matrices `U[1],...,U[K]`,
bias vectors `b[1],...,b[K]`,
and metadata about the algorithm run.

KAS seeks to cluster data points by their affine space
by minimizing the following total cost
```math
\\sum_{i=1}^N \\| X[:, i] - (U[assignments[i]] U[assignments[i]]' (X[:, i] - b[assignments[i]]) + b[assignments[i]]) \\|_2^2
```
with respect to the cluster assignments `assignments[1],...,assignments[N]`,
affine space basis matrices `U[1],...,U[K]`,
and bias vectors `b[1],...,b[K]`.

# Keyword arguments
- `maxiters::Integer = 100`: maximum number of iterations
- `rng::AbstractRNG = default_rng()`: random number generator
    (used when reinitializing the affine space for an empty cluster)
- `init::AbstractVector{<:Tuple{<:AbstractMatrix{TUb},<:AbstractVector{TUb}}}
    = [(randsubspace(rng, float(eltype(X)), size(X, 1), di), zeros(float(eltype(X)), size(X, 1))) for di in d]`:
    vector of `K` initial pair of affine space basis matrices containing `U[1],...,U[K]`
    and bias vectors containing `b[1],...,b[K]`
    where `TUb` is a floating point type.
- `showprogress::Bool = false`: whether to log progress during the algorithm run

See also [`KASResult`](@ref).
"""
function kas(
    X::AbstractMatrix{<:Number},
    d::AbstractVector{<:Integer};
    maxiters::Integer = 100,
    rng::AbstractRNG = default_rng(),
    init::AbstractVector{<:Tuple{<:AbstractMatrix{TUb},<:AbstractVector{TUb}}} = [
        (
            randsubspace(rng, float(eltype(X)), size(X, 1), di),
            zeros(float(eltype(X)), size(X, 1)),
        ) for di in d
    ],
    showprogress::Bool = false,
) where {TUb<:Union{AbstractFloat,Complex{<:AbstractFloat}}}
    # Unpack the initial affine space basis matrices and bias vectors
    Uinit = first.(init)
    binit = last.(init)

    # Require one-based indexing
    Base.require_one_based_indexing(X, d, Uinit, binit)
    for Uk in Uinit
        Base.require_one_based_indexing(Uk)
    end
    for bk in binit
        Base.require_one_based_indexing(bk)
    end

    # Extract sizes and check that they agree
    K = (only ∘ unique)([length(d), length(Uinit), length(binit)])
    D = (only ∘ unique)([size(X, 1); size.(Uinit, 1); length.(binit)])

    # Check affine space dimensions
    for k in 1:K
        d[k] == size(Uinit[k], 2) || throw(
            ArgumentError(
                "Basis matrix initialization `Uinit[$k]` must have `d[$k]=$(d[k])` columns.",
            ),
        )
        0 <= d[k] <= D || throw(
            DimensionMismatch(
                "Affine space dimension `d[$k]=$(d[k])` must be between `0` and `D=$D`.",
            ),
        )
        length(binit[k]) == D || throw(
            ArgumentError(
                "Bias vector initialization `binit[$k]` must be of length `D=$D`.",
            ),
        )
    end

    # Check maxiters
    maxiters >= 0 || throw(
        ArgumentError(
            "Maximum number of iterations must be nonnegative. Got `maxiters=$maxiters`.",
        ),
    )

    # Initialize model parameters
    U = deepcopy(Uinit)
    b = deepcopy(binit)
    assignments = kas_assign_clusters(U, b, X)

    # Main loop
    cprev = copy(assignments)
    iterations, converged = 0, false
    log_every = max(1, maxiters ÷ 100)
    @withprogressif showprogress while iterations < maxiters && !converged
        iterations += 1

        # Update affine space basis matrices and bias vectors
        for k in 1:K
            inds = findall(==(k), assignments)
            if !isempty(inds)
                U[k], b[k] = kas_estimate_affinespace(view(X, :, inds), d[k])
            else
                @warn "Empty cluster detected at iteration $iterations - reinitializing the affine space. Consider reducing the number of clusters."
                randsubspace!(rng, U[k])
                fill!(b[k], zero(eltype(b[k])))
            end
        end

        # Update cluster assignments
        kas_assign_clusters!(assignments, U, b, X)

        # Check for convergence
        if cprev == assignments
            @info "Converged after $iterations $(iterations == 1 ? "iteration" : "iterations")."
            converged = true
        end
        copyto!(cprev, assignments)

        # Log progress
        if iterations % log_every == 0
            @logprogressif showprogress iterations / maxiters
        end
    end

    # Compute final counts and costs
    counts = [count(==(k), assignments) for k in 1:K]
    costs = [
        sum(abs2, (xi - b[assignments[i]])) -
        sum(abs2, U[assignments[i]]' * (xi - b[assignments[i]])) for
        (i, xi) in pairs(eachcol(X))
    ]

    return KASResult(U, b, assignments, iterations, sum(costs), counts, converged)
end

# Subroutines

"""
    kas_assign_clusters(U, b, X)

Assign the `N` data points in `X` to the `K` affine spaces in `(U,b)`
and return a vector of the assignments.

See also [`kas_assign_clusters!`](@ref), [`kas`](@ref).
"""
kas_assign_clusters(U, b, X) =
    kas_assign_clusters!(similar(Vector{Int}, (axes(X, 2),)), U, b, X)

"""
    kas_assign_clusters!(assignments, U, b, X)

Assign the `N` data points in `X` to the `K` affine spaces in `(U,b)`,
update the vector `assignments`,
and return the final vector of assignments.

See also [`kas_assign_clusters`](@ref), [`kas`](@ref).
"""
function kas_assign_clusters!(assignments, U, b, X)
    for (i, xi) in pairs(eachcol(X))
        assignments[i] = argmin(
            sum(abs2, (xi - b[k])) - sum(abs2, U[k]' * (xi - b[k])) for k in eachindex(U)
        )
    end
    return assignments
end

"""
    kas_estimate_affinespace(Xk, dk)

Return `dk`-dimensional affine space that best fits the data points in `Xk`.

See also [`kas`](@ref).
"""
function kas_estimate_affinespace(Xk, dk)
    bhat = mean(eachcol(Xk))
    Uhat = svd!(Xk .- bhat; full = true).U[:, 1:dk]
    return Uhat, bhat
end
