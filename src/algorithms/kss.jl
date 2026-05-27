## Algorithm: KSS

# Result type

"""
    KSSResult{
        TU<:AbstractVector{<:AbstractMatrix{<:Union{AbstractFloat,Complex{<:AbstractFloat}}}},
        Tc<:AbstractVector{<:Integer},
        T<:Real}

The output of [`kss`](@ref).

# Fields
- `U::TU`: vector of subspace basis matrices `U[1],...,U[K]`
- `assignments::Tc`: vector of cluster assignments `assignments[1],...,assignments[N]`
- `iterations::Int`: number of iterations performed
- `totalcost::T`: final value of total cost function
- `counts::Vector{Int}`: vector of cluster sizes `counts[1],...,counts[K]`
- `converged::Bool`: final convergence status
"""
struct KSSResult{
    TU<:AbstractVector{<:AbstractMatrix{<:Union{AbstractFloat,Complex{<:AbstractFloat}}}},
    Tc<:AbstractVector{<:Integer},
    T<:Real,
}
    U::TU
    assignments::Tc
    iterations::Int
    totalcost::T
    counts::Vector{Int}
    converged::Bool
end

function show(io::IO, ::MIME"text/plain", result::KSSResult)
    println(
        io,
        " KSSResult ($(length(result.counts)) clusters, $(length(result.assignments)) cluster assignments)",
    )
    println(io)

    assignments_preview =
        length(result.assignments) > 10 ?
        string("[", join(result.assignments[1:10], ","), ", ...]") :
        string(result.assignments)

    println(io, " assignments       :   ", assignments_preview)
    println(io)
    println(io, " Additional Fields:")
    println(io)
    println(io, " counts            :   ", result.counts)
    println(io, " iterations        :   ", result.iterations)
    println(io, " converged         :   ", result.converged)
    println(io, " U                 ::  ", typeof(result.U))
    return println(io, " totalcost         ::  ", typeof(result.totalcost))
end

# Main function

"""
    kss(X::AbstractMatrix{<:Number}, d::AbstractVector{<:Integer};
        maxiters = 100,
        rng = default_rng(),
        Uinit = [randsubspace(rng, size(X, 1), di) for di in d],
        showprogress = false)

Cluster the `N` data points in the `D×N` data matrix `X`
into `K` clusters via the **K**-**s**ub**s**paces (KSS) algorithm
with corresponding subspace dimensions `d[1],...,d[K]`.
Output is a [`KSSResult`](@ref) containing the resulting
cluster assignments `assignments[1],...,assignments[N]`,
subspace basis matrices `U[1],...,U[K]`,
and metadata about the algorithm run.

KSS seeks to cluster data points by their subspace
by minimizing the following total cost
```math
\\sum_{i=1}^N \\| X[:, i] - U[assignments[i]] U[assignments[i]]' X[:, i] \\|_2^2
```
with respect to the cluster assignments `assignments[1],...,assignments[N]`
and subspace basis matrices `U[1],...,U[K]`.

# Keyword arguments
- `maxiters::Integer = 100`: maximum number of iterations
- `rng::AbstractRNG = default_rng()`: random number generator
    (used when reinitializing the subspace for an empty cluster)
- `Uinit::AbstractVector{<:AbstractMatrix{T}}
    = [randsubspace(rng, float(eltype(X)), size(X, 1), di) for di in d]`:
    vector of `K` initial subspace basis matrices to use
    (each `Uinit[k]` should be `D×d[k]` and have eltype `T`
    where `T` is a floating point type)
- `showprogress::Bool = false`: whether to log progress during the algorithm run

See also [`KSSResult`](@ref).
"""
function kss(
    X::AbstractMatrix{<:Number},
    d::AbstractVector{<:Integer};
    maxiters::Integer = 100,
    rng::AbstractRNG = default_rng(),
    Uinit::AbstractVector{
        <:AbstractMatrix{<:Union{AbstractFloat,Complex{<:AbstractFloat}}},
    } = [randsubspace(rng, float(eltype(X)), size(X, 1), di) for di in d],
    showprogress::Bool = false,
)
    # Require one-based indexing
    Base.require_one_based_indexing(X, d, Uinit)
    for Uk in Uinit
        Base.require_one_based_indexing(Uk)
    end

    # Extract sizes and check that they agree
    K = (only ∘ unique)([length(d), length(Uinit)])
    D = (only ∘ unique)([size(X, 1); size.(Uinit, 1)])

    # Check subspace dimensions
    for k in 1:K
        d[k] == size(Uinit[k], 2) || throw(
            ArgumentError(
                "Basis matrix initialization `Uinit[$k]` must have `d[$k]=$(d[k])` columns.",
            ),
        )
        0 <= d[k] <= D || throw(
            DimensionMismatch(
                "Subspace dimension `d[$k]=$(d[k])` must be between `0` and `D=$D`.",
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
    assignments = kss_assign_clusters(U, X)

    # Main loop
    cprev = copy(assignments)
    iterations, converged = 0, false
    log_every = max(1, maxiters ÷ 100)
    @withprogressif showprogress while iterations < maxiters && !converged
        iterations += 1

        # Update subspaces
        for k in 1:K
            inds = findall(==(k), assignments)
            if !isempty(inds)
                U[k] = kss_estimate_subspace(view(X, :, inds), d[k])
            else
                @warn "Empty cluster detected at iteration $iterations - reinitializing the subspace. Consider reducing the number of clusters."
                randsubspace!(rng, U[k])
            end
        end

        # Update cluster assignments
        kss_assign_clusters!(assignments, U, X)

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
        sum(abs2, xi) - sum(abs2, U[assignments[i]]' * xi) for (i, xi) in pairs(eachcol(X))
    ]

    return KSSResult(U, assignments, iterations, sum(costs), counts, converged)
end

# Subroutines

"""
    kss_assign_clusters(U, X)

Assign the `N` data points in `X` to the `K` subspaces in `U`
and return a vector of the assignments.

See also [`kss_assign_clusters!`](@ref), [`kss`](@ref).
"""
kss_assign_clusters(U, X) = kss_assign_clusters!(similar(Vector{Int}, (axes(X, 2),)), U, X)

"""
    kss_assign_clusters!(assignments, U, X)

Assign the `N` data points in `X` to the `K` subspaces in `U`,
update the vector `assignments`,
and returns the final vector of cluster assignments.

See also [`kss_assign_clusters`](@ref), [`kss`](@ref).
"""
function kss_assign_clusters!(assignments, U, X)
    for (i, xi) in pairs(eachcol(X))
        assignments[i] = argmax(sum(abs2, U[k]' * xi) for k in eachindex(U))
    end
    return assignments
end

"""
    kss_estimate_subspace(Xk, dk)

Return `dk`-dimensional subspace that best fits the data points in `Xk`.

See also [`kss`](@ref).
"""
function kss_estimate_subspace(Xk, dk)
    decomp, history = partialschur(Xk * Xk'; nev = dk, which = :LR)
    history.converged ||
        @warn "Iterative algorithm for subspace update did not converge - results may be inaccurate."
    return decomp.Q
end
