## Algorithm: KAS

# Result type

"""
    KASResult{
        TU<:AbstractVector{<:AbstractMatrix{<:AbstractFloat}},
        Tb<:AbstractVector{<:AbstractVector{<:AbstractFloat}},
        Tc<:AbstractVector{<:Integer},
        T<:Real}


The output of [`kas`](@ref).

# Fields
- `U::TU`: vector of affine space basis matrices `U[1],...,U[K]`
- `b::Tb`: vector of base vectors `b[1],...,b[K]`
- `c::Tc`: vector of cluster assignments `c[1],...,c[N]`
- `iterations::Int`: number of iterations performed
- `totalcost::T`: final value of total cost function
- `counts::Vector{Int}`: vector of cluster sizes `counts[1],...,counts[K]`
- `converged::Bool`: final convergence status
"""

struct KASResult{
    TU<:AbstractVector{<:AbstractMatrix{<:AbstractFloat}},
    Tb<:AbstractVector{<:AbstractVector{<:AbstractFloat}},
    Tc<:AbstractVector{<:Integer},
    T<:Real,
}
    U::TU
    b::Tb
    c::Tc
    iterations::Int
    totalcost::T
    counts::Vector{Int}
    converged::Bool
end

"""
    kas(X::AbstractMatrix{<:Real}, d::AbstractVector{<:Integer};
        maxiters = 100,
        rng = default_rng(),
        init = [randaffinespace(rng, size(X, 1), di) for di in d])

Cluster the `N` data points in the `D×N` data matrix `X`
into `K` clusters via the K-Affinespaces (KAS) algorithm
with corresponding affine space dimensions `d[1],...,d[K]`.
Output is a [`KASResult`](@ref) containing the resulting
cluster assignments `c[1],...,c[N]`,
affine space basis matrices `U[1],...,U[K]`, bias vectors `b[1],...,b[K]`,
and metadata about the algorithm run.

KAS seeks to cluster data points by their affine space
by minimizing the following total cost
```math
\\sum_{i=1}^N \\| (X[:, i]  - [U[c[i]] U[c[i]]' (X[:, i] - b[c[i]]) + b[c[i]])] \\|_2^2
```
with respect to the cluster assignments `c[1],...,c[N]`, affine space basis matrices `U[1],...,U[K]`, and bias vectors `b[1],...,b[K]`.

# Keyword arguments
- `maxiters::Integer = 100`: maximum number of iterations
- `rng::AbstractRNG = default_rng()`: random number generator
    (used when reinitializing the affine space for an empty cluster)
- `init::AbstractVector{<:Tuple{<:AbstractMatrix{<:AbstractFloat}, <:AbstractVector{<:AbstractFloat}}}
    = [randaffinespace(rng, size(X, 1), di) for di in d]`:
    vector of `K` initial pair of affine space basis matrices containing `U[1],...,U[K]`
    and bias vectors containing `b[1],...,b[K]`.

See also [`KASResult`](@ref).
"""

function kas(
    X::AbstractMatrix{<:Real}, 
    d::AbstractVector{<:Integer}; 
    maxiters::Integer = 100,
    rng::AbstractRNG = default_rng(),
    init::AbstractVector{<:Tuple{<:AbstractMatrix{<:AbstractFloat}, <:AbstractVector{<:AbstractFloat}}} = [
        randaffinespace(rng, size(X, 1), di) for di in d
    ]
)

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
	K = (only ∘ unique)([length(d), length(binit)])
    D = (only ∘ unique)([size(X, 1); size.(Uinit, 1)])

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
                "Bias vector initialiation `binit[$k]` must be of length `D=$D`.",
            ),
        )
    end

    # Check maxiters
    maxiters >= 0 || throw(
        ArgumentError(
            "Maximum number of iterations must be nonnegative. Got `maxiters=$maxiters`.",
        ),
    )

	# Initialize Model parameters
	U = deepcopy(Uinit)
	b = deepcopy(binit)
    c = kas_assign_clusters(U, b, X)
	

	# Main loop
    cprev = copy(c)
    iterations, converged = 0, false
	@withprogress while iterations < maxiters && !converged
        iterations += 1

		# Update Affine space basis and base vectors
		for k in 1:K
			inds = findall(==(k), c)
			if !isempty(inds)
				U[k], b[k] = kas_estimate_affinespace(view(X, :, inds), d[k])
			else
				@warn "Empty cluster detected at iteration $iterations - reinitializing the affine space. Consider reducing the number of clusters."
                U[k], b[k] = randaffinespace(rng, D, d[k])
			end
		end

		# Update cluster assignments
		kas_assign_clusters!(c, U, b, X)

        # Check for convergence
        if cprev == c
            @info "Converged after $iterations $(iterations == 1 ? "iteration" : "iterations")."
            converged = true
        end
        copyto!(cprev, c)

        # Log progress
        if iterations % (maxiters ÷ 100) == 0
            @logprogress iterations / maxiters
        end
	end

    # Compute final counts and costs
    counts = [count(==(k), c) for k in 1:K]
    costs = [sum(abs2, (xi - b[c[i]])) - sum(abs2, U[c[i]]' * (xi - b[c[i]])) for (i, xi) in pairs(eachcol(X))]

	return KASResult(U, b, c, iterations, sum(costs), counts, converged)
end

"""
    kas_assign_clusters(U, X)

Assign the `N` data points in `X` to the `K` affine spaces in (`U`, `b`)
and return a vector of the assignments.

See also [`kas_assign_clusters!`](@ref), [`kas`](@ref).
"""

kas_assign_clusters(U, b, X) = kas_assign_clusters!(similar(Vector{Int}, (axes(X, 2),)), U, b, X)

"""
    kas_assign_clusters!(c, U, X)

Assign the `N` data points in `X` to the `K` affine spaces in (`U`, `b`),
update the vector of assignments `c`,
and return this vector of assignments.

See also [`kas_assign_clusters`](@ref), [`kas`](@ref).
"""

function kas_assign_clusters!(c, U, b, X)
    for (i, xi)  in pairs(eachcol(X))
        c[i] = argmin(sum(abs2, (xi - b[k])) - sum(abs2, U[k]' * (xi - b[k])) for k in eachindex(U))
    end
    return c
end

"""
    kas_estimate_affinespace(Xk, dk)

Return `dk`-dimensional affine space that best fits the data points in `Xk`.

See also [`kas`](@ref).
"""

function kas_estimate_affinespace(Xk, dk)
    bhat = mean(eachcol(Xk))
    Uhat = svd(Xk - bhat*ones(size(Xk, 2))'; full=true).U[:, 1:dk]
    return Uhat, bhat
end