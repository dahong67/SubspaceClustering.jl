## Algorithm: KSS

# Result type
"""
    KSSResult{M<:AbstractMatrix{<:AbstractFloat}, T<:Real}

The output of K-subspaces clustering algorithm.

#Type Parameters
- `M<:AbstractMatrix{<:AbstractFloat}`: Type of the subspace basis matrices.
- `T<:Real`: Type of the total cost of the clustering.


# Fields
- `U::Vector{M}`: Subspace bases for each cluster.
- `c::Vector{Int}`: Cluster assignments for each data point.
- `iterations::Int`: Number of iterations performed.
- `totalcost::T`: Total cost of the clustering.
- `counts::Vector{Int}`: Number of data points in each cluster.
- `converged::Bool`: Convergence status.
"""

struct KSSResult{M<:AbstractMatrix{<:AbstractFloat},T<:Real}
    U::Vector{M} # Subspace bases for each cluster
    c::Vector{Int} # Cluster assignments for each data point
    iterations::Int # Number of iterations performed
    totalcost::T # Total cost of the clustering
    counts::Vector{Int} # Number of data points in each cluster
    converged::Bool # Convergence status
end

# Main function
"""
    kss(X, d; niters=100, rng::AbstractRNG = default_rng(), Uinit::AbstractVector{<:AbstractMatrix{<:Real}} = [randsubspace(rng, size(X, 1), d_i) for d_i in d])

Run K-subspaces on the data matrix `X` with subspace dimensions `d[1], ..., d[K]`.

# Arguments
- `X::AbstractMatrix`: Data matrix of size `(D, N)`. Each column represents a data point.
- `d::Vector{Int}`: A vector of length `K` containing the subspace dimensions for each of the 'K' subspaces.

# Keyword Arguments
- `niters::Int=100`: Maximum number of iterations.
- `rng::AbstractRNG=default_rng()`: Random Number generator with AbstractRNG type.
- `Uinit::Vector{AbstractMatrix}`: A vector of length `K` containing initial subspace bases. If not provided, they are initialized randomly via `randsubspace`.

# Returns
A [`KSSResult`](@ref KSSResult) struct containing the clustering result including:
    - The computed subspace bases `U`.
    - The cluster assignments `c`.
    - The number of iterations performed `iterations`.
    - The total cost of the clustering `totalcost`.
    - The number of data points in each cluster `counts`.
    - The convergence status `converged`.
"""
function kss(
    X::AbstractMatrix{T},                                                                             #in: data matrix with size (D, N)
    d::Vector{<:Integer};                                                                                  #in: a vector of subspace dimensions of length K
    niters::Integer = 100,                                                                   #in: maximum number of iterations
    rng::AbstractRNG = default_rng(),                                                                    #in: a random number generator with AbstractRNG type
    Uinit::Vector{M} = [randsubspace(rng, size(X, 1), d_i) for d_i in d],    #in: a vector of length K containing initial subspaces
) where {T<:Real,M<:AbstractMatrix{T}}
    U = deepcopy(Uinit)

    # Check arguments
    D = size(X, 1) #Feature space dimension
    niters > 0 ||
        throw(ArgumentError("Number of iterations must be positive. Got: $niters"))
    all(d_i -> d_i <= D, d) || throw(
        DimensionMismatch("Subspace Dimensions are greater than Feature space Dimensions"),
    )
    all(d_i -> d_i >= 0, d) ||
        throw(ArgumentError("All subspace dimensions must be positive. Got: $d"))

    # Main calculation
    K = length(d)
    D, N = size(X)

    c = kss_assign_clusters(U, X)
    c_prev = copy(c)
    converged = false
    iter = niters # default to maximum iterations if no early convergence

    # Iterations
    @progress for t in 1:niters
        # Update subspaces
        for k in 1:K
            ilist = findall(==(k), c)
            @debug "Cluster $k got assigned $(length(ilist)) data points"

            if isempty(ilist)
                @warn "Empty clusters detected at iteration $t - reinitializing the subspace. Consider reducing the number of clusters."
                U[k] = randsubspace(rng, D, d[k])
            else
                U[k] = kss_estimate_subspace(view(X, :, ilist), d[k])
            end
        end

        # Update clusters and calculate total cost
        kss_assign_clusters!(c, U, X)

        # Break if clusters did not change, update otherwise
        if c == c_prev
            @info "Terminated early at iteration $t"
            converged = true
            iter = t
            break
        end
        c_prev .= c
    end

    #Count data points in each cluster
    counts = [count(x -> x == k, c) for k in 1:K]

    # Calculate total cost
    totalcost = 0
    for i in 1:N
        totalcost += norm(U[c[i]]' * view(X, :, i))
    end

    return KSSResult(U, c, iter, totalcost, counts, converged)
end

# Subroutines

"""
    kss_assign_clusters(U, X)

Assign the `N` data points in `X` to the `K` subspaces in `U`
and return a vector of the assignments.

See also [`kss_assign_clusters!`](@ref), [`kss`](@ref).
"""
kss_assign_clusters(U, X) = kss_assign_clusters!(Vector{Int}(undef, size(X, 2)), U, X)

"""
    kss_assign_clusters!(c, U, X)

Assign the `N` data points in `X` to the `K` subspaces in `U`,
update the vector of assignments `c`,
and return this vector of assignments.

See also [`kss_assign_clusters`](@ref), [`kss`](@ref).
"""
function kss_assign_clusters!(c, U, X)
    N = length(c)
    K = length(U)
    for i in 1:N
        c[i] = argmax(norm(U[k]' * view(X, :, i)) for k in 1:K)
    end
    return c
end

"""
    kss_estimate_subspace(Xk, dk)

Return `dk`-dimensional subspace that best fits the data points in `Xk`.

See also [`kss`](@ref).
"""
function kss_estimate_subspace(Xk, dk)
    A = Xk * transpose(Xk)
    decomp, history = partialschur(A; nev = dk, which = :LR)
    @debug "Cluster partialschur decomposition history:
    matrix-vector products: $(history.mvproducts),
    Number of eigenvalues: $(history.nev),
    number of converged eigenvalues: $(history.nconverged),
    converged? = $(history.converged)"
    return decomp.Q
end
