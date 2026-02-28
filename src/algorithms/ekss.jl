## Algorithm: EKSS

# Result type

"""
    EKSSResult{
        TU<:AbstractVector{<:AbstractMatrix{<:AbstractFloat}},
        Tc<:AbstractVector{<:Integer},
        T<:Real}

The output of [`ekss`](@ref).

# Fields
- `U::TU`: vector of ensemble subspace basis matrices `U[1],...,U[K]`
- `c::Tc`: vector of cluster assignments `c[1],...,c[N]`
- `iterations::Int`: number of iterations performed
- `totalcost::T`: final value of total cost function
- `counts::Vector{Int}`: vector of cluster sizes `counts[1],...,counts[K]`
- `converged::Bool`: final convergence status
"""

struct EKSSResult{
    TU<:AbstractVector{<:AbstractMatrix{<:AbstractFloat}},
    Tc<:AbstractVector{<:Integer},
    T<:Real,
}
    U::TU
    c::Tc
    iterations::Int
    totalcost::T
    counts::Vector{Int}
    converged::Bool
end

function ekss(
    X::AbstractMatrix{<:Real},
    d::AbstractVector{<:Integer};
    maxiters::Integer = 100,
    nruns::Integer = 100,
    rng::AbstractRNG = default_rng(),
)

    Base.require_one_based_indexing(X, d)
    nruns > 0 || throw(ArgumentError("`nruns` must be positive. Got nruns=$nruns."))
    runs = Vector{KSSResult}(undef, nruns)

    for r in 1:nruns
        Uinit_r = [randsubspace(rng, size(X, 1), di) for di in d]
        runs[r] = kss(X, d; maxiters=maxiters, rng=rng, Uinit=Uinit_r)
    end

    cluster_labels = [r.c for r in runs]
    C = co_association(cluster_labels)



end

# Co-Association matrix

function co_association(label_runs::AbstractVector{<:AbstractVector{<:Integer}})
    
    nruns = length(label_runs)
    nruns > 0 || throw(ArgumentError("Need at least one run to form co-association matrix."))
    N = length(label_runs[1])
    C = zeros(Float64, N, N)

    for labels in label_runs
        length(labels) == N || throw(DimensionMismatch("All label vectors must have same length."))

        K = maximum(labels)
        Z = zeros(Float64, N, K)
        for i in 1:N
            Z[i, labels[i]] = 1.0
        end

        C .+= Z * Z'
    end

    C ./= nruns

    return C

end

# Embedding

function embedding(A, k)
    # Compute node degrees and form Laplacian matrix `L` from `A`
    D = Diagonal(vec(sum(A; dims = 2)))
    L = Symmetric(I - (inv(sqrt(D)) * A * inv(sqrt(D))))

    # Compute eigenvectors corresponding to `K` smallest eigenvalues
    decomp, history = partialschur(L; nev = K, which = :SR)
    history.converged ||
        @warn "Iterative algorithm for eigenvectors did not converge - results may be inaccurate."

    # Permute and normalize to obtain embeddings
    E = mapslices(normalize, permutedims(decomp.Q); dims = 1)

    # Return the embeddings
    return E
end

# Thresholded Affinity matrix

function affinity_thresh(A::AbstractMatrix{<:Real}, q::Integer)

end