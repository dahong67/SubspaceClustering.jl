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

"""
    ekss(X::AbstractMatrix{<:Real}, d::Integer, K::Integer;
        Kbar::Integer = K,
        maxiters::Integer = 100,
        nruns::Integer = 100,
        kmeans_nruns::Integer = 50,
        q::Integer = 10,
        rng::AbstractRNG = default_rng())

# Keyword arguments
- `Kbar::Integer = K`: Number of candidate subspaces
- `maxiters::Integer = 100`: Maximum number of KSS iterations
- `nruns::Integer = 100`: Number of KSS runs/Base Clusterings
- `kmeans_nruns::Integer = 50`: Number of K-means runs to perform
- `q::Integer = 10`: Maximum number of neighbors or Thresholding parameter
- `rng::AbstractRNG = default_rng()`: random number generator
    (used when reinitializing the subspace for an empty cluster)

See also [`EKSSResult`](@ref).
"""

function ekss(
    X::AbstractMatrix{<:Real},
    d::Integer,
    K::Integer;
    Kbar::Integer = K,
    maxiters::Integer = 100,
    nruns::Integer = 100,
    kmeans_nruns::Integer = 50,
    q::Integer = 10,
    rng::AbstractRNG = default_rng(),
)

    N = size(X, 2)

    # Validate arguments
    Base.require_one_based_indexing(X)
    d > 0 || throw(ArgumentError("`d` must be positive. Got d=$d."))
    K > 0 || throw(ArgumentError("`K` must be positive. Got K=$K."))
    Kbar > 0 || throw(ArgumentError("`Kbar` must be positive. Got Kbar=$Kbar."))
    maxiters > 0 || throw(ArgumentError("`maxiters` must be positive. Got maxiters=$maxiters."))
    nruns > 0 || throw(ArgumentError("`nruns` must be positive. Got nruns=$nruns."))
    kmeans_nruns > 0 || throw(ArgumentError("`kmeans_nruns` must be positive. Got kmeans_nruns=$kmeans_nruns."))
    1 <= q <= N - 1 || throw(ArgumentError("`q` must be in 1:(N-1). Got q=$q, N=$N."))
    

    # Candidate subspace dimensions for each base KSS run
    dvec = fill(d, Kbar)

    # Run KSS Ensemble
    runs = Vector{KSSResult}(undef, nruns)
    for r in 1:nruns
        Uinit_r = [randsubspace(rng, size(X, 1), d) for _ in 1:Kbar]
        runs[r] = kss(X, dvec; maxiters=maxiters, rng=rng, Uinit=Uinit_r)
    end

    # Form Co-association matrix
    cluster_labels = [run.c for run in runs]
    C = co_association(cluster_labels)



end

# Co-Association matrix

function co_association(label_runs::AbstractVector{<:AbstractVector{<:Integer}})
    nruns = length(label_runs)
    nruns > 0 || throw(ArgumentError("Need at least one run to form co-association matrix."))

    N = length(label_runs[1])
    C = spzeros(Float64, N, N)

    rows = collect(1:N)
    vals = ones(Float64, N)

    for labels in label_runs
        length(labels) == N || throw(DimensionMismatch("All label vectors must have same length."))

        Kbar = maximum(labels)
        Z = sparse(rows, labels, vals, N, Kbar)

        C .+= Z * Z'
    end

    C ./= nruns

    return C
end

# Embedding

function embedding(A, K)
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

function affinity_thresh(A::SparseMatrixCSC{<:Real, <:Integer}, q::Integer)
    Base.require_one_based_indexing(A)

    Zcol = _topq_per_column(A, q)
    Zrow = _topq_per_column(A', q)'

    return 0.5 .* (Zrow .+ Zcol)
end

function _topq_per_column(A::SparseMatrixCSC{<:Real, <:Integer}, q::Integer)
    N = size(A, 1)
    size(A, 2) == N || throw(DimensionMismatch("`A` must be square. Got size(A)=$(size(A))."))
    1 <= q <= N - 1 || throw(ArgumentError("`q` must be in 1:(N-1). Got q=$q, N=$N."))

    rows_out = Int[]
    cols_out = Int[]
    vals_out = Float64[]

    rowinds = rowvals(A)
    nzvals = nonzeros(A)

    for j in 1:N
        rng = nzrange(A, j)

        # collect off-diagonal entries in column j
        rows_j = Int[]
        vals_j = Float64[]
        for p in rng
            i = rowinds[p]
            i == j && continue
            push!(rows_j, i)
            push!(vals_j, float(nzvals[p]))
        end

        isempty(vals_j) && continue

        qj = min(q, length(vals_j))
        inds = partialsortperm(vals_j, 1:qj; rev=true)

        append!(rows_out, rows_j[inds])
        append!(cols_out, fill(j, qj))
        append!(vals_out, vals_j[inds])
    end

    return sparse(rows_out, cols_out, vals_out, N, N)
end