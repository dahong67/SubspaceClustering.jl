## Algorithm: EKSS

# Result type

"""
    EKSSResult{
        TA<:AbstractMatrix{<:Real},
        TE<:AbstractMatrix{<:Real},
        TK<:KmeansResult,
        Tc<:AbstractVector{<:Integer}}

The output of [`ekss`](@ref).

# Fields
- `coassoc::TA`: `N×N` Thresholded Co-Association Matrix
- `embedding::TE`: `K×N` EKSS Embedding Matrix
- `kmeans_runs::Vector{TK}` : vector of outputs from batched K-means
- `assignments::Tc` : vector of final assignments
"""
struct EKSSResult{
    TA<:AbstractMatrix{<:Real},
    TE<:AbstractMatrix{<:Real},
    TK<:KmeansResult,
    Tc<:AbstractVector{<:Integer},
}
    coassoc::TA
    embedding::TE
    kmeans_runs::Vector{TK}
    assignments::Tc
end

# Main function

"""
    ekss(X::AbstractMatrix{<:Real}, d::Integer, K::Integer;
        Kbar::Integer = K,
        parallel::Bool = false,
        maxiters::Integer = 100,
        q::Union{Nothing,Integer} = nothing,
        rng::AbstractRNG = default_rng(),
        nruns::Integer = 50,
        kmeans_nruns::Integer = 50,
        kmeans_opts = (;))

Cluster the `N` data points in the `D×N` data matrix `X` using the
**Ensemble K-Subspaces (EKSS)** algorithm.

EKSS improves the robustness of subspace clustering by running multiple
independent **K-Subspaces (KSS)** base clusterings and combining their
results through a **co-association matrix**, followed by **thresholding**
and **spectral clustering**.

Output is a [`EKSSResult`](@ref) containing the resulting cluster assignments
with the internally computed co-association matrix, embedding matrix, and K-means runs.

The algorithm proceeds as follows:

1. Run `nruns` independent KSS clusterings with `Kbar` candidate subspaces
   of dimension `d`.
2. Construct a co-association matrix measuring how frequently pairs of
   points are assigned to the same cluster across runs.
3. Threshold the co-association matrix by keeping the top `q` neighbors
   per point.
4. Perform spectral clustering on the thresholded co-association matrix to obtain
   the final `K` clusters.

# Keyword arguments
- `Kbar::Integer = K`: Number of candidate subspaces
- `parallel::Bool = false`: If `true`, the ensemble base KSS runs are executed in parallel using Julia multi-threading.
- `maxiters::Integer = 100`: Maximum number of KSS iterations
- `q::Union{Nothing,Integer} = nothing`: Threshold parameter controlling the number
  of neighbors retained per point. If `nothing`, a default is chosen automatically
  from the number of data points.
- `rng::AbstractRNG = default_rng()`: random number generator
- `nruns::Integer = 50`: Number of KSS runs/Base Clusterings
- `kmeans_nruns::Integer = 50`: Number of K-means runs to perform
- `kmeans_opts = (;)`: additional options for `kmeans`

See also [`EKSSResult`](@ref), [`ekss_affinity`](@ref), [`embedding`](@ref).
"""
function ekss(
    X::AbstractMatrix{<:Real},
    d::Integer,
    K::Integer;
    Kbar::Integer = K,
    parallel::Bool = false,
    maxiters::Integer = 100,
    q::Union{Nothing,Integer} = nothing,
    rng::AbstractRNG = default_rng(),
    nruns::Integer = 50,
    kmeans_nruns::Integer = 50,
    kmeans_opts = (;),
)

    N = size(X, 2)
    q = isnothing(q) ? max(10, min(cld(N, 20), 100)) : q

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
    seeds = rand(rng, UInt, nruns)
    if parallel
        Threads.@threads for r in 1:nruns
            rng_r = MersenneTwister(seeds[r])
            Uinit_r = [randsubspace(rng_r, size(X, 1), d) for _ in 1:Kbar]
            runs[r] = kss(X, dvec; maxiters=maxiters, rng=rng_r, Uinit=Uinit_r)
        end
    else
        for r in 1:nruns
            rng_r = MersenneTwister(seeds[r])
            Uinit_r = [randsubspace(rng_r, size(X, 1), d) for _ in 1:Kbar]
            runs[r] = kss(X, dvec; maxiters=maxiters, rng=rng_r, Uinit=Uinit_r)
        end
    end

    cluster_labels = [run.c for run in runs]

    @info "Forming Thresholded Co-Association Matrix with top $q values"
    A = ekss_affinity(cluster_labels; q=q, N=N, nruns=nruns)

    @info "Computing embedding"
    E = embedding(A, K)

    # Compute cluster assignments via batched K-means
    @info "Running batched K-means with $kmeans_nruns runs"
    results = @withprogress map(1:kmeans_nruns) do run
        result = kmeans(E, K; rng, kmeans_opts...)
        @logprogress run / kmeans_nruns
        return result
    end

    # Extract assignments from best K-means run and return EKSSResult
    assignments = argmin(result -> result.totalcost, results).assignments

    return EKSSResult(A, E, results, assignments)
end

# Subroutines

"""
    ekss_affinity(label_runs; q, N, nruns, max_chunksize=1000)

Form sparse co-association matrix by computing the clustering frequency between all pairs of points
and thresholding each column by retaining only top `q` largest entries and then symmetrizing the result
by averaging thresholded rows and columns. 

To handle datasets with a large number of points `N`, the computation is performed
over chunks of at most `max_chunksize` points at a time.

See also [`ekss`](@ref).
"""
function ekss_affinity(
    label_runs::AbstractVector{<:AbstractVector{<:Integer}};
    q::Integer,
    N::Integer,
    nruns::Integer,
    max_chunksize::Integer = 1000,
)
    1 <= q <= N - 1 || throw(ArgumentError("`q` must be in 1:(N-1). Got q=$q, N=$N."))

    length(label_runs) == nruns || throw(DimensionMismatch("Expected nruns=$nruns label runs, got $(length(label_runs))."))

    for labels in label_runs
        length(labels) == N ||
            throw(DimensionMismatch("All label vectors must have same length."))
    end

    # Chunking
    chunksize = min(max_chunksize, N)
    chunks = Iterators.partition(1:N, chunksize)

    # Sorting Buffer
    s_buf = Vector{Int}(undef, N)

    # Collect Sparse Triplets
    Z_nzs = @withprogress mapreduce(vcat, enumerate(chunks)) do (chunk_idx, chunk)

        m = length(chunk)
        C_chunk = zeros(Float64, N, m)

        # Forming Co-Association Matrix 
        for labels in label_runs
            Kbar = maximum(labels)
            clusters = [Int[] for _ in 1:Kbar]
            @inbounds for i in 1:N
                push!(clusters[labels[i]], i)
            end

            @inbounds for (local_j, j) in enumerate(chunk)
                lbl = labels[j]
                idx = clusters[lbl]
                for i in idx
                    C_chunk[i, local_j] += 1
                end
            end
        end

        # Average 
        C_chunk ./= nruns

        # Threshold each column in this chunk by keeping top-q values
        Z_nzs_chunk = map(zip(chunk, eachcol(C_chunk))) do (j, c)

            # Zero-ing out Diagonal Values
            c[j] = 0.0

            inds = partialsortperm!(s_buf, c, 1:q; rev = true)

            return (;
                rows = copy(inds),
                cols = fill(j, q),
                vals = copy(view(c, inds)),
            )
        end

        @logprogress chunk_idx / cld(N, chunksize)
        return Z_nzs_chunk
    end

    # assemble sparse matrix
    Z_rows = reduce(vcat, getindex.(Z_nzs, :rows))
    Z_cols = reduce(vcat, getindex.(Z_nzs, :cols))
    Z_vals = reduce(vcat, getindex.(Z_nzs, :vals))

    A = sparse([Z_rows; Z_cols],
               [Z_cols; Z_rows],
               [Z_vals; Z_vals],
               N, N, +)

    return 0.5 .* A
end


"""
    embedding(A, K)

Compute the `K`-dimensional embedding for the `N×N` Co-Association matrix `A`,
returning a `K×N` matrix of embeddings.
"""
function embedding(A, K)
    # Compute node degrees and form Laplacian matrix `L` from Co-Association Matrix `A`
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

