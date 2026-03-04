## Algorithm: TSC

# Result type

""""
    TSCResult{
        TA<:AbstractMatrix{<:Real},
        TE<:AbstractMatrix{<:Real},
        TK<:KmeansResult,
        Tc<:AbstractVector{<:Integer}}

The output of [`tsc`](@ref).

# Fields
- `affinity::TA` : `N×N` TSC affinity matrix
- `embedding::TE` : `K×N` TSC embedding matrix
- `kmeans_runs::Vector{TK}` : vector of outputs from batched K-means
- `assignments::Tc` : vector of final assignments
"""
struct TSCResult{
    TA<:AbstractMatrix{<:Real},
    TE<:AbstractMatrix{<:Real},
    TK<:KmeansResult,
    Tc<:AbstractVector{<:Integer},
}
    affinity::TA
    embedding::TE
    kmeans_runs::Vector{TK}
    assignments::Tc
end

# Main function

"""
    tsc(X::AbstractMatrix{<:Real}, K::Integer;
        max_nz = max(4, cld(size(X, 2), max(1, K))),
        max_chunksize = 1000,
        rng = default_rng(),
        kmeans_nruns = 10,
        kmeans_opts = (;))

Cluster the `N` data points in the `D×N` data matrix `X` into `K` clusters
via the **T**hresholding-based **S**ubspace **C**lustering (TSC) algorithm
with affinity matrix formed using at most `max_nz` neighbors.
Output is a [`TSCResult`](@ref) containing the resulting cluster assignments
with the internally computed affinity matrix, embedding matrix, and K-means runs.

TSC seeks to cluster data points by treating them as nodes of a weighted graph
with weights given by a thresholded affinity matrix formed by thresholding the
(transformed) absolute cosine similarities between every pair of points
at `max_nz` neighbors then symmetrizing. Cluster assignments are then obtained
via normalized spectral clustering of the graph.

# Keyword arguments
- `max_nz::Integer = max(4, cld(size(X, 2), max(1, K)))`: maximum number of neighbors
- `max_chunksize::Integer = 1000`: chunk size used in [`tsc_affinity`](@ref)
- `rng::AbstractRNG = default_rng()`: random number generator used by K-means
- `kmeans_nruns::Integer = 10`: number of K-means runs to perform
- `kmeans_opts = (;)`: additional options for `kmeans`

See also [`TSCResult`](@ref), [`tsc_affinity`](@ref), [`tsc_embedding`](@ref).
"""
function tsc(
    X::AbstractMatrix{<:Real},
    K::Integer;
    max_nz::Integer = max(4, cld(size(X, 2), max(1, K))),
    max_chunksize::Integer = 1000,
    rng::AbstractRNG = default_rng(),
    kmeans_nruns::Integer = 10,
    kmeans_opts = (;),
)
    # Validate arguments
    Base.require_one_based_indexing(X)
    1 <= K <= size(X, 2) ||
        throw(ArgumentError("`K` must be between 1 and `N=$(size(X,2))`."))
    max_nz >= 1 || throw(
        ArgumentError("Maximum number of nonzeros must be positive. Got `max_nz=$max_nz`."),
    )
    max_chunksize >= 1 || throw(
        ArgumentError(
            "Maximum chunk size must be positive. Got `max_chunksize=$max_chunksize`.",
        ),
    )
    kmeans_nruns >= 1 || throw(
        ArgumentError(
            "Number of K-means runs must be positive. Got `kmeans_nruns=$kmeans_nruns`.",
        ),
    )

    # Form affinity matrix
    @info "Forming affinity matrix"
    A = tsc_affinity(X; max_nz, max_chunksize)

    # Compute embedding
    @info "Computing embedding"
    E = tsc_embedding(A, K)

    # Compute cluster assignments via batched K-means
    @info "Running batched K-means with $kmeans_nruns runs"
    results = @withprogress map(1:kmeans_nruns) do run
        result = kmeans(E, K; rng, kmeans_opts...)
        @logprogress run / kmeans_nruns
        return result
    end

    # Extract assignments from best K-means run and return TSCResult
    assignments = argmin(result -> result.totalcost, results).assignments
    return TSCResult(A, E, results, assignments)
end

# Subroutines

"""
    tsc_affinity(X; max_nz = max(2, cld(size(X, 2), 4)), max_chunksize = 1000)

Compute the sparse TSC affinity (i.e., adjacency) matrix for the `N` data points in `X`
formed by thresholding their pairwise absolute cosine similarities at `max_nz` neighbors
then symmetrizing.

To handle datasets with a large number of points `N`, the computation is performed
over chunks of at most `max_chunksize` points at a time.

See also [`tsc`](@ref).
"""
function tsc_affinity(X; max_nz = max(2, cld(size(X, 2), 4)), max_chunksize = 1000)
    # Precompute normalized data points and extract needed dims
    Y = mapslices(normalize, X; dims = 1)
    N = size(X, 2)

    # Compute nonzero values of thresholded similarity matrix Z in chunks
    chunksize = min(max_chunksize, N)
    C_buf = similar(Y, N, chunksize)    # buffer for pairwise absolute cosine similarities
    s_buf = Vector{Int}(undef, N)       # buffer for sorting
    chunks = Iterators.partition(1:N, chunksize)
    Z_nzs = @withprogress mapreduce(vcat, enumerate(chunks)) do (chunk_idx, chunk)
        # Compute pairwise absolute cosine similarities for chunk using appropriate buffer
        C_chunk = length(chunk) == chunksize ? C_buf : similar(Y, N, length(chunk))
        mul!(C_chunk, Y', view(Y, :, chunk))
        C_chunk .= abs.(C_chunk)

        # Identify at most `max_nz` largest values to keep for each column `c` in chunk
        q = min(max_nz, N)
        Z_nzs_chunk = map(chunk, eachcol(C_chunk)) do col, c
            # Zero out the self-loop in `c`
            c[col] = zero(eltype(c))

            # Find indices for the `q` largest values in `c`
            inds = partialsortperm!(s_buf, c, 1:q; rev = true)

            # Return corresponding rows, columns and values
            return (;
                rows = copy(inds),
                cols = fill(col, q),
                vals = exp.(-2 .* acos.(min.(view(c, inds), oneunit(eltype(c))))),
            )
        end

        # Update progress bar and return
        @logprogress chunk_idx / cld(N, chunksize)
        return Z_nzs_chunk
    end
    Z_rows = reduce(vcat, getindex.(Z_nzs, :rows))
    Z_cols = reduce(vcat, getindex.(Z_nzs, :cols))
    Z_vals = reduce(vcat, getindex.(Z_nzs, :vals))

    # Form and return affinity matrix corresponding to Z+Z'
    A_rows = [Z_rows; Z_cols]
    A_cols = [Z_cols; Z_rows]
    A_vals = [Z_vals; Z_vals]
    A = sparse(A_rows, A_cols, A_vals, N, N, +)
    return A
end

"""
    tsc_embedding(A, K)

Compute the `K`-dimensional TSC embedding for the `N×N` affinity matrix `A`,
returning a `K×N` matrix of embeddings.
"""
function tsc_embedding(A, K)
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
