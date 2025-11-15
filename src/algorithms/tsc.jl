## Algorithm: TSC

# Result type

""""
    TSCResult{
        TU<:AbstractMatrix{<:AbstractFloat},
        Tc<:AbstractVector{<:Integer},
        T<:Real}

The output of [`tsc`](@ref).

# Fields
- `U::TU`: centers matrix
- `c::Tc`: vector of cluster assignments `c[1],...,c[N]`
- `iterations::Int`: number of iterations performed
- `totalcost::T`: final value of total cost function
- `counts::Vector{Int}`: vector of cluster sizes `counts[1],...,counts[K]`
- `converged::Bool`: final convergence status
"""

struct TSCResult{TU<:AbstractMatrix{<:AbstractFloat},Tc<:AbstractVector{<:Integer},T<:Real}
    U::TU
    c::Tc
    iterations::Int
    totalcost::T
    counts::Vector{Int}
    converged::Bool
end

# Main function

"""
    tsc(X::AbstractMatrix{<:Real}, k::Integer; 
    maxiters::Integer = 100)


Cluster the `N` data points in the `D×N` data matrix `X`
into `K` clusters via the **T**hreshold **S**ubspace **C**lustering (TSC) algorithm.
Output is a [`TSCResult`](@ref) containing the resulting
cluster assignments `c[1],...,c[N]`,
centers matrix `U` of ize `k×k`,
and metadata about the algorithm run.

Threshold subspace clustering (TSC) algorithm treats data points as nodes
in a graph. Three important matrices that make up the TSC algorithm are
1. **Adjacency matrix**: `S` is a `N×N` matrix where `S[i, j]` is the cosine similarity
   between the `i`-th and `j`-th data points.
```math
S[i, j] = \\exp \\left[-2 \\cdot \\arccos \\left( \\frac{X[:, i]^\\top \\cdot X[:, j]}{\\|X[:, i]\\|_2 \\cdot \\|X[:, j]\\|_2} \\right) \\right]
```
2. **Degree matrix**: `D` is a `N×N` diagonal matrix where `D[i, i]` is the sum of the `i`-th row of `S`.
```math
D[i, i] = \\sum_{j=1}^N S[i, j]
```
3. **Laplacian matrix** `L` is a `N×N` Symmetric matrix, it captures the structure of the graph by combining
information from the adjacency and degree matrices.
```math
L_sym = I - D^{-1/2} S D^{-1/2}
```
TSC seeks to cluster data points by performing K-means clustering on the
normalized versions of the `k` smallest eigenvectors of the Laplacian matrix `L_sym`.

# Keyword arguments
- `maxiters::Integer = 100`: maximum number of iterations

See also [`TSCResult`](@ref).
"""

function tsc(X::AbstractMatrix{<:Real}, k::Integer; maxiters::Integer = 100)
    #  Check clusters are more than 1 and not greater than the number of data points
    2 <= k <= size(X, 2) || throw(
        ArgumentError(
            "Number of clusters `k` must be between 2 and the number of columns in `X`, got k = $k.",
        ),
    )

    # Check maxiters
    maxiters >= 0 || throw(
        ArgumentError(
            "Maximum number of iterations must be nonnegative. Got `maxiters=$maxiters`.",
        ),
    )

    # Affinity matrix
    S = tsc_affinity(X; chunksize = 1000)
    V = tsc_embedding(S, k)

    # Compute cluster assignments via k-means
    result = kmeans(permutedims(V), k; maxiter = maxiters)
    U = result.centers
    c = result.assignments
    iterations = result.iterations
    totalcost = result.totalcost
    counts = result.counts
    converged = result.converged

    return TSCResult(U, c, iterations, totalcost, counts, converged)
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
    tsc_embedding(A, k)

Compute the `k`-dimensional TSC embedding for the `N×N` affinity matrix `A`,
returning a `k×N` matrix of embeddings.
"""
function tsc_embedding(A, k)
    # Compute node degrees and form Laplacian matrix `L` from `A`
    D = Diagonal(vec(sum(A; dims = 2)))
    L = Symmetric(I - (inv(sqrt(D)) * A * inv(sqrt(D))))

    # Compute eigenvectors corresponding to `k` smallest eigenvalues
    decomp, history = partialschur(L; nev = k, which = :SR)
    history.converged ||
        @warn "Iterative algorithm for threshold subspace did not converge - results may be inaccurate."

    # Permute and normalize to obtain embeddings
    E = mapslices(normalize, permutedims(decomp.Q); dims = 1)

    # Return the embeddings
    return E
end
