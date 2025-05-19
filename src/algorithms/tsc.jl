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
    S = affinity(X; chunksize = 1000)

    # Compute node degrees and form Laplacian matrix
    D = Diagonal(vec(sum(S; dims = 2)))
    D_sqrinv = sqrt(inv(D))
    L_sym = Symmetric(I - (D_sqrinv * S * D_sqrinv))

    # Compute eigenvectors
    decomp, history = partialschur(L_sym; nev = k, which = :SR)
    history.converged ||
        @warn "Iterative algorithm for threshold subspace did not converge - results may be inaccurate."
    V = mapslices(normalize, decomp.Q; dims = 2)

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

"""
    affinity(X::AbstractMatrix; max_nz::Integer=30, chunksize::Integer=isqrt(size(X,2)))

Compute a **sparse**, **symmetric** cosine-angle affinity matrix for the data points in `D×N` data matrix `X`
and return a `N×N` size sparsematrix of type `SparseMatrixCSC{Float64, Int64}`  with `max_nz` values per row/column .

# Keyword arguments
- `max_nz::Integer = 30`: maximum number of nonzero values per row/column
- `chunksize::Integer = isqrt(size(X,2))`: Number of columnns to process at a time when computing pairwise cosine. 
"""
function affinity(
    X::AbstractMatrix{<:Real};
    max_nz::Integer = 30,
    chunksize::Integer = isqrt(size(X, 2)),
)
    func = c -> exp(-2 * acos(clamp(c, -1, 1)))

    # Compute normalized spectra (so that inner product = cosine of angle)
    X = mapslices(normalize, X; dims = 1)

    # Find nonzero values (in chunks)
    C_buf = similar(X, size(X, 2), chunksize)    # pairwise cosine buffer
    s_buf = Vector{Int}(undef, size(X, 2))       # sorting buffer
    nz_list = @withprogress mapreduce(
        vcat,
        enumerate(Iterators.partition(1:size(X, 2), chunksize)),
    ) do (chunk_idx, chunk)

        # Compute cosine angles (for chunk) and store in appropriate buffer
        C_chunk = length(chunk) == chunksize ? C_buf : similar(X, size(X, 2), length(chunk))
        mul!(C_chunk, X', view(X, :, chunk))

        # Zero out all but `max_nz` largest values
        nzs = map(chunk, eachcol(C_chunk)) do col, c
            idx = partialsortperm!(s_buf, c, 1:max_nz; rev = true)
            return collect(idx), fill(col, max_nz), func.(view(c, idx))
        end

        # Log progress and return
        @logprogress chunk_idx / cld(size(X, 2), chunksize)
        return nzs
    end

    # Form and return sparse array
    rows = reduce(vcat, getindex.(nz_list, 1))
    cols = reduce(vcat, getindex.(nz_list, 2))
    vals = reduce(vcat, getindex.(nz_list, 3))
    return sparse([rows; cols], [cols; rows], [vals; vals])
end