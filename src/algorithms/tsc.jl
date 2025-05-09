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

struct TSCResult{
    TU<:AbstractMatrix{<:AbstractFloat},
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


function tsc(
    X::AbstractMatrix{<:Real}, 
    k::Integer;
    maxiters::Integer = 100,
)
    #  Check clusters are more than 1 and not greater than the number of data points
    2 <= k <= size(X, 2) || throw(
        ArgumentError("Number of clusters `k` must be between 2 and the number of columns in `X`, got k = $k."),
    )

    # Check maxiters
    maxiters >= 0 || throw(
        ArgumentError(
            "Maximum number of iterations must be nonnegative. Got `maxiters=$maxiters`.",
        ),
    )

    # Affinity matrix
    col_norms = [norm(xi) for xi in eachcol(X)]
    Norm_vec = [xi ./ col_norms[i] for (i, xi) in pairs(eachcol(X))]
    Norm_mat = hcat(Norm_vec...)
    A = transpose(Norm_mat) * Norm_mat

    S = exp.((-2 .* acos.(clamp.(A, -1, 1))))

    # Compute node degrees and form Laplacian matrix
    D = Diagonal(vec(sum(S, dims=2)))
    D_sqrinv = sqrt(inv(D))
    L_sym = Symmetric(I - (D_sqrinv * S * D_sqrinv))

    # Compute eigenvectors
    decomp, history = partialschur(L_sym; nev=k, which=:SR)
    history.converged || @warn "Iterative algorithm for threshold subspace did not converge - results may be inaccurate."
    V = mapslices(normalize, decomp.Q; dims=2)

    # Compute cluster assignments via k-means
    result = kmeans(permutedims(V), k; maxiter=maxiters)
    U = result.centers
    c = result.assignments
    iterations = result.iterations
    totalcost = result.totalcost
    counts = result.counts
    converged = result.converged

    return TSCResult(U, c, iterations, totalcost, counts, converged)

end

