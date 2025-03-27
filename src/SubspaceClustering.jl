module SubspaceClustering

# Write your package code here.

## K-Subspaces SubspaceClustering

#Imports
import LinearAlgebra: norm, svd, transpose
import Base: copy, deepcopy, findall, view
import Base:argmax
using StableRNGs
using Logging: @info, @warn
using ProgressLogging: @progress
using ArnoldiMethod: partialschur

#Exports
export polar, KSS

#Polar function

function polar(X)
	U, _, V = svd(X)
	U*V'
end

"""
	KSS(X, d; niters=100, rng=StableRNG(1234), Uinit=polar.(randn.(rng, size(X, 1), collect(d))))

Run K-subspaces on the data matrix `X`
with subspace dimensions `d[1], ..., d[K]`.

# Arguments
- `X::AbstractMatrix`: Data matrix of size `(D, N)`. Each column represents a data point.
- `d::Vector{Int}`: A vector of length `K` containing the subspace dimensions for each of the 'K' subspaces.

# Keyword Arguments
- `niters::Int=100`: Maximum number of iterations (default is 100).
- `rng::StableRNG=StableRNG(1234)`: Random number generator.
- 'Uinit::Vector{AbstractMatrix}': A vector of length 'K' containing initial subspace bases. If not provided, they are initialized randomly.

# Returns
- 'U::Vector{AbstractMatrix}': A vector of length `K` containing the subspace basis matrices for 'K' clusters.
- 'c::Vector{Int}': A vector of length `N` containing the cluster assignments for each data point.
"""
function KSS(X, d; niters=100, rng = StableRNG(1234), Uinit=polar.(randn.(rng, size(X, 1), collect(d))))
	K = length(d)
	D, N = size(X)

	if any(d_i -> d_i > D, d)
		throw(DimensionMismatch("Subspace Dimensions are greater than Feature space Dimensions"))
	end

	# Initialize
	U = deepcopy(Uinit)
	c = [argmax(norm(U[k]' * view(X, :, i)) for k in 1:K) for i in 1:N]
	c_prev = copy(c)

	

	# Iterations
	@progress for t in 1:niters
		# Update subspaces
		for k in 1:K
			ilist = findall(==(k), c)
			@debug "Cluster $k got assigned $(length(ilist)) data points"

			if isempty(ilist)
				@warn "Empty clusters detected at iteration $t - reinitializing the subspace. Consider reducing the number of clusters."
				U[k] = polar(randn(rng, D, d[k]))
			else
				A = view(X, :, ilist) * transpose(view(X, :, ilist))
				decomp, history = partialschur(A; nev=d[k], which=:LR)
				@debug "Cluster $k partialschur decomposition history:
				matrix-vector products: $(history.mvproducts),
				Number of eigenvalues: $(history.nev),
				number of converged eigenvalues: $(history.nconverged),
				converged? = $(history.converged)"
				U[k] = decomp.Q
			end
		end

		# Update clusters
		for i in 1:N
			c[i] = argmax(norm(U[k]' * view(X, :, i)) for k in 1:K)
		end

		# Break if clusters did not change, update otherwise
		if c == c_prev
			@info "Terminated early at iteration $t"
			break
		end
		c_prev .= c
	end

	return U, c
end

end
