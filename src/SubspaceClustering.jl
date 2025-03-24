module SubspaceClustering

# Write your package code here.

## K-Subspaces SubspaceClustering

#Imports
import LinearAlgebra: norm, svd, transpose
import Base: copy, deepcopy, findall, view
import Base:argmax
using StableRNGs: randn
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
	KSS(X, d; niters=100, Uinit=nothing)

Run K-subspaces on the data matrix `X`
with subspace dimensions `d[1], ..., d[K]`.

# Arguments
- `X::AbstractMatrix`: Data matrix of size `(D, N)`. Each column represents a data point.
- `d::Vector{Int}`: A vector of length `K` containing the subspace dimensions for each of the 'K' subspaces.

# Keyword Arguments
- `niters::Int=100`: Maximum number of iterations (default is 100).
-'Uinit::Union{Nothing, Vector{AbstractMatrix}}': Initial subspaces for the algorithm. If set to nothing, random subspaces are initialized.

# Returns
- 'U::Vector{AbstractMatrix}': A vector of length `K` containing the subspace basis matrices for 'K' clusters.
- 'c::Vector{Int}': A vector of length `N` containing the cluster assignments for each data point.
"""
function KSS(X, d; niters=100, Uinit=polar.(randn.(size(X, 1), collect(d))))
	K = length(d)
	D, N = size(X)

	# Initialize
	U = deepcopy(Uinit)
	c = [argmax(norm(U[k]' * view(X, :, i)) for k in 1:K) for i in 1:N]
	c_prev = copy(c)

	if any(d_i -> d_i > D, d)
		throw(DimensionMismatch("Subspace Dimensions are greater than Feature space Dimensions"))
	end

	# Iterations
	@progress for t in 1:niters
		# Update subspaces
		for k in 1:K
			ilist = findall(==(k), c)
			@debug "Cluster $k, ilist size: $(length(ilist))"

			if isempty(ilist)
				@warn "Cluster $k is empty; re-initializing subspace."
				U[k] = polar(randn(D, d[k]))
			else
				A = view(X, :, ilist) * transpose(view(X, :, ilist))
				decomp, history = partialschur(A; nev=d[k], which=:LR)
				@debug history
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
