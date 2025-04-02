module SubspaceClustering

# Write your package code here.

## K-Subspaces SubspaceClustering

#Imports
import LinearAlgebra: norm, svd, transpose
import Base: copy, deepcopy, findall, view
import Base:argmax
import Random: default_rng, AbstractRNG
using StableRNGs
using Compat
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

#Public function
@compat public randsubspace

"""
	randsubspace(D::Int, d::Vector{Int}; rng::AbstractRNG=default_rng())

	Generate random d-dimensional subspaces. 

# Arguments
- `D::Int`: Dimension of the feature space.
- `d::Vector{Int}`: Dimensions of the subspace.
- `rng::AbstractRNG=default_rng()`: Default global random number generator (RNG) with AbstractRNG type.
"""

function randsubspace(D::Int, d::Vector{Int}; rng::AbstractRNG=default_rng())
	[polar(randn(rng, D, d_i)) for d_i in d]
end

"""
	KSSResult{M<:AbstractMatrix{<:AbstractFloat}, T<:Real}

The output of [`KSS`](@ref).

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

struct KSSResult{M<:AbstractMatrix{<:AbstractFloat}, T<:Real}
	U::Vector{M} # Subspace bases for each cluster
	c::Vector{Int} # Cluster assignments for each data point
	iterations::Int # Number of iterations performed
	totalcost::T # Total cost of the clustering
	counts::Vector{Int} # Number of data points in each cluster
	converged::Bool # Convergence status

end

"""
	KSS(X, d; niters=100, randng=StableRNG(1234), Uinit=polar.(randn.(rng, size(X, 1), collect(d))))

Run K-subspaces on the data matrix `X`
with subspace dimensions `d[1], ..., d[K]`.

# Arguments
- `X::AbstractMatrix`: Data matrix of size `(D, N)`. Each column represents a data point.
- `d::Vector{Int}`: A vector of length `K` containing the subspace dimensions for each of the 'K' subspaces.

# Keyword Arguments
- `niters::Int=100`: Maximum number of iterations (default is 100).
- `randng::StableRNG=StableRNG(1234)`: Random number generator.
- 'Uinit::Vector{AbstractMatrix}': A vector of length 'K' containing initial subspace bases. If not provided, they are initialized randomly using randsubspace function.

# Returns
A `KSSResult` containing:
- 'U::Vector{AbstractMatrix}': A vector of length `K` containing the subspace basis matrices for 'K' clusters.
- 'c::Vector{Int}': A vector of length `N` containing the cluster assignments for each data point.
- 'iterations::Int': Number of iterations performed.
- 'totalcost::Real': Total cost of the clustering.
- 'counts::Vector{Int}': A vector of length `K` containing the number of data points in each cluster.
"""
function KSS(X::AbstractMatrix{<:Real},                                                     #in: data matrix with size (D, N)
			d::Vector{<:Integer};                                                           #in: a vector of subspace dimensions of length K
			niters::Integer = 100,                                                          #in: number of iterations
			randng::StableRNG = StableRNG(1234),                                            #in: a random number generator with StableRNG type
			Uinit::Vector{Matrix{Float64}} = randsubspace(size(X, 1), d; rng=default_rng()) #in: a vector of length K containing initial subspaces
			)                                  
	
	K = length(d)
	D, N = size(X)

	if any(d_i -> d_i > D, d)
		throw(DimensionMismatch("Subspace Dimensions are greater than Feature space Dimensions"))
	end

	# Initialize
	U = deepcopy(Uinit)
	c = [argmax(norm(U[k]' * view(X, :, i)) for k in 1:K) for i in 1:N]
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
				U[k] = polar(randn(randng, D, d[k]))
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

		# Update clusters and calculate total cost
		for i in 1:N
			c[i] = argmax(norm(U[k]' * view(X, :, i)) for k in 1:K)
		end

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
		cost = norm(U[c[i]]' * view(X, :, i))
		totalcost += cost
	end

	return KSSResult(U, c, iter, totalcost, counts, converged)
end

end
