## Batch KSS and KAS Algorithms

"""
    batch(alg::Union{typeof(kss), typeof(kas)}, X::AbstractMatrix{<:Number},d::AbstractVector{<:Integer};
    nruns::Integer = 50,
    maxiters::Integer = 100,
    showprogress::Bool = false)

Both K-Subpaces (KSS) and K-Affinespaces (KAS) may get stuck in poor local minima depending on the initialization. Run `kss` and `kas` algorithm multiple times using different random initializations and return the run with the lowest cost.

# Arguments
- `alg::Union{typeof(kss), typeof(kas)}`: Clustering algorithm to run
- `X::AbstractMatrix{<:Number}`: Data matrix whose columns are observations
- `d::AbstractVector{<:Integer}`: Dimensions of the subspaces/affine spaces

# Keyword arguments
- `nruns::Integer = 50`: Number of independent runs
- `maxiters::Integer = 100`: Maximum number of iterations per run
- `showprogress::Bool = false`: whether to log progress for different algorithm runs

See also [`kss`](@ref), [`kas`](@ref).
"""
function batch(
    alg::Union{typeof(kss), typeof(kas)},
    X::AbstractMatrix{<:Number},
    d::AbstractVector{<:Integer};
    nruns::Integer = 50,
    maxiters::Integer = 100,
    showprogress::Bool = false,
)
    # check number of runs
    nruns > 0 || throw(ArgumentError("nruns must be positive. Got `nruns=$nruns`"))

    runs = @withprogressif showprogress map(1:nruns) do idx
        rng = MersenneTwister(idx)
        result = alg(X, d; rng=rng, maxiters=maxiters)
        @logprogressif showprogress idx/nruns
        return result
    end
    
    # Info on number of converged runs
    nconverged = count(run -> run.converged, runs)
    @info "$nconverged/$nruns runs converged"

    return runs[argmin([run.totalcost for run in runs])]
end
