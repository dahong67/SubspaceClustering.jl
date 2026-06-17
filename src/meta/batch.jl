## Batch KSS and KAS Algorithms

"""
    batch(alg, X, d)
"""
function batch(
    alg::Function,
    X::AbstractMatrix{<:Number},
    d::AbstractVector{<:Integer};
    nruns::Integer = 50,
    maxiters::Integer = 100,
    showprogress::Bool = false,
)
    # check nruns
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

    return first(sort(runs; by = run -> run.totalcost))
end
