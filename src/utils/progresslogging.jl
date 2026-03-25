## Utils: conditional progress logging

"""
    @withprogressif cond [name=""] [parentid=uuid4()] ex

Conditional version of `@withprogress` that only sets up a progress bar if `cond` is `true`,
in which case it passes the remaining arguments to `@withprogress`. Otherwise, it executes
`ex` directly.

See also [`@logprogressif`](@ref).
"""
macro withprogressif(cond, exprs...)
    ex_withlogging = :(@withprogress $(exprs...))  # version with logging
    ex_withoutlogging = exprs[end]                 # version without logging
    return quote
        if $(esc(cond))
            $(esc(ex_withlogging))
        else
            $(esc(ex_withoutlogging))
        end
    end
end

"""
    @logprogressif cond [name] progress [key1=val1 [key2=val2 ...]]

Conditional version of `@logprogress` that only logs progress if `cond` is `true`,
in which case it passes the remaining arguments to `@logprogress`. Otherwise, it
does nothing.

See also [`@withprogressif`](@ref).
"""
macro logprogressif(cond, exprs...)
    logexpr = :(@logprogress $(exprs...))
    return quote
        if $(esc(cond))
            $(esc(logexpr))
        end
    end
end
