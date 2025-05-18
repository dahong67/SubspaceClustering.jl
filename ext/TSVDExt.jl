module TSVDExt

using SubspaceClustering
using TSVD:tsvd

const base_estimate = SubspaceClustering.kss_estimate_subspace

function SubspaceClustering.kss_estimate_subspace(Xk, dk; method::Symbol = :svd)
    if method == :tsvd
        U, _, _ = tsvd(Xk)
        return U[:, 1:dk]
    else
        return base_estimate(Xk, dk; method=method)
    end
end

end
