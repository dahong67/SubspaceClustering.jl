module IncSVDExt

using SubspaceClustering
using IncrementalSVD: isvd

const base_estimate = SubspaceClustering.kss_estimate_subspace

function SubspaceClustering.kss_estimate_subspace(Xk, dk; method::Symbol = :svd)
    if method == :isvd
        U, _, _ = isvd(Xk)
        return U[:, 1:dk]
    else
        return base_estimate(Xk, dk; method=method)
    end
end

end