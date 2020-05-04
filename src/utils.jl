module utils

using LinearAlgebra

function KL_MvNormal(mu0, Sigma0, mu1, Sigma1)
    k = length(mu0)
    inv_Sigma1 = inv(Sigma1)
    
    t1 = tr(inv_Sigma1 * Sigma0)
    t2 = (mu1 - mu0)' * inv_Sigma1 * (mu1 - mu0)
    t3 = logdet(Sigma1) - logdet(Sigma0)
    0.5 * (t1 + t2 - k + t3)
end

end