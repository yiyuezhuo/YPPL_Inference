module Laplace

using Optim
using ForwardDiff: hessian!
using LinearAlgebra

function laplace_approximate(likeli::Function, theta0::Vector)
    f = (p) -> (-likeli(p))
    res = optimize(f, theta0)
    theta_fit = Optim.minimizer(res)
    hess = similar(theta0, length(theta0), length(theta0))
    hessian!(hess, likeli, theta_fit)
    Sigma = -inv(hess)
    return theta_fit, Symmetric(Sigma)
end

end