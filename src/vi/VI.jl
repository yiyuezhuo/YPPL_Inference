module VI

using ForwardDiff
using LinearAlgebra

function build_elbo_meanfield(likeli::Function, size_p::Int)
    function (p::Vector{T}) where T
        mu = p[1:size_p]
        log_sigma = p[size_p+1:end]
        sigma = exp.(log_sigma)
        noise = randn(T, size_p)
        z = mu .+ sigma  .* noise
        logp = likeli(z)
        #logq = (-0.5 * ((z .- mu) ./ sigma) .^ 2) |> sum
        logq = sum(-log_sigma) # other terms are useless for gradient
        logp - logq
    end
end

function build_elbo_rank1(likeli::Function, size_p::Int)
    function(p::Vector{T}) where T
        mu = p[1:size_p]
        log_sigma = p[size_p+1:2*size_p]
        sigma = exp.(log_sigma)
        alpha = p[2*size_p+1:end]
        
        noise = randn(T, size_p)
        noise_alpha = randn(T)
        z = mu + sigma .* noise + alpha * noise_alpha
        
        logp = likeli(z)
        #logq = (-0.5 * ((z .- mu) ./ sigma) .^ 2) |> sum
        #logq = sum(-log_sigma) # other terms are useless for gradient
        logq = -0.5 * (log(1 + (alpha.^2)' * (1 ./ sigma .^ 2)) + 2 * sum(log_sigma))
        logp - logq
    end
end

function build_elbo_lowrank(likeli::Function, size_p::Int, vi_rank::Int)
    function(p::Vector{T}) where T
        mu = p[1:size_p]
        log_sigma = p[size_p+1:2*size_p]
        sigma = exp.(log_sigma)
        U = reshape(p[2*size_p+1:end], size_p, vi_rank)

        noise = randn(T, size_p)
        noise_U = randn(T, vi_rank)
        z = mu + sigma .* noise + U * noise_U

        logp = likeli(z)
        logq = -0.5 * (logdet(I + U' * ((1. ./ sigma .^ 2) .* U)) + 2 * sum(log_sigma))
        logp - logq
    end
end


# Algorithm trait, maybe some attributions will be added lately, but for now they're pure traits
abstract type VariationalDistribution end;
struct MeanField <: VariationalDistribution end
struct Rank1 <: VariationalDistribution end
struct LowRank <: VariationalDistribution
    rank::Int
end
struct FullRank <: VariationalDistribution end

init_vp(T, ::MeanField, size_p::Int) = zeros(T, size_p*2)
init_vp(T, ::Rank1, size_p::Int) = zeros(T, size_p*3)
init_vp(T, lr::LowRank, size_p::Int) = zeros(T, size_p*(2+lr.rank))

build_elbo(likeli::Function, size_p::Int, ::MeanField) = build_elbo_meanfield(likeli, size_p)
build_elbo(likeli::Function, size_p::Int, ::Rank1) = build_elbo_rank1(likeli, size_p)
build_elbo(likeli::Function, size_p::Int, lr::LowRank) = build_elbo_lowrank(likeli, size_p, lr.rank)


abstract type OptimizingMethod{T} end

struct SGD{T <: Real} <: OptimizingMethod{T}
    L::Int
    eps::T
end

struct Adam{T <: Real} <: OptimizingMethod{T}
    L::Int
    eps::T
    eps_smooth::T
    beta1::T
    beta2::T
end
Adam(L::Int, eps::T) where T = Adam(L, eps, T(1e-6), T(0.9), T(0.9))


function vi(likeli::Function, size_p::Int, vd::VariationalDistribution, om::OptimizingMethod{T}) where T
    elbo = build_elbo(likeli, size_p, vd)
    vp = init_vp(T, vd, size_p) # variational parameters

    optim(om, elbo, vp)
end

function optim(om::SGD{T}, elbo::Function, vp::Vector{T}) where T
    eps = om.eps
    L = om.L
    
    elbo_list = Vector{T}(undef, L)
    vp_list = Vector{Vector{T}}(undef, L)
    
    for i in 1:L
        grad = similar(vp)
        ForwardDiff.gradient!(grad, elbo, vp)
        vp += eps * grad
        
        elbo_list[i] = elbo(vp)
        vp_list[i] = vp
    end
    vp_list, elbo_list
end

function optim(om::Adam{T}, elbo::Function, vp::Vector{T}) where T
    L = om.L
    beta1 = om.beta1
    beta2 = om.beta2
    eps = om.eps
    eps_smooth = om.eps_smooth

    elbo_list = Vector{T}(undef, L)
    vp_list = Vector{Vector{T}}(undef, L)

    m = zeros(T, size(vp)...)
    v = ones(T, size(vp)...)   

    for i in 1:L
        grad = similar(vp)
        ForwardDiff.gradient!(grad, elbo, vp)
        
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad .^ 2
        
        m_hat = m / (1 - beta1 ^ i)
        v_hat = v / (1 - beta2 ^ i)
        
        vp = vp .+ eps * m_hat ./ (sqrt.(v_hat) .+ eps_smooth)
        
        elbo_list[i] = elbo(vp)
        vp_list[i] = vp
    end

    return vp_list, elbo_list
end

end
