"""
Warmup, thus dual average step
"""

struct HMCWarmupState{TV <: AbstractVector, T} <: AbstractHMCState
    theta::TV
    prob::T
    eps::T
    eps_bar::T
    H_bar::T
    m::Int
end

function HMCWarmupState(likeli::Function, theta::Vector{T}) where T
    prob = likeli(theta)
    eps = FindReasonableEpsilon(likeli, theta)
    eps_bar = T(1)
    H_bar = T(0)
    m = 2 # "next" m
    HMCWarmupState(theta, prob, eps, eps_bar, H_bar, m)
end

# Base.copy # We don't need to define copy for this type

struct HMCWarmupSamplerInfo{F <: Function, T} <: AbstractHMCSamplerInfo
    likeli::F # log joint probability, up to a constant
    lambda::T # "invariate" step size
    size_p::Int # length(theta)
    delta::T # expected accept probability
    mu::T # log(10*eps0)

    gamma::T
    t0::T
    kappa::T
end

function HMCWarmupSamplerInfo(likeli::Function, lambda::T, size_p::Int, delta::T, mu::T) where T
    HMCWarmupSamplerInfo(likeli, lambda, size_p, delta, mu, 
                         T(0.05), T(10), T(0.75))
end

function setup(likeli::Function, size_p::Int, ::Type{HMCWarmupSamplerInfo}, ::Type{HMCWarmupState})
    theta0 = draw_init(size_p)
    hmc_warmup_state = HMCWarmupState(likeli, theta0)
    lambda = 15.
    delta = 0.65
    mu = log(10 * hmc_warmup_state.eps)
    hmc_warmup_sampler_info = HMCWarmupSamplerInfo(likeli, lambda, size_p, delta, mu)
    return hmc_warmup_sampler_info, hmc_warmup_state
end


transition_type(::Type{HMCWarmupSamplerInfo{F, T}}) where {F, T} = HMCTransitionInfo{T}

get_L(info::HMCWarmupSamplerInfo, prev::HMCWarmupState) = Int(max(1, round(info.lambda / prev.eps)))

function LeapfrogIteration(info::HMCWarmupSamplerInfo, prev::HMCWarmupState, r_tilde::Vector)
    L = get_L(info, prev)
    Leapfrog(info.likeli, prev.theta, r_tilde, prev.eps, L)
end

function general_update(info::HMCWarmupSamplerInfo, prev::HMCWarmupState,
                        theta_tilde, prob, grad_cached, log_alpha)
    H_bar = (1. - 1. /(prev.m + info.t0)) * prev.H_bar + (1. / (prev.m + info.t0)) * (info.delta - exp(log_alpha))
    log_eps = info.mu - sqrt(prev.m) / info.gamma * H_bar
    w = prev.m ^ (-info.kappa)
    log_eps_bar = w * log_eps + (1 - w) * log(prev.eps_bar)
    return H_bar, log_eps, log_eps_bar                 
end

function accept_update(info::HMCWarmupSamplerInfo, prev::HMCWarmupState, 
                       theta_tilde, prob, grad_cached, log_alpha)
    H_bar, log_eps, log_eps_bar = general_update(info, prev, theta_tilde, prob, grad_cached, log_alpha)
    HMCWarmupState(theta_tilde, prob, exp(log_eps), exp(log_eps_bar), H_bar, prev.m + 1)
end

function reject_update(info::HMCWarmupSamplerInfo, prev::HMCWarmupState,
                       theta_tilde, prob, grad_cached, log_alpha)
    H_bar, log_eps, log_eps_bar = general_update(info, prev, theta_tilde, prob, grad_cached, log_alpha)
    HMCWarmupState(prev.theta, prev.prob, exp(log_eps), exp(log_eps_bar), H_bar, prev.m + 1)
end

# phase shift

function HMCState(info::HMCWarmupSamplerInfo, prev::HMCWarmupState)
    HMCState(prev.theta, prev.prob)
end

function HMCSamplerInfo(info::HMCWarmupSamplerInfo, prev::HMCWarmupState)
    HMCSamplerInfo(info.likeli, prev.eps_bar, info.size_p, get_L(info, prev))
end

