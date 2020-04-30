
abstract type AbstractHMCState <: ChainState end
abstract type AbstractHMCSamplerInfo <: SamplerInfo end

struct HMCState{TV <: AbstractVector, T} <: AbstractHMCState 
    theta::TV
    # energy::T # L(theta) - 0.5 * r' * r
    prob::T # L(theta)
end

HMCState(likeli::Function, theta::Vector) = HMCState(theta, likeli(theta))

Base.copy(s::HMCState) = HMCState(s.theta, s.prob)

struct HMCTransitionInfo{T} <: TransitionInfo
    log_alpha::T # log accept probability
end

struct HMCSamplerInfo{F <: Function, T} <: AbstractHMCSamplerInfo
    likeli::F # log joint probability, up to a constant
    eps::T # step size
    size_p::Int # length(theta)
    L::Int # simulation length
end

"""
setup series functions try to setup a reasonable initial value. 
"""
function setup(likeli::Function, size_p::Int, ::Type{HMCSamplerInfo}, ::Type{HMCState})
    theta0 = draw_init(size_p)
    hmc_state = HMCState(likeli, theta0)
    eps = FindReasonableEpsilon(likeli, theta0)
    println("selected eps: $eps")
    L = 15
    hmc_sampler_info = HMCSamplerInfo(likeli, eps, size_p, L)
    return hmc_sampler_info, hmc_state
end

transition_type(::Type{HMCSamplerInfo{F, T}}) where {F, T} = HMCTransitionInfo{T}

function NextSample2(info::HMCSamplerInfo, prev::HMCState)
    likeli = info.likeli

    r0 = r_tilde = randn(info.size_p)
    theta_tilde = prev.theta

    theta_tilde, r_tilde = Leapfrog(likeli, theta_tilde, r_tilde, info.eps, info.L)

    prob = likeli(theta_tilde)
    energy = prob - 0.5 * r_tilde' * r_tilde
    log_alpha = energy - (prev.prob - 0.5 * r0' * r0)

    if log(rand()) < log_alpha
        next_state = HMCState(theta_tilde, prob)
    else
        next_state = copy(prev)
    end
    transition_info = HMCTransitionInfo(log_alpha)

    return next_state, transition_info
end

struct HMCStateCached{TV <: AbstractVector, T} <: AbstractHMCState 
    theta::TV
    prob::T # L(theta)
    grad::TV
end

function HMCStateCached(likeli::Function, theta::Vector)
    grad = similar(theta)
    gradient!(grad, likeli, theta)
    HMCStateCached(theta, likeli(theta), grad)
end

Base.copy(s::HMCStateCached) = HMCStateCached(s.theta, s.prob, s.grad)

function NextSample2(info::HMCSamplerInfo, prev::HMCStateCached)
    likeli = info.likeli

    r0 = r_tilde = randn(info.size_p)
    theta_tilde = prev.theta

    theta_tilde, r_tilde, grad_cached = LeapfrogCache(likeli, theta_tilde, r_tilde, info.eps, info.L, prev.grad)

    prob = likeli(theta_tilde)
    energy = prob - 0.5 * r_tilde' * r_tilde
    log_alpha = min(0.0, energy - (prev.prob - 0.5 * r0' * r0))

    if log(rand()) < log_alpha
        next_state = HMCStateCached(theta_tilde, prob, grad_cached)
    else
        next_state = copy(prev)
    end
    transition_info = HMCTransitionInfo(log_alpha)

    return next_state, transition_info
end

function LeapfrogIteration(info::HMCSamplerInfo, prev::HMCState, r_tilde::Vector)
    Leapfrog(info.likeli, prev.theta, r_tilde, info.eps, info.L)
end
function LeapfrogIteration(info::HMCSamplerInfo, prev::HMCStateCached, r_tilde::Vector)
    LeapfrogCache(info.likeli, prev.theta, r_tilde, info.eps, info.L, prev.grad)
end

accept_update(::SamplerInfo, ::HMCState, 
               theta_tilde, prob, grad_cached, log_alpha) = HMCState(theta_tilde, prob)
accept_update(::SamplerInfo, ::HMCStateCached, 
              theta_tilde, prob, grad_cached, log_alhpha) = HMCStateCached(theta_tilde, prob, grad_cached)

reject_update(::SamplerInfo, prev::AbstractHMCState,
              theta_tilde, prob, grad_cached, log_alhpha) = copy(prev)

function NextSample(info::AbstractHMCSamplerInfo, prev::S) where S <: AbstractHMCState
    likeli = info.likeli

    r0 = r_tilde = randn(info.size_p)

    theta_tilde, r_tilde, grad_cached = LeapfrogIteration(info, prev, r_tilde)

    prob = likeli(theta_tilde)
    energy = prob - 0.5 * r_tilde' * r_tilde
    log_alpha = min(0., energy - (prev.prob - 0.5 * r0' * r0))

    if log(rand()) < log_alpha
        next_state = accept_update(info, prev, theta_tilde, prob, grad_cached, log_alpha)
    else
        next_state = reject_update(info, prev, theta_tilde, prob, grad_cached, log_alpha)
    end
    transition_info = HMCTransitionInfo(log_alpha)

    return next_state, transition_info
end


