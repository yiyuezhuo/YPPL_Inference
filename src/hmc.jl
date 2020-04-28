
struct HMCState{TV <: AbstractVector, T} <: ChainState 
    theta::TV
    # energy::T # L(theta) - 0.5 * r' * r
    prob::T # L(theta)
end

Base.copy(s::HMCState) = HMCState(s.theta, s.prob)

struct HMCTransitionInfo{T} <: TransitionInfo
    log_alpha::T # log accept probability
end

struct HMCSamplerInfo{F <: Function, T} <: SamplerInfo
    likeli::F # log joint probability, up to a constant
    eps::T # step size
    size_p::Int # length(theta)
    L::Int # simulation length
end

transition_type(::Type{HMCSamplerInfo{F, T}}) where {F, T} = HMCTransitionInfo{T}

function NextSample(info::HMCSamplerInfo, prev::HMCState)
    likeli = info.likeli

    r0 = r_tilde = randn(info.size_p)
    theta_tilde = prev.theta

    grad = similar(theta_tilde)
    grad_cached = gradient!(grad, likeli, theta_tilde)
    for i in 1:info.L
        theta_tilde, r_tilde, grad_cached = LeapfrogCache(likeli, theta_tilde, r_tilde, info.eps, grad_cached)
    end
    #=
    for i in 1:info.L
        theta_tilde, r_tilde = Leapfrog(likeli, theta_tilde, r_tilde, info.eps)
    end
    =#

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

struct HMCStateCached{TV <: AbstractVector, T} <: ChainState 
    theta::TV
    prob::T # L(theta)
    grad::TV
end

Base.copy(s::HMCStateCached) = HMCStateCached(s.theta, s.prob, s.grad)

function NextSample(info::HMCSamplerInfo, prev::HMCStateCached)
    likeli = info.likeli

    r0 = r_tilde = randn(info.size_p)
    theta_tilde = prev.theta
    grad_cached = prev.grad
    for i in 1:info.L
        theta_tilde, r_tilde, grad_cached = LeapfrogCache(likeli, theta_tilde, r_tilde, info.eps, grad_cached)
    end
    prob = likeli(theta_tilde)
    energy = prob - 0.5 * r_tilde' * r_tilde
    log_alpha = energy - (prev.prob - 0.5 * r0' * r0)

    if log(rand()) < log_alpha
        next_state = HMCStateCached(theta_tilde, prob, grad_cached)
    else
        next_state = copy(prev)
    end
    transition_info = HMCTransitionInfo(log_alpha)

    return next_state, transition_info
end