abstract type ChainState end;
abstract type SamplerInfo end;
abstract type TransitionInfo end;


function Leapfrog(likeli::Function, theta_tilde::AbstractVector{T}, r_tilde::AbstractVector{T},  eps::T) where T
    grad = similar(theta_tilde)

    gradient!(grad, likeli, theta_tilde) # for type stable
    r_tilde = r_tilde + (eps/2) * grad
    theta_tilde = theta_tilde + eps * r_tilde

    gradient!(grad, likeli, theta_tilde)
    r_tilde = r_tilde + (eps/2) * grad
    return theta_tilde, r_tilde
end

function LeapfrogCache(likeli::Function, theta_tilde::AbstractVector{T}, r_tilde::AbstractVector{T},  
                       eps::T, grad_cached::AbstractVector{T}) where T
    r_tilde = r_tilde + (eps/2) * grad_cached
    theta_tilde = theta_tilde + eps * r_tilde

    grad = similar(theta_tilde)
    grad_cached_new = gradient!(grad, likeli, theta_tilde)
    r_tilde = r_tilde + (eps/2) * grad_cached_new
    return theta_tilde, r_tilde, grad_cached_new
end


function FindReasonableEpsilon(likeli::Function, theta::AbstractVector{T}) where T
    eps = 1.
    size_p = length(theta)
    r = randn(size_p)
    theta_tilde, r_tilde = Leapfrog(theta, r, eps)
    likeli_theta = likeli(theta)
    log_ratio = likeli(theta_tilde) - likeli_theta
    if log_ratio > 0
        while log_ratio > 0
            eps *= 1.1
            theta_tilde, r_tilde = Leapfrog(theta, r, eps)
            log_ratio = likeli(theta_tilde) - likeli_theta
        end
    else
        while log_ratio < 0
            eps /= 1.1
            theta_tilde, r_tilde = Leapfrog(theta, r, eps)
            log_ratio = likeli(theta_tilde) - likeli_theta
        end
    end
    return eps
end

"""
Examples:

```
using YPPL_Inference.Examples.eight_schools_non_centered
using YPPL_Inference: HMCSamplerInfo, HMCState, sampling

likeli = eight_schools_non_centered.likeli
theta0 = ones(eight_schools_non_centered.size_p)
M = 2000
eps = 1e-1
L = 15

sampler_info = HMCSamplerInfo(likeli, eps, size_p, L)
state = HMCState(theta0, likeli(theta0))

sampling(state, sampler_info, M)
```
"""
function sampling(state::ChainStateT, sampler_info::SamplerInfoT, M::Int
    ) where {ChainStateT<:ChainState, SamplerInfoT<:SamplerInfo}

    state_list = Vector{ChainStateT}(undef, M+1)
    transition_list = Vector{transition_type(SamplerInfoT)}(undef, M)
    state_list[1] = state

    for i in 1:M
        state, transition_info = NextSample(sampler_info, state)
        state_list[i+1] = state
        transition_list[i] = transition_info
    end

    return state_list, transition_list
end

function sampling(state::ChainStateT, sampler_info::SamplerInfoT, M::Int, chains::Int
        ) where {ChainStateT<:ChainState, SamplerInfoT<:SamplerInfo}
    state_list_list = Vector{Vector{ChainStateT}}(undef, chains)
    transition_list_list = Vector{Vector{transition_type(SamplerInfoT)}}(undef, chains)

    for i in 1:chains
        state_list_list[i], transition_list_list[i] = sampling(state, sampler_info, M)
    end

    return state_list_list, transition_list_list
end

function extract(state_list_list::Vector{Vector{S}}) where S <: ChainState
    chains = length(state_list_list)
    M = length(state_list_list[1])
    size_p = length(state_list_list[1][1].theta)

    posterior = Array{Float64, 3}(undef, chains, M - (M - M÷2), size_p)

    for (i, state_list) in enumerate(state_list_list)
        for (j, state) in enumerate(state_list[end - M÷2 + 1: end])
            posterior[i, j, :] = state.theta
        end
    end

    return posterior
end