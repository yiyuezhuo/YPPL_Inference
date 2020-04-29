abstract type ChainState end;
abstract type SamplerInfo end;
abstract type TransitionInfo end;

"""
Use to select implementation for performance benchmark
"""
abstract type Impl end

struct UseNextSample <: Impl end
struct UseNextSample2 <: Impl end


function Leapfrog(likeli::Function, theta_tilde::AbstractVector{T}, 
                  r_tilde::AbstractVector{T}, eps::T
                  ) where T
    grad = similar(theta_tilde)

    gradient!(grad, likeli, theta_tilde) # for type stable
    r_tilde = r_tilde + (eps/2) * grad
    theta_tilde = theta_tilde + eps * r_tilde

    gradient!(grad, likeli, theta_tilde)
    r_tilde = r_tilde + (eps/2) * grad
    return theta_tilde, r_tilde
end              

function LeapfrogCache(likeli::Function, theta_tilde::AbstractVector{T}, 
                       r_tilde::AbstractVector{T}, eps::T, 
                       grad_cached::AbstractVector{T}) where T
    r_tilde = r_tilde + (eps/2) * grad_cached
    theta_tilde = theta_tilde + eps * r_tilde

    grad = similar(theta_tilde)
    grad_cached_new = gradient!(grad, likeli, theta_tilde)
    r_tilde = r_tilde + (eps/2) * grad_cached_new
    return theta_tilde, r_tilde, grad_cached_new
end

function Leapfrog(likeli::Function, theta_tilde::AbstractVector{T}, 
                  r_tilde::AbstractVector{T}, eps::T,
                  L::Int) where T
    grad = similar(theta_tilde)
    grad_cached = gradient!(grad, likeli, theta_tilde)
    for i in 1:L
        theta_tilde, r_tilde, grad_cached = LeapfrogCache(likeli, theta_tilde, r_tilde, eps, grad_cached)
    end
    return theta_tilde, r_tilde, grad_cached
end

function LeapfrogCache(likeli::Function, theta_tilde::AbstractVector{T},
                       r_tilde::AbstractVector{T}, eps::T, 
                       L::Int, grad_cached::AbstractVector{T}) where T
    for i in 1:L
        theta_tilde, r_tilde, grad_cached = LeapfrogCache(likeli, theta_tilde, r_tilde, eps, grad_cached)
    end
    return theta_tilde, r_tilde, grad_cached
end

function FindReasonableEpsilon(likeli::Function, theta::AbstractVector{T}) where T
    eps = 1.
    size_p = length(theta)
    r = randn(size_p)
    theta_tilde, r_tilde = Leapfrog(likeli, theta, r, eps)

    p0 = exp(likeli(theta) - r' * r)
    p_tilde = exp(likeli(theta_tilde) - r_tilde' * r_tilde)
    
    if p_tilde / p0 > 0.5
        while p_tilde / p0 > 0.5
            eps *= 1.1
            theta_tilde, r_tilde = Leapfrog(likeli, theta, r, eps)
            p_tilde = exp(likeli(theta_tilde) - r_tilde' * r_tilde)
        end
    else
        while p_tilde / p0 < 0.5
            eps /= 1.1
            theta_tilde, r_tilde = Leapfrog(likeli, theta, r, eps)
            p_tilde = exp(likeli(theta_tilde) - r_tilde' * r_tilde)
        end
    end
    return eps
end

#=

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
=#

# default implementation use NextSample2 now.
function sampling(sampler_info::SamplerInfoT, state::ChainStateT, M::Int
        ) where {ChainStateT<:ChainState, SamplerInfoT<:SamplerInfo}
    sampling(UseNextSample(), sampler_info, state, M)
end

function sampling(sampler_info::SamplerInfoT, state::ChainStateT, M::Int, chains::Int
    ) where {ChainStateT<:ChainState, SamplerInfoT<:SamplerInfo}
    sampling(UseNextSample(), sampler_info, state,  M, chains)
end

# use impl to select which NextSample version to use 

function sampling(impl::Impl, sampler_info::SamplerInfoT, state::ChainStateT, M::Int
    ) where {ChainStateT<:ChainState, SamplerInfoT<:SamplerInfo}
    state_list = Vector{ChainStateT}(undef, M+1)
    transition_list = Vector{transition_type(SamplerInfoT)}(undef, M)
    state_list[1] = state

    for i in 1:M
        state, transition_info = NextSample(impl, sampler_info, state)
        state_list[i+1] = state
        transition_list[i] = transition_info
    end

    return state_list, transition_list
end

function sampling(impl::Impl, sampler_info::SamplerInfoT, state::ChainStateT, M::Int, chains::Int
        ) where {ChainStateT<:ChainState, SamplerInfoT<:SamplerInfo}
    state_list_list = Vector{Vector{ChainStateT}}(undef, chains)
    transition_list_list = Vector{Vector{transition_type(SamplerInfoT)}}(undef, chains)

    for i in 1:chains
        state_list_list[i], transition_list_list[i] = sampling(impl, sampler_info, state, M)
    end

    return state_list_list, transition_list_list
end


# helper

function zip2(state_transition_list)
    #state_list_list = [state_transition[1] for state_transition in state_transition_list]
    #transition_list_list = [state_transition[2] for state_transition in state_transition_list]
    t = [[state_transition[i] for state_transition in state_transition_list] for i in 1:2]
    t[1], t[2]
end

function zip4(state_transition_list)
    t = [[state_transition[i] for state_transition in state_transition_list] for i in 1:4]
    t[1], t[2], t[3], t[4]
end


function sampling(likeli::Function, size_p::Int, sampler_info_type, state_type, M::Int)
    sampler_info, state = setup(likeli, size_p, sampler_info_type, state_type)
    sampling(sampler_info, state, M)
end

function sampling(likeli::Function, size_p::Int, sampler_info_type, state_type, M::Int, chains::Int)
    [sampling(likeli, size_p, sampler_info_type, state_type, M) for i in 1:chains] |> zip2
end

function sampling(likeli::Function, size_p::Int, sampler_info_type, state_type)
    sampling(likeli, size_p, sampler_info_type, state_type, 1000, 4)
end

# split sampling, used to implement dual average scheme


function sampling_split(sampler_info::SamplerInfoT, state::ChainStateT, M1::Int,
                        sampler_info_constructor::Type, state_constructor::Type, M2::Int
                       ) where {ChainStateT<:ChainState, SamplerInfoT<:SamplerInfo}
    state_list, transition_list = sampling(sampler_info, state, M1)
    sampler_info2 = sampler_info_constructor(sampler_info, state_list[end])
    state2 = state_constructor(sampler_info, state_list[end])
    state_list2, transition_list2 = sampling(sampler_info2, state2, M2)
    return state_list, transition_list, state_list2, transition_list2
end

function sampling_split(sampler_info::SamplerInfoT, state::ChainStateT, M1::Int,
                        sampler_info_constructor::Type, state_constructor::Type, M2::Int,
                        chains::Int) where {ChainStateT<:ChainState, SamplerInfoT<:SamplerInfo}
    [sampling_split(sampler_info, state, M1, sampler_info_constructor, state_constructor, M2) for i in 1:chains] |> zip4
end


# utils functions

draw_init(size_p::Int) = rand(size_p)*4 .- 2.

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

function extract_full(state_list_list::Vector{Vector{S}}) where S <: ChainState
    chains = length(state_list_list)
    M = length(state_list_list[1])
    size_p = length(state_list_list[1][1].theta)

    posterior = Array{Float64, 3}(undef, chains, M, size_p)

    for (i, state_list) in enumerate(state_list_list)
        for (j, state) in enumerate(state_list)
            posterior[i, j, :] = state.theta
        end
    end

    return posterior
end