abstract type AbstractNUTSState <: ChainState end
abstract type AbstractNUTSSamplerInfo <: SamplerInfo end

struct NUTSSamplerInfo{F <: Function, T} <: AbstractNUTSSamplerInfo
    likeli::F
    eps::T
    size_p::Int

    max_tree_depth::Int
    delta_max::T
end

function NUTSSamplerInfo(likeli::Function, eps::T, size_p::Int) where T
    NUTSSamplerInfo(likeli, eps, size_p, 15, T(1000))
end

struct NUTSState{TV <: AbstractVector, T} <: AbstractNUTSState
    theta::TV
    prob::T
end

NUTSState(likeli::Function, theta::Vector) = NUTSState(theta, likeli(theta))

struct NUTSTransitionInfo{T} <: TransitionInfo
    log_u::T # slice value
    tree_depth::Int # tree depth
    n::Int # considered size
end

transition_type(::Type{NUTSSamplerInfo{F, T}}) where {F, T} = NUTSTransitionInfo{T}

#=
# altnative slow implementation
function NextSample2(info::NUTSSamplerInfo, prev::NUTSState)
    r0 = r_neg = r_pos = randn(info.size_p)
    energy = prev.prob - 0.5 * r0' * r0
    log_u = log(rand()) + energy # u ~ U([0, exp(L(theta) - 0.5 r0' * r0)])

    theta_m = theta_neg = theta_pos = prev.theta
    j = 0 # cureent tree depth
    n = 1 # considered size
    s = true # continue ?
    while s
        if j > info.max_tree_depth
            println("Reach max tree depth, increase eps or max_tree_depth to fix")
            break
        end
        v = round(rand())*2-1 # -1. or 1.
        if v == -1.
            theta_neg, r_neg, _, _, theta_p, n_p, s_p = BuildTree2(info, theta_neg, r_neg, log_u, v, j)
        else
            _, _, theta_pos, r_pos, theta_p, n_p, s_p = BuildTree2(info, theta_pos, r_pos, log_u, v, j)
        end
        if s_p
            if rand() < n_p / n
                theta_m = theta_p
            end
        end
        n = n + n_p
        theta_diff = theta_pos .- theta_neg
        s = s_p & (theta_diff' * r_neg >= 0) & (theta_diff' * r_pos >= 0)
        j = j + 1
    end
    return NUTSState(theta_m, info.likeli(theta_m)), NUTSTransitionInfo(log_u, j, n)
end

function BuildTree2(info::NUTSSamplerInfo, theta::Vector, r::Vector, log_u::T, v::T, j::Int) where T
    if j == 0
        # Base case - take one leapfrog step in the direction v
        theta_p, r_p = Leapfrog(info.likeli, theta, r, v * info.eps)
        energy = info.likeli(theta_p) - 0.5 * r_p' * r_p

        n_p = (log_u <= energy) * 1
        s_p = energy > log_u - info.delta_max
        if !s_p
            println("Divergent: $energy <= $log_u - $(info.delta_max)")
        end
        return theta_p, r_p, theta_p, r_p, theta_p, n_p, s_p
    else
        # Recursion - implicitly build the left and right subtree
        theta_neg, r_neg, theta_pos, r_pos, theta_p, n_p, s_p = BuildTree2(info, theta, r, log_u, v, j-1)
        if s_p
            if v == -1.
                theta_neg, r_neg, _, _, theta_pp, n_pp, s_pp = BuildTree2(info, theta_neg, r_neg, log_u, v, j-1)
            else
                _, _, theta_pos, r_pos, theta_pp, n_pp, s_pp = BuildTree2(info, theta_pos, r_pos, log_u, v, j-1)
            end
            if rand() < n_pp / (n_p + n_pp)
                theta_p = theta_pp
            end
            theta_diff = theta_pos .- theta_neg
            s_p = s_pp & (theta_diff' * r_neg >= 0) & (theta_diff' * r_pos >= 0)
            n_p = n_p + n_pp
        end
        return theta_neg, r_neg, theta_pos, r_pos, theta_p, n_p, s_p
    end
end
=#

function NextSample(info::NUTSSamplerInfo, prev::NUTSState)
    r0 = r_neg = r_pos = randn(info.size_p)
    energy = prev.prob - 0.5 * r0' * r0
    log_u = log(rand()) + energy # u ~ U([0, exp(L(theta) - 0.5 r0' * r0)])

    theta_m = theta_neg = theta_pos = prev.theta
    j = 0 # cureent tree depth
    n = 1 # considered size
    s = true # continue ?

    grad = similar(theta_m)
    gradient!(grad, info.likeli, theta_m)
    grad_neg = grad_pos = grad

    while s
        if j > info.max_tree_depth
            println("Reach max tree depth, increase eps or max_tree_depth to fix")
            break
        end
        v = round(rand())*2-1 # -1. or 1.
        if v == -1.
            theta_neg, r_neg, _, _, theta_p, n_p, s_p, grad_neg = BuildTree(info, theta_neg, r_neg, log_u, v, j, info.eps, grad_neg)
        else
            _, _, theta_pos, r_pos, theta_p, n_p, s_p, grad_pos = BuildTree(info, theta_pos, r_pos, log_u, v, j, info.eps, grad_pos)
        end
        if s_p
            if rand() < n_p / n
                theta_m = theta_p
            end
        end
        n = n + n_p
        theta_diff = theta_pos .- theta_neg
        s = s_p & (theta_diff' * r_neg >= 0) & (theta_diff' * r_pos >= 0)
        j = j + 1
    end
    return NUTSState(theta_m, info.likeli(theta_m)), NUTSTransitionInfo(log_u, j, n)
end

function BuildTree(info::NUTSSamplerInfo, theta::Vector{T}, r::Vector{T}, log_u::T, v::T, j::Int, eps::T,
                   grad::Vector{T}) where T
    if j == 0
        # Base case - take one leapfrog step in the direction v
        theta_p, r_p, grad = LeapfrogCache(info.likeli, theta, r, v * info.eps, grad)
        energy = info.likeli(theta_p) - 0.5 * r_p' * r_p

        n_p = (log_u <= energy) * 1
        s_p = energy > log_u - info.delta_max
        if !s_p
            println("Divergent: $energy <= $log_u - $(info.delta_max)")
        end
        return theta_p, r_p, theta_p, r_p, theta_p, n_p, s_p, grad
    else
        # Recursion - implicitly build the left and right subtree
        theta_neg, r_neg, theta_pos, r_pos, theta_p, n_p, s_p, grad = BuildTree(info, theta, r, log_u, v, j-1, eps, grad)
        if s_p
            if v == -1.
                theta_neg, r_neg, _, _, theta_pp, n_pp, s_pp, grad = BuildTree(info, theta_neg, r_neg, log_u, v, j-1, eps, grad)
            else
                _, _, theta_pos, r_pos, theta_pp, n_pp, s_pp, grad = BuildTree(info, theta_pos, r_pos, log_u, v, j-1, eps, grad)
            end
            if rand() < n_pp / (n_p + n_pp)
                theta_p = theta_pp
            end
            theta_diff = theta_pos .- theta_neg
            s_p = s_pp & (theta_diff' * r_neg >= 0) & (theta_diff' * r_pos >= 0)
            n_p = n_p + n_pp
        end
        return theta_neg, r_neg, theta_pos, r_pos, theta_p, n_p, s_p, grad
    end
end

# warmup


struct NUTSWarmupSamplerInfo{F <: Function, T} <: AbstractNUTSSamplerInfo
    likeli::F
    size_p::Int

    delta::T
    mu::T # log(10*eps0)

    max_tree_depth::Int
    delta_max::T

    gamma::T
    t0::T
    kappa::T
end

function NUTSWarmupSamplerInfo(likeli::Function, size_p::Int, delta::T, mu::T) where T
    max_tree_depth = 15
    delta_max = T(1000)
    gamma =  T(0.05)
    t0 = T(10)
    kappa = T(0.75)
    NUTSWarmupSamplerInfo(likeli, size_p, delta, mu, max_tree_depth, delta_max, gamma, t0, kappa)
end

struct NUTSWarmupState{TV <: AbstractVector, T} <: AbstractNUTSState
    theta::TV
    prob::T
    eps::T
    eps_bar::T
    H_bar::T
    m::Int
end

function NUTSWarmupState(likeli::Function, theta::Vector)
    prob = likeli(theta)
    eps = FindReasonableEpsilon(likeli, theta)
    eps_bar = 1.
    H_bar = 0.0
    m = 2
    NUTSWarmupState(theta, prob, eps, eps_bar, H_bar, m)
end

function setup(likeli::Function, size_p::Int, ::Type{NUTSWarmupSamplerInfo}, ::Type{NUTSWarmupState})
    theta0 = draw_init(size_p)
    nuts_warmup_state = NUTSWarmupState(likeli, theta0)
    #delta = 0.65
    delta = 0.75
    #delta = 0.85
    mu = log(10 * nuts_warmup_state.eps)
    nuts_warmup_sampler_info = NUTSWarmupSamplerInfo(likeli, size_p, delta, mu)
    return nuts_warmup_sampler_info, nuts_warmup_state
end

struct NUTSWarmupTransitionInfo{T} <: TransitionInfo
    log_u::T # slice value
    tree_depth::Int # tree depth
    n::Int # considered size
    alpha::T
    n_alpha::T
end

transition_type(::Type{NUTSWarmupSamplerInfo{F, T}}) where {F, T} = NUTSWarmupTransitionInfo{T}


function NextSample(info::NUTSWarmupSamplerInfo, prev::NUTSWarmupState)
    r0 = r_neg = r_pos = randn(info.size_p)
    energy0 = prev.prob - 0.5 * r0' * r0
    log_u = log(rand()) + energy0 # u ~ U([0, exp(L(theta) - 0.5 r0' * r0)])

    theta_m = theta_neg = theta_pos = prev.theta
    j = 0 # cureent tree depth
    n = 1 # considered size
    s = true # continue ?

    grad = similar(theta_m)
    gradient!(grad, info.likeli, theta_m)
    grad_neg = grad_pos = grad

    local alpha, n_alpha

    while s
        if j > info.max_tree_depth
            println("Reach max tree depth, increase eps or max_tree_depth to fix")
            break
        end
        v = round(rand())*2-1 # -1. or 1.
        if v == -1.
            theta_neg, r_neg, _, _, theta_p, n_p, s_p, alpha, n_alpha, grad_neg = BuildTree(info, theta_neg, r_neg, log_u, v, j, prev.eps, energy0, grad_neg)
        else
            _, _, theta_pos, r_pos, theta_p, n_p, s_p, alpha, n_alpha, grad_pos = BuildTree(info, theta_pos, r_pos, log_u, v, j, prev.eps, energy0, grad_pos)
        end
        if s_p
            if rand() < n_p / n
                theta_m = theta_p
            end
        end
        n = n + n_p
        theta_diff = theta_pos .- theta_neg
        s = s_p & (theta_diff' * r_neg >= 0) & (theta_diff' * r_pos >= 0)
        j = j + 1
    end
    w = 1. / (prev.m + info.t0)
    H_bar = (1. - w) * prev.H_bar + w * (info.delta - alpha / n_alpha)
    log_eps = info.mu - sqrt(prev.m) / info.gamma * H_bar
    w = prev.m ^ (-info.kappa)
    log_eps_bar = w * log_eps + (1-w)*log(prev.eps_bar)

    return NUTSWarmupState(theta_m, info.likeli(theta_m), exp(log_eps), exp(log_eps_bar), H_bar, prev.m + 1), NUTSWarmupTransitionInfo(log_u, j, n, alpha, n_alpha)
end

function BuildTree(info::NUTSWarmupSamplerInfo, theta::Vector{T}, r::Vector{T}, log_u::T, v::T, j::Int, eps::T,
                   energy0::T, grad::Vector{T}) where T
    if j == 0
        # Base case - take one leapfrog step in the direction v
        theta_p, r_p, grad = LeapfrogCache(info.likeli, theta, r, v * eps, grad)
        energy = info.likeli(theta_p) - 0.5 * r_p' * r_p

        n_p = (log_u <= energy) * 1
        s_p = energy > log_u - info.delta_max
        if !s_p
            println("Warmup Divergent: $energy <= $log_u - $(info.delta_max)")
        end
        alpha_p = min(1., exp(energy - energy0))
        return theta_p, r_p, theta_p, r_p, theta_p, n_p, s_p, alpha_p, 1., grad
    else
        # Recursion - implicitly build the left and right subtree
        theta_neg, r_neg, theta_pos, r_pos, theta_p, n_p, s_p, alpha_p, n_alpha_p, grad = BuildTree(info, theta, r, log_u, v, j-1, eps, energy0, grad)
        if s_p
            if v == -1.
                theta_neg, r_neg, _, _, theta_pp, n_pp, s_pp, alpha_pp, n_alpha_pp, grad = BuildTree(info, theta_neg, r_neg, log_u, v, j-1, eps, energy0, grad)
            else
                _, _, theta_pos, r_pos, theta_pp, n_pp, s_pp, alpha_pp, n_alpha_pp, grad = BuildTree(info, theta_pos, r_pos, log_u, v, j-1, eps, energy0, grad)
            end
            if rand() < n_pp / (n_p + n_pp)
                theta_p = theta_pp
            end
            alpha_p = alpha_p + alpha_pp
            n_alpha_p = n_alpha_p + n_alpha_pp
            theta_diff = theta_pos .- theta_neg
            s_p = s_pp & (theta_diff' * r_neg >= 0) & (theta_diff' * r_pos >= 0)
            n_p = n_p + n_pp
        end
        return theta_neg, r_neg, theta_pos, r_pos, theta_p, n_p, s_p, alpha_p, n_alpha_p, grad
    end
end

#=
struct NUTSSamplerInfo{F <: Function, T} <: AbstractNUTSSamplerInfo
    likeli::F
    eps::T
    size_p::Int

    max_tree_depth::Int
    delta_max::T
end

struct NUTSState{TV <: AbstractVector, T} <: AbstractNUTSState
    theta::TV
    prob::T
end

=#

function NUTSSamplerInfo(info::NUTSWarmupSamplerInfo, prev::NUTSWarmupState)
    NUTSSamplerInfo(info.likeli, prev.eps_bar, info.size_p, info.max_tree_depth, info.delta_max)
end

function NUTSState(info::NUTSWarmupSamplerInfo, prev::NUTSWarmupState)
    NUTSState(prev.theta, prev.prob)
end