using Test

@time using YPPL_Parser.Examples.eight_schools_non_centered: likeli, p
@time using YPPL_Diagnosis: mcmc_summary
@time using YPPL_Inference
@time using DistributionsAD

using YPPL_Inference.MCMC: sampling, sampling_split, extract, extract_full, setup, 
                      HMCSamplerInfo, HMCState, HMCStateCached, HMCWarmupState, HMCWarmupSamplerInfo,
                      NUTSSamplerInfo, NUTSState, NUTSWarmupSamplerInfo, NUTSWarmupState
 
using YPPL_Parser.Examples.ref_eight_schools_non_centered: decode, reference_mean, schools_dat
import DataFrames: insertcols!

M = 1000
chains = 4
size_p = 10
theta0 = ones(size_p)
eps = 3e-1
L = 15

sampler_info = HMCSamplerInfo(likeli, eps, size_p, L)
state = HMCState(likeli, theta0)

@time state_list_list, transition_list_list = sampling(sampler_info, state, M, chains)
posterior = extract(state_list_list)

df = mcmc_summary(decode(extract(state_list_list)))
insertcols!(df, 1, :params => [["mu", "tau"]; ["theta[$i]" for i in 1:8]])
println(df)

@test all(abs.(df.mean .- reference_mean) .< df.se_mean*4)

hmc_warmup_sampler_info, hmc_warmup_state = setup(likeli, 10, HMCWarmupSamplerInfo, HMCWarmupState)

@time warm_state_list, warm_transition_list, state_list, transition_list = sampling_split(
    hmc_warmup_sampler_info, hmc_warmup_state, 500,
    HMCSamplerInfo, HMCState, 500,
    4
)
posterior = extract_full(state_list)
df = mcmc_summary(decode(posterior))
insertcols!(df, 1, :params => [["mu", "tau"]; ["theta[$i]" for i in 1:8]])
println(df)

@test all(abs.(df.mean .- reference_mean) .< df.se_mean*4)

nuts_warmup_sampler_info, nuts_warmup_state = setup(likeli, size_p, NUTSWarmupSamplerInfo, NUTSWarmupState)

@time warmup_state_list, warmup_transition_list, state_list, transition_list = sampling_split(
    nuts_warmup_sampler_info, nuts_warmup_state, 500,
    NUTSSamplerInfo, NUTSState, 500, 
    4)
df = state_list |> extract_full |> decode |> mcmc_summary
println(df)

@test all(abs.(df.mean .- reference_mean) .< df.se_mean*4)


@time using Plots
unicodeplots()

@time plot([state.eps for state in warmup_state_list[1]])
title!("eps")

plot([log(state.eps) for state in warmup_state_list[1]])
title!("log(eps)")

plot([state.theta[1] for state in state_list[1]])
title!("mu")

include("./test_laplace.jl")