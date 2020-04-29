using Test

@time using YPPL_Parser.Examples.eight_schools_non_centered: likeli, p
@time using YPPL_Diagnosis: mcmc_summary
@time using YPPL_Inference
@time using DistributionsAD

using YPPL_Inference: HMCSamplerInfo, HMCState, HMCStateCached, HMCWarmupState, HMCWarmupSamplerInfo, sampling, sampling_split, extract, extract_full, setup
using YPPL_Parser.Examples.ref_eight_schools_non_centered: decode, reference_mean
import DataFrames: insertcols!

M = 1000
chains = 4
size_p = 10
theta0 = ones(size_p)
eps = 1e-1
L = 15

sampler_info = HMCSamplerInfo(likeli, eps, size_p, L)
state = HMCState(likeli, theta0)

@time state_list_list, transition_list_list = sampling(sampler_info, state, M, chains)
posterior = extract(state_list_list)

df = mcmc_summary(decode(extract(state_list_list)))
insertcols!(df, 1, :params => [["mu", "tau"]; ["theta[$i]" for i in 1:8]])
println(df)

@test all(abs.(df.mean .- reference_mean) .< df.se_mean*3)

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

@test all(abs.(df.mean .- reference_mean) .< df.se_mean*3)
