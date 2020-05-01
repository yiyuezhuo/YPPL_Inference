using Test

@time using YPPL_Parser.Examples.eight_schools_non_centered: likeli, p
@time using YPPL_Diagnosis: mcmc_summary
@time using YPPL_Inference
@time using DistributionsAD

using YPPL_Inference.MCMC: sampling, sampling_split, extract, extract_full, setup, 
                      HMCSamplerInfo, HMCState, HMCStateCached, HMCWarmupState, HMCWarmupSamplerInfo,
                      NUTSSamplerInfo, NUTSState, NUTSWarmupSamplerInfo, NUTSWarmupState
 
using YPPL_Parser.Examples.ref_eight_schools_non_centered: decode, reference_mean, size_p
import DataFrames: insertcols!

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

