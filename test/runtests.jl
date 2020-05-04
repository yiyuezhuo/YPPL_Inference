using Test

@time using YPPL_Parser.Examples.eight_schools_non_centered: likeli
@time using YPPL_Diagnosis: mcmc_summary
@time using YPPL_Inference
@time using DistributionsAD
@time using Plots
@time using Optim


include("./test_mcmc.jl")
include("./test_laplace.jl")
include("./test_vi.jl")