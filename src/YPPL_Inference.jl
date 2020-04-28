module YPPL_Inference

#using ForwardDiff: gradient
using ForwardDiff: gradient!

include("mcmc.jl")
include("hmc.jl")
include("nuts.jl")

end # module
