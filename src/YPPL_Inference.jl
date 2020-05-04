module YPPL_Inference

#using ForwardDiff: gradient

include("mcmc/MCMC.jl")
include("laplace/Laplace.jl")
include("optimizing/Optimizing.jl")
include("vi/VI.jl")
include("utils.jl")

end # module
