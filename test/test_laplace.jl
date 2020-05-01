using Test

using YPPL_Inference.Laplace: laplace_approximate
using Distributions
using YPPL_Parser.Examples.eight_schools_non_centered: likeli, schools_dat
using YPPL_Parser.Examples.ref_eight_schools_non_centered: reference_mean, reference_std
using DataFrames: DataFrame

mu, Sigma = laplace_approximate(likeli, ones(10))

dist = MvNormal(mu, Sigma)

sampled = rand(dist, 10000)

sampled[2, :] = exp.(sampled[2, :])
sampled[3:10, :] = (sampled[1, :] .+ sampled[2, :] .* sampled[3:10, :]')'

mean_theta = mean(sampled[3:10, :], dims=2)
std_theta = std(sampled[3:10, :], dims=2)

@test all(abs.(schools_dat.y .- mean_theta) .< std_theta)

df = DataFrame(mean_theta=mean_theta[:, 1], std_theta=std_theta[:, 1], 
               y=schools_dat.y, obs_sigma=schools_dat.sigma, 
               reference_mean=reference_mean[3:10], reference_std=reference_std[3:10])
println(df)