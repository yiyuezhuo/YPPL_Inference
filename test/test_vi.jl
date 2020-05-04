using Test

using Plots
unicodeplots()

using YPPL_Inference.VI: vi, SGD, Adam, MeanField, Rank1, LowRank
using YPPL_Inference.utils: KL_MvNormal
using Statistics
using DataFrames: DataFrame
using YPPL_Parser.Examples.eight_schools_non_centered: likeli
using YPPL_Parser.Examples.ref_eight_schools_non_centered: mean_ref, std_ref, cov_ref, size_p
using LinearAlgebra


vp_list, elbo_list = vi(likeli, size_p, MeanField(), SGD(100000, 1e-3))

for i in [1,2,3,11]
    p = plot([vp[i] for vp in vp_list])
    title!("p[$i]")
    display(p)
end

vp_mat = hcat(vp_list...)

df = DataFrame(
    mean_ref = mean_ref,
    std_ref = std_ref,
    mean_MF_SGD = mean(vp_mat[1:10, end-10000:end], dims=2)[:, 1],
    std_MF_SGD = mean(vp_mat[11:20, end-10000:end], dims=2)[:, 1] .|> exp
)
println(df)

@test all(abs.(df.mean_ref - df.mean_MF_SGD) .< 1.0)

vp_list, elbo_list = vi(likeli, size_p, MeanField(), Adam(100000, 1e-3))
vp_mat = hcat(vp_list...)

mean_MF_Adam = mean(vp_mat[1:10, end-10000:end], dims=2)[:, 1]
std_MF_Adam = mean(vp_mat[11:20, end-10000:end], dims=2)[:, 1] .|> exp
Sigma_MF_Adam = diagm(std_MF_Adam .^ 2)

@show KL_MvNormal(mean_MF_Adam, Sigma_MF_Adam, mean_ref, cov_ref)
@show KL_MvNormal(mean_ref, cov_ref, mean_MF_Adam, Sigma_MF_Adam)

vp_list, elbo_list = vi(likeli, size_p, Rank1(), Adam(100000, 1e-3))
vp_mat = hcat(vp_list...)

mean_R1_Adam = mean(vp_mat[1:10, end-10000:end], dims=2)[:, 1]
diag_std = mean(vp_mat[11:20, end-10000:end], dims=2)[:, 1] .|> exp
alpha = mean(vp_mat[21:30, end-10000:end], dims=2)[:, 1]
Sigma_R1_Adam = diagm(diag_std .^ 2) + alpha * alpha'

@show KL_MvNormal(mean_R1_Adam, Sigma_R1_Adam, mean_ref, cov_ref)
@show KL_MvNormal(mean_ref, cov_ref, mean_R1_Adam, Sigma_R1_Adam)

vp_list, elbo_list = vi(likeli, size_p, LowRank(2), Adam(300000, 1e-3))
vp_mat = hcat(vp_list...)

mean_R2_Adam = mean(vp_mat[1:10, end-10000:end], dims=2)[:, 1]
diag_std = mean(vp_mat[11:20, end-10000:end], dims=2)[:, 1] .|> exp
U_raw = reshape(mean(vp_mat[21:40, end-10000:end], dims=2)[:, 1], size_p, 2)
Sigma_R2_Adam = diagm(diag_std .^ 2) + U_raw * U_raw'

@show KL_MvNormal(mean_R2_Adam, Sigma_R2_Adam, mean_ref, cov_ref)
@show KL_MvNormal(mean_ref, cov_ref, mean_R2_Adam, Sigma_R2_Adam)

vp_list, elbo_list = vi(likeli, size_p, LowRank(3), Adam(300000, 1e-3))
vp_mat = hcat(vp_list...)

mean_R3_Adam = mean(vp_mat[1:10, end-10000:end], dims=2)[:, 1]
diag_std = mean(vp_mat[11:20, end-10000:end], dims=2)[:, 1] .|> exp
U_raw = reshape(mean(vp_mat[21:end, end-10000:end], dims=2)[:, 1], size_p, 3)
Sigma_R3_Adam = diagm(diag_std .^ 2) + U_raw * U_raw'

@show KL_MvNormal(mean_R3_Adam, Sigma_R3_Adam, mean_ref, cov_ref)
@show KL_MvNormal(mean_ref, cov_ref, mean_R3_Adam, Sigma_R3_Adam)