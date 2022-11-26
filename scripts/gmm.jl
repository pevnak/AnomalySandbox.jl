using AnomalySandbox
using AnomalySandbox: two_bananas, fit_gmm
using ConditionalDists
using Distributions
using StatsBase
using Flux
using GLMakie
include(joinpath(dirname(pathof(AnomalySandbox)), "plotting.jl"))

xtrn, (xtst, ytst) = train_test_split(two_bananas()...)

# construct the VAE
zdim = 1
xdim = size(xtrn, 1)

plot_no = 1
for nc in [2, 4, 8, 16, 32]
    model = fit_gmm(nc, xtrn) # for more options see the documentations of GaussianMixtures.jl

    fig = heatcontplot(x -> predict(model, x), -4:0.01:4, -3:0.01:3, xtrn, xtst, ytst);
    fig[0, :] = Label(fig, "inner dim = $(hdim) blocks = $(nf) layers = $(nl) network no: $(plot_no)")
    save("../plots/bananas/maf_$(plot_no).png", fig)
    plot_no += 1
end
