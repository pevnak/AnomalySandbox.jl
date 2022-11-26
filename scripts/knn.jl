using AnomalySandbox
using AnomalySandbox: two_bananas, Parzen
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
for σ in Float32.( 2 .^ collect(-5:5))
    model = KNN(xtrn, k) # for more options see the documentations of GaussianMixtures.jl

    fig = heatcontplot(x -> predict(model, x), -4:0.01:4, -3:0.01:3, xtrn, xtst, ytst);
    fig[0, :] = Label(fig, "sigma = $(σ)")
    save("../plots/bananas/knn_$(plot_no).png", fig)
    plot_no += 1
end
