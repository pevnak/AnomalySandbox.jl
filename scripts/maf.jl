using AnomalySandbox
using AnomalySandbox: two_bananas, maf_loss
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
for (hdim, nl, nf) in Iterators.product([2, 4, 8], [1,2,3], [2,4,8])
    for i in 1:5
        global plot_no
        """
        kwargs:
            num_flows = 4,
            num_layers = 2,
            act_loc = "relu",
            act_scl = "relu",
            bn = true,
            hsize = 5,
            ordering = "natural",
            seed = 42,
            lastzero = true,
            lastlayer = "linear"
        """
        model = MAF(xdim; num_flows = nf, hsize = hdim, num_layers = nl, seed = i)
        opt = ADAM(1e-4)
        base = MvNormal(xdim, 1.0f0)
        data = Iterators.repeated((xtrn,), 200);
        loss(x) = maf_loss(x, model, base)

        loss(xtrn)
        trainmode!(model, true)
        Flux.train!(loss, Flux.params(model), data, opt)
        loss(xtrn)
        testmode!(model, true)

        fig = heatcontplot(x -> predict(model, x, base), -4:0.01:4, -3:0.01:3, xtrn, xtst, ytst);
        fig[0, :] = Label(fig, "inner dim = $(hdim) blocks = $(nf) layers = $(nl) network no: $(plot_no)")
        save("../plots/bananas/maf_$(plot_no).png", fig)
        plot_no += 1
    end
end