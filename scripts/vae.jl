using AnomalySandbox
using AnomalySandbox: two_bananas, update_params!
using ConditionalDists
using StatsBase
using Flux
using GLMakie
include(joinpath(dirname(pathof(AnomalySandbox)), "plotting.jl"))

xtrn, (xtst, ytst) = train_test_split(two_bananas()...)

zdim = 1
xdim = size(xtrn, 1)
act = relu

plot_no = 1
for (hdim, nl) in Iterators.product([2, 4, 8, 16], [1,2,3])
	for i in 1:5
		global plot_no
		enc = Chain(
			AnomalySandbox.stack_layers([xdim, fill(hdim, nl)...], act, last=act)...,
			SplitLayer(hdim, [zdim, zdim], [identity, softplus])
			)
		enc_dist = ConditionalMvNormal(enc)
		dec = Chain(
			AnomalySandbox.stack_layers([zdim, fill(hdim, nl)...], act, last=act)...,
			SplitLayer(hdim, [xdim, 1], [identity, softplus])
			)
		dec_dist = ConditionalMvNormal(dec)
		model = VAE(zdim, enc_dist, dec_dist)
		# now train it
		opt = ADAM()
		lossf(x) = - elbo(model, x, β=1e0)
		data =  Iterators.repeated((xtrn,), 2000);
		Flux.train!(lossf, Flux.params(model), data, opt)

		fig = heatcontplot(x -> predict(model, x), -4:0.01:4, -3:0.01:3, xtrn, xtst, ytst);
		fig[0, :] = Label(fig, "inner dim = $(hdim) layers = $(nl) network no: $(plot_no)")
		save("../plots/bananas/vae_$(plot_no).png", fig)
		plot_no += 1
	end
end
