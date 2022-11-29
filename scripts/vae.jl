using AnomalySandbox
using AnomalySandbox: two_bananas, update_params!
using ConditionalDists
using StatsBase
using Flux
using GLMakie
include(joinpath(dirname(pathof(AnomalySandbox)), "plotting.jl"))

sdir(s...) = joinpath("/Users/tomas.pevny/Work/Teaching/Anomaly/animations", s...)
dataset1 = train_test_split(lofdata()...)
dataset2 = train_test_split(two_bananas()...)

function train_model(xtrn;hdim, nl, zdim = 1, xdim = size(xtrn,1), act = relu)
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
	opt = ADAM()
	lossf(x) = - elbo(model, x, β=1e0)
	data =  Iterators.repeated((xtrn,), 2000);
	Flux.train!(lossf, Flux.params(model), data, opt)
	model
end

plot_no = 1
for (hdim, nl) in Iterators.product([2, 4, 8, 16], [1,2,3])
	for i in 1:3
		global plot_no
		fig = Figure(resolution = (1800, 700));
	    ga = fig[1, 1] = GridLayout()

		for (i, dataset) in enumerate([dataset1, dataset2])
			xtrn = dataset[1]
	        model = train_model(xtrn; hdim, nl)
		    ax = Axis(ga[1, i])
		    heatcontplot!(ax, x -> predict(model, x), dataset);
		end
		label = "hidden dim = $(hdim) layers = $(nl) network no: $(plot_no)"
		Label(ga[1, 1:2, Top()], label, valign = :bottom,
		    font = "TeX Gyre Heros Bold", textsize = 26,
		    padding = (0, 0, 5, 0))

		save(sdir("vae_$(plot_no).png"), fig)
		plot_no += 1
	end
end
