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

function train_model(xtrn;hdim, nl, zdim = 1, xdim = size(xtrn,1), batchsize = size(xtrn, 2), act = relu, steps = 2000)
	disc = AnomalySandbox.stack_layers([xdim, fill(hdim, nl)..., 1], act, last=Flux.σ)
	disc_dist = ConditionalMvNormal(disc)

	gen = AnomalySandbox.stack_layers([zdim, fill(hdim, nl)..., xdim], act)
	gen_dist = ConditionalMvNormal(gen)

	model = GAN(zdim, gen_dist, disc_dist)

	gen_opt = ADAM()
	disc_opt = ADAM()
	data =  Iterators.repeated((xtrn,), steps);
	gloss(x) = generator_loss(model, batchsize) # gen_loss does not need input data
	dloss(x) = discriminator_loss(model, x)
	for _data in data
		update_params!(model.generator, _data, gloss, gen_opt)
		update_params!(model.discriminator, _data, dloss, disc_opt)
	end
	model
end

plot_no = 1
for (hdim, nl) in Iterators.product([16, 32, 64], [1,2,3])
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

		save(sdir("gan_$(plot_no).png"), fig)
		plot_no += 1
	end
end
