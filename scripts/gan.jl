using AnomalySandbox
using AnomalySandbox: two_bananas, update_params!
using ConditionalDists
using StatsBase
using Flux
using GLMakie
include(joinpath(dirname(pathof(AnomalySandbox)), "plotting.jl"))

# create train and test set
xtrn, (xtst, ytst) = train_test_split(two_bananas()...)
# construct the GAN
zdim = 1
xdim = size(xtrn, 1)
batchsize = size(xtrn, 2)
act = relu

plot_no = 1
for (hdim, nl) in Iterators.product([16, 32, 64], [1,2,3])
	for i in 1:5
		global plot_no
		disc = AnomalySandbox.stack_layers([xdim, fill(hdim, nl)..., 1], act, last=Flux.σ)
		disc_dist = ConditionalMvNormal(disc)

		gen = AnomalySandbox.stack_layers([zdim, fill(hdim, nl)..., xdim], act)
		gen_dist = ConditionalMvNormal(gen)

		model = GAN(zdim, gen_dist, disc_dist)

		# now train it
		gen_opt = ADAM()
		disc_opt = ADAM()
		data =  Iterators.repeated((xtrn,), 2000);
		gloss(x) = generator_loss(model, batchsize) # gen_loss does not need input data
		dloss(x) = discriminator_loss(model, x)
		for _data in data
			update_params!(model.generator, _data, gloss, gen_opt)
			update_params!(model.discriminator, _data, dloss, disc_opt)
		end

		fig = heatcontplot(x -> predict(model, x), -4:0.01:4, -3:0.01:3, xtrn, xtst, ytst);
		fig[0, :] = Label(fig, "inner dim = $(hdim) layers = $(nl) network no: $(plot_no)")
		save("../plots/bananas/gan_$(plot_no).png", fig)
		plot_no += 1
	end
end
