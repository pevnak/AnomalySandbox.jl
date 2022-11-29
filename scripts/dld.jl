using AnomalySandbox
using AnomalySandbox: two_bananas, update_params!
using ConditionalDists
using StatsBase
using Flux
using GLMakie
using OneHotArrays
include(joinpath(dirname(pathof(AnomalySandbox)), "plotting.jl"))

sdir(s...) = joinpath("/Users/tomas.pevny/Work/Teaching/Anomaly/animations", s...)
dataset1 = train_test_split(lofdata()...)
dataset2 = train_test_split(two_bananas()...)

function sample_noise(xd, yd, n)
	vcat(rand(xd, n)',rand(yd, n)')
end

function train_model(xtrn, (xd, yd);hdim, nl, xdim = size(xtrn,1), batchsize = size(xtrn, 2), act = relu, steps = 10000)
	model = AnomalySandbox.stack_layers([xdim, fill(hdim, nl)..., 2], act)
	opt = ADAM()
	y = Flux.onehotbatch(vcat(fill(0, size(xtrn,2)), fill(1, batchsize)), [0,1])
	loss(x) = Flux.Losses.logitcrossentropy(model(x), y)
	for s in 1:steps
		x = hcat(xtrn, sample_noise(xd, yd, batchsize))
		update_params!(model, (x,), loss, opt)
	end
	model
end

plot_no = 1
for (hdim, nl) in Iterators.product([8, 16, 32, 64, 128], [1, 2, 3])
	global plot_no
	fig = Figure(resolution = (1800, 700));
    ga = fig[1, 1] = GridLayout()

	for (i, dataset) in enumerate([dataset1, dataset2])
		xtrn = dataset[[1,3]]
        model = train_model(xtrn...; hdim, nl)
	    ax = Axis(ga[1, i])
	    heatcontplot!(ax, x -> softmax(model(x))[2,:], dataset);
	end
	label = "hidden dim = $(hdim) layers = $(nl) network no: $(plot_no)"
	Label(ga[1, 1:2, Top()], label, valign = :bottom,
	    font = "TeX Gyre Heros Bold", textsize = 26,
	    padding = (0, 0, 5, 0))

	save(sdir("dld_$(plot_no).png"), fig)
	plot_no += 1
end
