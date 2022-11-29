using AnomalySandbox
using AnomalySandbox: two_bananas, lofdata
using AnomalySandbox.IsolationForest: train_iforest
using StatsBase
using GLMakie
include(joinpath(dirname(pathof(AnomalySandbox)), "plotting.jl"))

sdir(s...) = joinpath("/Users/tomas.pevny/Work/Teaching/Anomaly/animations", s...)
dataset1 = train_test_split(lofdata()...)
dataset2 = train_test_split(two_bananas()...)

plot_no = 1
for (num_trees, sub_sampling_size) in Iterators.product([25,50,100], [64, 128, 256])
    fig = Figure(resolution = (1800, 700));
    ga = fig[1, 1] = GridLayout()

	for (i, dataset) in enumerate([dataset1, dataset2])
		xtrn = dataset[1]
	    ax = Axis(ga[1, i])
	    model = train_iforest(xtrn; num_trees, sub_sampling_size)
	    heatcontplot!(ax, x -> predict(model, x), dataset);
	end
	label = "num_trees = $(num_trees) sub_sampling_size = $(sub_sampling_size)"
	Label(ga[1, 1:2, Top()], label, valign = :bottom,
	    font = "TeX Gyre Heros Bold", textsize = 26,
	    padding = (0, 0, 5, 0))

    save(sdir("if_$(plot_no).png"), fig)
    plot_no += 1
end
