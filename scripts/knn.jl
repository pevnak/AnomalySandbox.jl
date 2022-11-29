using AnomalySandbox
using AnomalySandbox: two_bananas, lofdata
using OutlierDetection 
using OutlierDetectionNeighbors: KNNDetector
using StatsBase
using Flux
using GLMakie
include(joinpath(dirname(pathof(AnomalySandbox)), "plotting.jl"))

sdir(s...) = joinpath("/Users/tomas.pevny/Work/Teaching/Anomaly/animations", s...)
dataset1 = train_test_split(lofdata()...)
dataset2 = train_test_split(two_bananas()...)

plot_no = 1
for (k, reduction) in Iterators.product(2 .^ (1:6), [:mean, :median, :maximum])
    fig = Figure(resolution = (1800, 700));
    ga = fig[1, 1] = GridLayout()

	for (i, dataset) in enumerate([dataset1, dataset2])
		xtrn = dataset[1]
	    ax = Axis(ga[1, i])
	    d = KNNDetector(;k, reduction)
	    model, _ = OutlierDetection.fit(d, xtrn; verbosity = 0)
	    heatcontplot!(ax, x -> OutlierDetection.transform(d, model, x), dataset);
	end
	label = "k = $(k) reduction = $(reduction)"
	Label(ga[1, 1:2, Top()], label, valign = :bottom,
	    font = "TeX Gyre Heros Bold", textsize = 26,
	    padding = (0, 0, 5, 0))

    save(sdir("knn_$(plot_no).png"), fig)
    plot_no += 1
end
