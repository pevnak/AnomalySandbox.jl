using AnomalySandbox
using AnomalySandbox: two_bananas, lofdata, Parzen
using StatsBase
using Flux
using GLMakie
include(joinpath(dirname(pathof(AnomalySandbox)), "plotting.jl"))

sdir(s...) = joinpath("/Users/tomas.pevny/Work/Teaching/Anomaly/animations", s...)
dataset1 = train_test_split(lofdata()...)
dataset2 = train_test_split(two_bananas()...)

plot_no = 1
for σ in Float32.( 2.0 .^ collect(-4:0.5:3))
    fig = Figure(resolution = (1800, 700));
    ga = fig[1, 1] = GridLayout()

	for (i, dataset) in enumerate([dataset1, dataset2])
		xtrn = dataset[1]
        model = Parzen(xtrn, σ) # for more options see the documentations of GaussianMixtures.jl
	    ax = Axis(ga[1, i])
	    heatcontplot!(ax, x -> predict(model, x), dataset);
	end
	label = "σ = $(σ)"
	Label(ga[1, 1:2, Top()], label, valign = :bottom,
	    font = "TeX Gyre Heros Bold", textsize = 26,
	    padding = (0, 0, 5, 0))

    save(sdir("parzen_$(plot_no).png"), fig)
    plot_no += 1
end
