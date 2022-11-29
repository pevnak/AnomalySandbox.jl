using AnomalySandbox
using AnomalySandbox: two_bananas, lofdata
using LIBSVM
using StatsBase
using GLMakie
include(joinpath(dirname(pathof(AnomalySandbox)), "plotting.jl"))

sdir(s...) = joinpath("/Users/tomas.pevny/Work/Teaching/Anomaly/animations", s...)
dataset1 = train_test_split(lofdata()...)
dataset2 = train_test_split(two_bananas()...)

plot_no = 1
for (nu, gamma) in Iterators.product([0.01,0.05, 0.1, 0.5, 0.9, 0.95, 0.99], 2.0 .^ (-3:3))
    fig = Figure(resolution = (1800, 700));
    ga = fig[1, 1] = GridLayout()

	for (i, dataset) in enumerate([dataset1, dataset2])
		xtrn = dataset[1]
	    ax = Axis(ga[1, i])
	    model = svmtrain(xtrn; svmtype = NuSVC, kernel = LIBSVM.Kernel.RadialBasis, gamma, nu)
	    heatcontplot!(ax, x -> -svmpredict(model, x)[2][1,:], dataset);
	end
	label = "nu = $(nu) gamma = $(gamma)"
	Label(ga[1, 1:2, Top()], label, valign = :bottom,
	    font = "TeX Gyre Heros Bold", textsize = 26,
	    padding = (0, 0, 5, 0))

    save(sdir("ocsvm_$(plot_no).png"), fig)
    plot_no += 1
end
