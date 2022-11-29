using AnomalySandbox
using AnomalySandbox: two_bananas, lofdata, PCADetector
using StatsBase
using GLMakie
include(joinpath(dirname(pathof(AnomalySandbox)), "plotting.jl"))

sdir(s...) = joinpath("/Users/tomas.pevny/Work/Teaching/Anomaly/animations", s...)
dataset1 = train_test_split(lofdata()...)
dataset2 = train_test_split(two_bananas()...)

plot_no = 1
fig = Figure(resolution = (1800, 700));
ga = fig[1, 1] = GridLayout()

for (i, dataset) in enumerate([dataset1, dataset2])
	xtrn = dataset[1]
    model = PCADetector(xtrn, 1)
    ax = Axis(ga[1, i])
    heatcontplot!(ax, x -> predict(model, x), dataset);
end
save(sdir("pca_$(plot_no).png"), fig)
