using Plots
using AnomalySandbox
using AnomalySandbox: two_bananas, maf_loss
using StatsBase
using Flux
using ConditionalDists
using Distributions

outdir = "maf_plots"
mkpath(outdir)

# create train and test set
Xn, Xa = two_bananas()
Xn = Float32.(Xn)
Xa = Float32.(Xa)
n = Int(size(Xn,2)/2)
na = size(Xa, 2)
ninds = sample(1:2*n, 2*n, replace=false)
Xtr = Xn[:,ninds[1:n]]
Xtst = hcat(Xn[:,ninds[n+1:end]], Xa)
ytst = vcat(zeros(Int,n), ones(Int,na))

# plot the data
scatter(Xtr[1,:], Xtr[2,:], label="train normal")
scatter!(Xtst[1,ytst.==0], Xtst[2,ytst.==0], label="test normal")
scatter!(Xtst[1,ytst.==1], Xtst[2,ytst.==1], label="test anomalous")
savefig(joinpath(outdir, "train_test.png"))

# build the MAF model
xdim = size(Xtr,1)
"""
kwargs:
	num_flows = 4,
    num_layers = 2,
    act_loc = "relu",
    act_scl = "relu",
    bn = true,
    hsize = 5,
    ordering = "natural",
    seed = 42,
    lastzero = true,
    lastlayer = "linear"
"""
model = MAF(xdim)
opt = ADAM(1e-4)
base = MvNormal(xdim, 1.0f0)
data = [(Xtr,) for i in 1:100];
loss(x) = maf_loss(x, model, base)

loss(data[1]...)
Flux.train!(loss, Flux.params(model), data, opt)
loss(data[1]...)
testmode!(model, true)

# get train, test scores
tr_scores = predict(model, Xtr, base)
tst_scores = predict(model, Xtst, base)
scatter(Xtst[1,:], Xtst[2,:], marker_z=tst_scores)
savefig(joinpath(outdir, "test_scores.png"))

# now plot scores on a grid
Xgr = AnomalySandbox.get_2D_grid([-4,4], [-3,3], 0.1)	
gr_scores = predict(model, Xgr, base)
scatter(Xgr[1,:], Xgr[2,:], marker_z=gr_scores)
scatter!(Xn[1,:], Xn[2,:], label="normal")
scatter!(Xa[1,:], Xa[2,:], label="anomalous")
savefig(joinpath(outdir, "grid_scores.png"))
