using Plots
using AnomalySandbox
using AnomalySandbox: two_bananas
using StatsBase
using Flux
using ConditionalDists

outdir = "vae_plots"
mkpath(outdir)

# create train and test set
Xn, Xa = two_bananas()
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

# construct the VAE
zdim = 1
xdim = size(Xn, 1)
hdim = 32
act = relu
enc = Chain(
	AnomalySandbox.stack_layers([xdim, hdim, hdim], act, last=act)...,
	SplitLayer(hdim, [zdim, zdim], [identity, softplus])
	)
enc_dist = ConditionalMvNormal(enc)
dec = Chain(
	AnomalySandbox.stack_layers([zdim, hdim, hdim], act, last=act)...,
	SplitLayer(hdim, [xdim, 1], [identity, softplus])
	)
dec_dist = ConditionalMvNormal(dec)
model = VAE(zdim, enc_dist, dec_dist)

# now train it
opt = ADAM()
data = [(Xtr,) for i in 1:2000];
lossf(x) = - elbo(model, x, β=1e0)
Flux.train!(lossf, Flux.params(model), data, opt)

# get train, test scores
tr_scores = predict(model, Xtr)
tst_scores = predict(model, Xtst)
scatter(Xtst[1,:], Xtst[2,:], marker_z=tst_scores)
savefig(joinpath(outdir, "test_scores.png"))

# now plot scores on a grid
Xgr = AnomalySandbox.get_2D_grid([-4,4], [-3,3], 0.1)	
gr_scores = predict(model, Xgr)
gr_scores[gr_scores .> 50] .= 50 # we have to threshold it so there is something visible
scatter(Xgr[1,:], Xgr[2,:], marker_z=gr_scores)
scatter!(Xn[1,:], Xn[2,:], label="normal")
scatter!(Xa[1,:], Xa[2,:], label="anomalous")
savefig(joinpath(outdir, "grid_scores.png"))
