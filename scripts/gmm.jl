using Plots
using AnomalySandbox
using AnomalySandbox: two_bananas, fit_gmm
using StatsBase

outdir = "gmm_plots"
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

# construct the GMM
nc = 10 # no. components
model = fit_gmm(nc, Xtr) # for more options see the documentations of GaussianMixtures.jl
   
# get train, test scores - weighted loglikelihood
tst_scores = predict(model, Xtst)
scatter(Xtst[1,:], Xtst[2,:], marker_z=tst_scores)
savefig(joinpath(outdir, "test_scores.png"))

# now plot scores on a grid
Xgr = Float32.(AnomalySandbox.get_2D_grid([-4,4], [-3,3], 0.1)	)
gr_scores = predict(model, Xgr)
scatter(Xgr[1,:], Xgr[2,:], marker_z=gr_scores)
scatter!(Xn[1,:], Xn[2,:], label="normal")
scatter!(Xa[1,:], Xa[2,:], label="anomalous")
savefig(joinpath(outdir, "grid_scores.png"))

# now compute the minimum loglikelihood score
# we don't compute the weighted averages across all components but
# get the loglikelihood from the most relevant (closest) one
tst_scores = predict(model, Xtst, score="min")
scatter(Xtst[1,:], Xtst[2,:], marker_z=tst_scores)
savefig(joinpath(outdir, "test_scores_min.png"))

# now plot scores on a grid
Xgr = Float32.(AnomalySandbox.get_2D_grid([-4,4], [-3,3], 0.1)	)
gr_scores = predict(model, Xgr, score="min")
scatter(Xgr[1,:], Xgr[2,:], marker_z=gr_scores)
scatter!(Xn[1,:], Xn[2,:], label="normal")
scatter!(Xa[1,:], Xa[2,:], label="anomalous")
savefig(joinpath(outdir, "grid_scores_min.png"))

