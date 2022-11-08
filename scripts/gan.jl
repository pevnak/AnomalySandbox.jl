using Plots
using AnomalySandbox
using AnomalySandbox: two_bananas, update_params!
using StatsBase
using Flux
using ConditionalDists

outdir = "gan_plots"
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

# construct the GAN
zdim = 1
xdim = size(Xn, 1)
hdim = 32
act = relu

disc = AnomalySandbox.stack_layers([xdim, hdim, hdim, 1], act, last=Flux.σ)
disc_dist = ConditionalMvNormal(disc)

gen = AnomalySandbox.stack_layers([zdim, hdim, hdim, xdim], act)
gen_dist = ConditionalMvNormal(gen)

model = GAN(zdim, gen_dist, disc_dist)

# now train it
batchsize = n
gen_opt = ADAM()
disc_opt = ADAM()
data = [(Xtr,) for i in 1:2000];
gloss(x) = generator_loss(model, batchsize) # gen_loss does not need input data
dloss(x) = discriminator_loss(model, x)
for _data in data
	update_params!(model.generator, _data, gloss, gen_opt)
	update_params!(model.discriminator, _data, dloss, disc_opt)
end
   
# get train, test scores
tr_scores = predict(model, Xtr)
tst_scores = predict(model, Xtst)
scatter(Xtst[1,:], Xtst[2,:], marker_z=tst_scores)
savefig(joinpath(outdir, "test_scores.png"))

# now plot scores on a grid
Xgr = Float32.(AnomalySandbox.get_2D_grid([-4,4], [-3,3], 0.1)	)
gr_scores = vec(predict(model, Xgr))
scatter(Xgr[1,:], Xgr[2,:], marker_z=gr_scores)
scatter!(Xn[1,:], Xn[2,:], label="normal")
scatter!(Xa[1,:], Xa[2,:], label="anomalous")
savefig(joinpath(outdir, "grid_scores.png"))
