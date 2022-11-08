function inv_flow(c::Flux.Chain, x)
    inv_flow_chain(reverse(c.layers), x)
end

inv_flow_chain(::Tuple{}, x) = x
inv_flow_chain(fs::Tuple, x) = inv_flow_chain(Base.tail(fs), inv_flow(first(fs), x))

function inv_flow(bn::Flux.BatchNorm, yl)
    Y, logJ = yl
    X = bn.μ .+ sqrt.(bn.σ² .+ bn.ϵ).*(Y .- bn.β) ./ bn.γ
    X, logJ .- sum(log.(bn.γ)) .+ 0.5*sum(log.(bn.σ² .+ bn.ϵ))
end

#=
Implements Masked AutoEncoder for Density Estimation, by Germain et al. 2015
Re-implementation by Andrej Karpathy based on https://arxiv.org/abs/1502.03509
Re-Re-implementation by Tejan Karmali using Flux.jl ;)
Simplification for use in MaskedAutoregressiveFlows by Jan Francu.
=#

using Flux, Random
using Flux: glorot_uniform, zeros32, ones32
# ------------------------------------------------------------------------------

add_dims_r(a) = reshape(a, size(a)..., 1)
add_dims_l(a) = reshape(a, 1, size(a)...)

function funnel(c::Chain)
    isize = size(c[1].W, 2)
    bsize = [length(l.b) for l in c]
    osize = bsize[end]
    isize, bsize[1:end-1], osize
end

# ------------------------------------------------------------------------------

struct MaskedDense{F,S,T,M}
  # same as Linear except has a configurable mask on the weights
  W::S
  b::T
  mask::M
  σ::F
end

function MaskedDense(in::Integer, out::Integer, σ = identity)
  return MaskedDense(glorot_uniform(out, in), zeros32(out), ones32(out, in), σ)
end

Flux.@functor MaskedDense
Flux.trainable(m::MaskedDense) = (m.W, m.b)

function (a::MaskedDense)(x)
  a.σ.(a.mask .* a.W * x .+ a.b)
end

function Base.show(io::IO, l::MaskedDense)
  print(io, "MaskedDense(", size(l.W, 2), ", ", size(l.W, 1))
  l.σ == identity || print(io, ", ", l.σ)
  print(io, ")")
end
# ------------------------------------------------------------------------------

mutable struct MADE
  net::Chain
  m::Dict{Int, Vector{Int}}

  function MADE(
        isize::Int, 
        hsizes, 
        osize::Int, 
        ordering::String="sequential";
        ftype=relu,
        ptype=identity,
        rs=time_ns())
    
    # define a simple MLP neural net
    hsizes = push!([isize], hsizes...)
    layers = [MaskedDense(hsizes[i], hsizes[i + 1], ftype) for i = 1:length(hsizes) - 1]
    net = Chain(layers..., MaskedDense(hsizes[end], osize, ptype))

    made = new(net, Dict{Int, Vector{Int}}())
    update_masks!(made, ordering, rs)
    made
  end
end

Flux.@functor MADE
Flux.trainable(m::MADE) = (m.net, )

function update_masks!(made::MADE, ordering, seed)
    isize, hsizes, osize = funnel(made.net)
    L = length(hsizes)

    # make masks reproducible
    rng = MersenneTwister(seed)

    # sample the order of the inputs and the connectivity of all neurons
    made.m[0] = (ordering == "sequential") ? collect(1:isize) : (
        ordering == "reversed" ? collect(isize:-1:1) : randperm(rng, isize))
    for l = 1:L
        made.m[l] = rand(rng, minimum(made.m[l - 1]):isize - 1, hsizes[l])
    end

    # construct the mask matrices
    masks = [add_dims_r(made.m[l - 1]) .<= add_dims_l(made.m[l]) for l = 1:L]
    push!(masks, add_dims_r(made.m[L]) .< add_dims_l(made.m[0]))

    # this assumes that there are no other layers than MaskedDense
    for (l, m) in zip(made.net, masks)
        l.mask .= permutedims(m, [2, 1])
    end
end

function (made::MADE)(x)
  made.net(x)
end

function Base.show(io::IO, made::MADE)
  print(io, "MADE(")
  join(io, made.net, ", ")
  print(io, ")")
end

abstract type AbstractContinuousFlow end

struct MaskedAutoregressiveFlow <: AbstractContinuousFlow
	cα::MADE
	cβ::MADE
	bn::Union{BatchNorm, Nothing}
end

Flux.@functor(MaskedAutoregressiveFlow)

function MaskedAutoregressiveFlow(
		isize::Integer, 
		hsize::Integer, 
		nlayers::Integer, 
		osize::Integer,
		activations,
		ordering::String="sequential";
		lastlayer::String="linear",
		use_batchnorm::Bool=true,
		lastzero::Bool=true,
		seed=time_ns())
	m = MaskedAutoregressiveFlow(
		MADE(
			isize, 
			fill(hsize, nlayers-1), 
			osize, 
			ordering,
			ftype=activations.α,
			ptype=(lastlayer == "linear") ? identity : activations.α,
			rs=seed), 
		MADE(
			isize, 
			fill(hsize, nlayers-1), 
			osize, 
			ordering, 
			ftype=activations.β,
			ptype=(lastlayer == "linear") ? identity : activations.β,
			rs=seed),
		use_batchnorm ? BatchNorm(isize; momentum=1.0f0) : nothing)
	if lastzero
		m.cα.net[end].W .*= 0.0f0
		m.cβ.net[end].W .*= 0.0f0
	end
	m
end

function (maf::MaskedAutoregressiveFlow)(xl::Tuple)
	X, logJ = xl
	α, β = maf.cα(X), maf.cβ(X)
	# Y = exp.( 0.5 .* β) .* X .+ α
	Y = exp.(-0.5 .* β) .*(X .- α) # inv
	# logJy = logJ .+ 0.5 .* sum(β, dims = 1)
	logJy = logJ .- 0.5 .* sum(β, dims = 1) # inv
	
	if maf.bn !== nothing
		bn = maf.bn
		Z = bn(Y)
		# @info("", X, Y, Z, α, β, exp.(0.5 .* β), hcat(bn.μ, bn.β), hcat(bn.σ², bn.γ))
		logJz = logJy .+ sum(log.(bn.γ)) .- 0.5*sum(log.(bn.σ² .+ bn.ϵ))
		return Z, logJz
	end
	Y, logJy
end

function inv_flow(maf::MaskedAutoregressiveFlow, yl)
	Y, logJ = (maf.bn !== nothing) ? inv_flow(maf.bn, yl) : yl
	D, N = size(Y)
	perm = maf.cα.m[0]
	
	X = zeros(eltype(Y), 0, N)
	for (d, pd) in zip(1:D, perm)
		X_cond = vcat(X, zeros(eltype(Y), D - d + 1, N))[perm, :]
		α, β = maf.cα(X_cond)[pd:pd, :], maf.cβ(X_cond)[pd:pd, :]
		# X = vcat(X, exp.(-0.5 .* β) .* (Y[pd:pd, :] .- α))
		X = vcat(X,  exp.(0.5 .* β) .* Y[pd:pd, :] .+ α) # inv
		# logJ .-= 0.5 .* β
		logJ .+= 0.5 .* β # inv
	end
	
	X[perm, :], logJ
end

function Base.show(io::IO, maf::MaskedAutoregressiveFlow)
	print(io, "MaskedAutoregressiveFlow(cα=")
	print(io, maf.cα)
	print(io, ", cβ=")
	print(io, maf.cβ)
	(maf.bn !== nothing) && print(io, ", ", maf.bn)
	print(io, ")")
end

struct MAF
	chain
end

Flux.@functor MAF

function MAF(
	xdim::Int;    
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
    )
   Random.seed!(seed)

    m = Chain([
        MaskedAutoregressiveFlow(
            xdim, 
            hsize,
            num_layers, 
            xdim, 
            (α=eval(:($(Symbol(act_loc)))), β=eval(:($(Symbol(act_scl))))),
            (ordering == "natural") ? (
                (mod(i, 2) == 0) ? "reversed" : "sequential"
              ) : "random";
            seed=rand(UInt),
            use_batchnorm=bn,
            lastlayer=lastlayer,
            lastzero=lastzero
        ) 
        for i in 1:num_flows]...)
    Random.seed!()
    return MAF(m)
end

_init_logJ(x) = zeros(eltype(x), 1, size(x, 2))
function maf_loss(x, model::MAF, base) 
    z, logJ = model.chain((x, _init_logJ(x)))
    -sum(logpdf(base, z) .+ logJ)/size(x, 2)
end
StatsBase.predict(m::MAF, x::AbstractMatrix, base) = 
	map(_x->maf_loss(reshape(_x,2,1), m, base), eachcol(x))

