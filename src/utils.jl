# for over/underflow in logs
"""
	softplus_safe(x,T=Float32)
A softplus with small additive constant for safe operations.
"""
softplus_safe(x,T=Float32) = softplus(x) .+ T(1e-6)

### Flux chain builders ###
"""
    function layer_builder(d,k,o,n,ftype,lastlayer = "",ltype = "Dense")
Create a chain with `n` layers of with `k` neurons with activation function `ftype`.
Input and output dimension is `d` / `o`.
If lastlayer is not specified, all layers use the same function.
If lastlayer is "linear", then the last layer is forced to be Dense with linear activation.
It is also possible to specify dimensions in a vector.
"""
layer_builder(k::Vector{Int},l::Vector,f::Vector) =
    Flux.Chain(map(i -> i[1](i[3],i[4],i[2]),zip(l,f,k[1:end-1],k[2:end]))...)

layer_builder(d::Int,k::Int,o::Int,n::Int, args...) =
    layer_builder(vcat(d,fill(k,n-1)...,o), args...)

function layer_builder(ks::Vector{Int},ftype::String,lastlayer::String = "",ltype::String = "Dense")
    ftype = (ftype == "linear") ? "identity" : ftype
    ls = Array{Any}(fill(eval(:($(Symbol(ltype)))),length(ks)-1))
    fs = Array{Any}(fill(eval(:($(Symbol(ftype)))),length(ks)-1))
    if !isempty(lastlayer)
        fs[end] = (lastlayer == "linear") ? identity : eval(:($(Symbol(lastlayer))))
        ls[end] = (lastlayer == "linear") ? Dense : ls[end]
    end
    layer_builder(ks,ls,fs)
end

"""
    stack_layers(lsize, activation, [layer=Dense; last=identity])
Construct a Flux chain (e.g. encoder/decoder) consisting of `length(lsize)-1` layers of 
type `layer` with  `activation` in between. Default last activation is `identity`. 
Width of layers is defined by the vector `lsize`.
"""
stack_layers(lsize::Union{Vector, Tuple}, activation, layer=Dense; last=identity) =  
    layer_builder([x for x in lsize], 
        Array{Any}(fill(layer, size(lsize,1)-1)), 
        Array{Any}([fill(activation, size(lsize,1)-2); last]))


### training ###
"""
    update_params!(model, data, loss, opt)
Basic training step - computation of the loss, backpropagation of gradients and optimisation 
of weights. The loss and opt arguments can be arrays/lists/tuples.
"""
function update_params!(model, data, loss, opt)
    ps = Flux.params(model)
    gs = gradient(ps) do
        loss(data...)
    end
    Flux.Optimise.update!(opt, ps, gs)
end 

"""
    train!(model, data, loss, optimiser, callback; [usegpu, memory_efficient])
Train the model. Function callback(model, batch, loss, opt) is 
called every iteration - use it to store or print training progress, stop training etc. 
"""
function train!(model, data, loss, optimiser, callback; 
    usegpu = false, memory_efficient = false)
    for _data in data
        try
            if usegpu
             _data = _data |> gpu
            end
            update_params!(model, _data, loss, optimiser)
            # now call the callback function
            # can be an object so it can store some values between individual calls
            callback(model, _data, loss, optimiser)
        catch e
            # setup a special kind of exception for known cases with a break
            rethrow(e)
        end
        if memory_efficient
            # this might be important for conv nets running on gpu
            # large nets might still train but slowly
            GC.gc();
        end
    end
end

### GAN ###
"""
    gen_loss(sg::AbstractArray)
Generator loss for generated data score `sg`.
"""
function generator_loss(sg::AbstractArray{T}) where T<:Real
    - mean(log.(sg) .+ eps(T))
end

"""
    disc_loss(st::AbstractArray, sg::AbstractArray)
Discriminator loss for true data score `st` and generated data score `sg`.
"""
function discriminator_loss(st::AbstractArray{T}, sg::AbstractArray{T}) where T<:Real
    - T(0.5)*(mean(log.(st .+ eps(T))) + mean(log.(1 .- sg) .+ eps(T)))
end

### KLD ###
function kl_divergence(p::BMN, q::TuMvNormal)
    (μ₁, σ₁) = Flux.mean(p), Flux.var(p)
    (μ₂, σ₂) = Flux.mean(q), Flux.var(q)
    _kld_gaussian1(μ₁, σ₁, μ₂, σ₂)
end

function kl_divergence(p::BMN, q::BMN)
    (μ₁, σ₁) = Flux.mean(p), Flux.var(p)
    (μ₂, σ₂) = Flux.mean(q), Flux.var(q)
    _kld_gaussian1(μ₁, σ₁, μ₂, σ₂)
end


function _kld_gaussian1(μ1::AbstractArray, σ1::AbstractArray, μ2::AbstractArray, σ2::AbstractArray)
    k  = size(μ1, 1)
    m1 = sum(σ1 ./ σ2, dims=1)
    m2 = sum((μ2 .- μ1).^2 ./ σ2, dims=1)
    m3 = sum(log.(σ2 ./ σ1), dims=1)
    (m1 .+ m2 .+ m3 .- k) ./ 2
end

function train_test_split(xn, xa)
    xn = Float32.(xn)
    xa = Float32.(xa)
    n = size(xn,2) ÷ 2
    na = size(xa, 2)
    ninds = randperm(2*n)
    xtrn = xn[:,ninds[1:n]]
    xtst = hcat(xn[:,ninds[n+1:end]], xa)
    ytst = vcat(zeros(Int,n), ones(Int,na))
    xtrn, (xtst, ytst)
end

train_test_split(xn, xa, grid) = (train_test_split(xn, xa)..., grid)