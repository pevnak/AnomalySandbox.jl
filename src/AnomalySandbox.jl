module AnomalySandbox

using Flux
using Random
using StatsBase
using GaussianMixtures
using ConditionalDists
using Distributions
using LinearAlgebra
using Distances
using DistributionsAD
using StatsBase: predict

using ConditionalDists: AbstractConditionalDistribution, ConditionalMvNormal, mean, BMN, condition
const ACD = AbstractConditionalDistribution
const TuMvNormal = Union{DistributionsAD.TuringDenseMvNormal,
                         DistributionsAD.TuringDiagMvNormal,
                         DistributionsAD.TuringScalMvNormal}

export VAE, GAN, MAF
export elbo, generator_loss, discriminator_loss
export predict, train!, softplus_safe

abstract type AbstractGM end
abstract type AbstractVAE <: AbstractGM end
abstract type AbstractGAN <: AbstractGM end

# to make Flux.gpu work on VAE/GAN/etc priors we need:
Flux.@functor DistributionsAD.TuringScalMvNormal
Flux.@functor DistributionsAD.TuringDiagMvNormal
Flux.@functor DistributionsAD.TuringDenseMvNormal

include("data.jl")
include("utils.jl")
include("vae.jl")
include("gan.jl")
include("gmm.jl")
include("maf.jl")
include("parzen.jl")
include("isolation_forest.jl")
include("pca.jl")

export train_test_split
end
